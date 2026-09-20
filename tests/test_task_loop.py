#!/usr/bin/env python3
"""Self-evolution repair loop: API auth/params, script CRLF, SOP reuse."""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from agent.protocol import Reasoning
from agent.task_loop import (
    ANALYZING, FAILED_TERMINAL, LEARNING, MAX_REQUEST_ATTEMPTS, RETRYING,
    advance_execution, apply_skill, buildRetryRequest, classifyExecutionError,
    detectScriptFormat, extract_credential, generateOrUpdateSkill,
    parse_http_command, redact, repairAuthentication, repairExecutePermission,
    repairLineEndings, repairLineEndingsFile, repairRequestParameters,
    request_fingerprint, run_closed_loop, script_repair_command,
    validateInterpreter, validateTaskResult,
)


SECRET = "test-secret-abc"
WEATHER_URL = "http://127.0.0.1/weather"
BEIJING_TASK = "请查询北京天气"
SHANGHAI_TASK = "请查询上海天气"
GUANGZHOU_TASK = "请查询广州天气"


def stale_docs_command(city):
    return (
        'curl -sS -X GET -H "x-api-key: %s" "%s?city=%s"'
        % (SECRET, WEATHER_URL, city)
    )


class FakeWeather:
    def __init__(self):
        self.calls = []

    def __call__(self, command):
        self.calls.append(command)
        spec = parse_http_command(command) or {}
        headers = {str(name).lower(): value for name, value in (spec.get("headers") or {}).items()}
        params = spec.get("params") or {}
        if "authorization" not in headers:
            return "[exitCode:1]\nMissing 'Authorization' header"
        if "location" not in params:
            return "[exitCode:1]\nMissing required parameter: location"
        city = params["location"]
        return '[exitCode:0]\n{"weather":"晴","city":"%s","temp":20}' % city


def task_snapshot(round_no, task, last="", llm="", errors=None):
    return {
        "roundNo": round_no,
        "phaseTask": task,
        "lastCmdResult": last,
        "llmResp": llm,
        "errors": errors or [],
        "lastRoundRoleActionResults": {"2": True},
        "teamOur": {
            "type": "red",
            "goldNum": 0,
            "roles": [{
                "id": "2", "pos": {"x": 5, "y": 5}, "health": 200,
                "roleType": "pioneer", "backpack": [],
            }],
            "playerTasks": [{
                "taskType": "自进化类1",
                "taskPosition": {"x": 5, "y": 5},
                "timeoutRounds": 40,
            }],
        },
        "mapInfo": {"width": 40, "height": 40, "zones": []},
        "weaponShopList": [],
        "vendorShopList": [],
    }


class WeatherRepairTests(unittest.TestCase):
    def test_docs_x_api_key_classified_as_missing_authorization(self):
        error = classifyExecutionError(
            stale_docs_command("北京"),
            "[exitCode:1]\nMissing 'Authorization' header",
        )
        self.assertEqual(error["type"], "MISSING_AUTH_HEADER")
        self.assertTrue(error["recoverable"])

    def test_auth_repair_then_request_succeeds(self):
        server = FakeWeather()
        result = run_closed_loop(stale_docs_command("北京"), server, task=BEIJING_TASK)
        self.assertEqual(result["state"], LEARNING)
        self.assertIn("北京", result["taskAnswer"])
        self.assertGreaterEqual(len(server.calls), 3)
        final = parse_http_command(server.calls[-1])
        self.assertIn("authorization", {name.lower() for name in final["headers"]})
        self.assertNotIn("x-api-key", {name.lower() for name in final["headers"]})

    def test_credentials_are_redacted_in_logs(self):
        spec = parse_http_command(stale_docs_command("北京"))
        dumped = redact(json.dumps({"command": stale_docs_command("北京"), "headers": spec["headers"]},
                                   ensure_ascii=False))
        self.assertNotIn(SECRET, dumped)
        self.assertIn("<redacted>", dumped)

    def test_city_parameter_classified_as_missing_location(self):
        spec = repairAuthentication(
            parse_http_command(stale_docs_command("北京")),
            {"type": "MISSING_AUTH_HEADER"}, SECRET,
        )
        error = classifyExecutionError(
            buildRetryRequest(spec),
            "[exitCode:1]\nMissing required parameter: location",
        )
        self.assertEqual(error["type"], "MISSING_PARAMETER")
        self.assertEqual(error["name"], "location")

    def test_city_mapped_to_location_keeps_beijing(self):
        spec = parse_http_command(stale_docs_command("北京"))
        spec = repairAuthentication(spec, {"type": "MISSING_AUTH_HEADER"}, SECRET)
        repaired = repairRequestParameters(spec, {"type": "MISSING_PARAMETER", "name": "location"})
        self.assertEqual(repaired["params"]["location"], "北京")
        self.assertNotIn("city", repaired["params"])
        self.assertEqual(repaired["param_map"]["city"], "location")

    def test_parameter_repair_reused_for_other_cities(self):
        for city, task in (("上海", SHANGHAI_TASK), ("广州", GUANGZHOU_TASK)):
            server = FakeWeather()
            result = run_closed_loop(stale_docs_command(city), server, task=task)
            self.assertEqual(result["state"], LEARNING, city)
            self.assertEqual(parse_http_command(server.calls[-1])["params"]["location"], city)
            self.assertIn(city, result["taskAnswer"])

    def test_identical_failed_request_is_not_repeated(self):
        command = stale_docs_command("北京")
        session = {
            "attempts": [], "failed": set(), "credential": SECRET,
            "skill": None, "repair_count": 0,
        }
        first = advance_execution(
            command, "[exitCode:1]\nMissing 'Authorization' header",
            session=session, task=BEIJING_TASK,
        )
        self.assertEqual(first["state"], RETRYING)
        session = first["session"]
        session["failed"].add(request_fingerprint(first["spec"]))
        second = advance_execution(
            first["executeCmd"], "[exitCode:1]\nMissing 'Authorization' header",
            session=session, task=BEIJING_TASK,
        )
        if second["state"] == RETRYING:
            session = second["session"]
            session["failed"].add(request_fingerprint(second["spec"]))
            third = advance_execution(
                second["executeCmd"], "[exitCode:1]\nMissing 'Authorization' header",
                session=session, task=BEIJING_TASK,
            )
            self.assertEqual(third["state"], FAILED_TERMINAL)
            self.assertEqual(third["reason"], "duplicate_request")
        else:
            self.assertEqual(second["state"], FAILED_TERMINAL)

    def test_max_attempts_stops_safely(self):
        def stubborn(_command):
            return "[exitCode:1]\nMissing required parameter: field%s" % len(stubborn.calls)

        stubborn.calls = []

        def executor(command):
            stubborn.calls.append(command)
            return stubborn(command)

        result = run_closed_loop(
            stale_docs_command("北京"), executor, task=BEIJING_TASK,
            credential=SECRET, max_attempts=MAX_REQUEST_ATTEMPTS,
        )
        self.assertEqual(result["state"], FAILED_TERMINAL)
        self.assertEqual(result["reason"], "max_attempts")
        self.assertLessEqual(len(stubborn.calls), MAX_REQUEST_ATTEMPTS)

    def test_missing_credential_is_not_invented(self):
        spec = parse_http_command('curl -sS "%s?city=北京"' % WEATHER_URL)
        self.assertIsNone(extract_credential(spec))
        self.assertIsNone(repairAuthentication(spec, {"type": "MISSING_AUTH_HEADER"}, None))
        result = advance_execution(
            'curl -sS "%s?city=北京"' % WEATHER_URL,
            "[exitCode:1]\nMissing 'Authorization' header",
            session={"attempts": [], "failed": set(), "credential": None, "repair_count": 0},
            task=BEIJING_TASK,
        )
        self.assertEqual(result["state"], ANALYZING)
        self.assertEqual(result["reason"], "missing_credential")
        self.assertFalse(result["executeCmd"])
        self.assertNotIn("Authorization: ", result.get("executeCmd") or "")


class ScriptRepairTests(unittest.TestCase):
    def test_bad_interpreter_from_crlf(self):
        error = classifyExecutionError(
            "./check",
            "[exitCode:126]\nbash: ./check: /bin/bash^M: bad interpreter: No such file or directory",
        )
        self.assertIn(error["type"], {"BAD_INTERPRETER", "CRLF_SHEBANG"})
        self.assertTrue(error["recoverable"])

    def test_crlf_converted_to_lf(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "check"
            path.write_bytes(b"#!/bin/bash\r\necho ok\r\n")
            original, repaired = repairLineEndingsFile("check", root=folder)
            self.assertIn(b"\r\n", original)
            self.assertNotIn(b"\r", repaired)
            self.assertEqual(repaired, b"#!/bin/bash\necho ok\n")

    def test_missing_execute_permission_repair(self):
        self.assertEqual(repairExecutePermission("./check"), "chmod +x ./check")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "check"
            path.write_bytes(b"#!/bin/bash\necho ok\n")
            path.chmod(0o644)
            subprocess.check_call(["bash", "-lc", "chmod +x ./check"], cwd=folder)
            self.assertTrue(os.access(path, os.X_OK))

    def test_shebang_cr_detected_and_fixed(self):
        info = detectScriptFormat(b"#!/bin/bash\r\necho ok\n")
        self.assertTrue(info["shebang_has_cr"])
        self.assertTrue(info["crlf"])
        cleaned = repairLineEndings(b"#!/bin/bash\r\necho ok\n")
        self.assertFalse(detectScriptFormat(cleaned)["shebang_has_cr"])

    def test_missing_interpreter_falls_back_to_bash(self):
        info = validateInterpreter("#!/usr/bin/definitely-not-real-interp")
        self.assertFalse(info["ok"])
        self.assertEqual(info["fallback"], "bash")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "check"
            path.write_bytes(b"#!/usr/bin/definitely-not-real-interp\necho recovered\n")
            path.chmod(0o755)
            proc = subprocess.run(
                ["bash", "-lc", script_repair_command("./check")],
                cwd=folder, capture_output=True, text=True,
            )
            self.assertEqual(proc.returncode, 0)
            self.assertIn("recovered", proc.stdout)

    def test_script_is_rerun_after_repair(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "check"
            path.write_bytes(b"#!/bin/bash\r\necho weather-ok\r\n")
            path.chmod(0o644)

            def executor(command):
                if command.strip() == "./check":
                    return "[exitCode:126]\nbash: ./check: /bin/bash^M: bad interpreter: No such file or directory"
                proc = subprocess.run(
                    ["bash", "-lc", command], cwd=folder, capture_output=True, text=True,
                )
                return "[exitCode:%s]\n%s" % (proc.returncode, proc.stdout + proc.stderr)

            result = run_closed_loop("./check", executor, task="运行 ./check 并输出结果")
            self.assertEqual(result["state"], LEARNING)
            self.assertTrue(path.read_bytes().find(b"\r") < 0)
            self.assertTrue(os.access(path, os.X_OK))

    def test_new_error_after_first_repair_keeps_diagnosing(self):
        session = {"attempts": [], "failed": set(), "repair_count": 0}
        first = advance_execution(
            "./check",
            "[exitCode:126]\nbash: ./check: /bin/bash^M: bad interpreter: No such file or directory",
            session=session, task="运行 ./check",
        )
        self.assertEqual(first["state"], RETRYING)
        second = advance_execution(
            first["executeCmd"],
            "[exitCode:126]\n./check: Permission denied",
            session=first["session"], task="运行 ./check",
        )
        self.assertEqual(second["state"], RETRYING)
        self.assertEqual(second["repair"], "repair_execute_permission")
        self.assertIn("chmod +x", second["executeCmd"])

    def test_refuses_to_rewrite_outside_task_dir(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                repairLineEndingsFile("/etc/hosts", root=folder)


class SkillReuseTests(unittest.TestCase):
    def test_first_weather_task_writes_skill(self):
        result = run_closed_loop(stale_docs_command("北京"), FakeWeather(), task=BEIJING_TASK)
        skill = result["skill"]
        self.assertEqual(skill["auth_header"], "Authorization")
        self.assertEqual(skill["param_map"]["city"], "location")
        self.assertNotIn(SECRET, json.dumps(skill))
        self.assertTrue(validateTaskResult(
            '[exitCode:0]\n{"weather":"晴","city":"北京","temp":20}', BEIJING_TASK))

    def test_later_cities_reuse_skill(self):
        first = run_closed_loop(stale_docs_command("北京"), FakeWeather(), task=BEIJING_TASK)
        skill = first["skill"]
        for city, task in (("上海", SHANGHAI_TASK), ("广州", GUANGZHOU_TASK)):
            spec = apply_skill(skill, task, credential=SECRET)
            self.assertEqual(spec["params"]["location"], city)
            self.assertEqual(list(spec["headers"])[0].lower(), "authorization")
            server = FakeWeather()
            result = run_closed_loop(buildRetryRequest(spec), server, task=task, skill=skill,
                                     credential=SECRET)
            self.assertEqual(result["state"], LEARNING)
            self.assertEqual(len(server.calls), 1)

    def test_stale_skill_is_updated_when_api_changed(self):
        stale = generateOrUpdateSkill(None, {
            "method": "GET", "url": WEATHER_URL,
            "headers": {"x-api-key": SECRET},
            "params": {"city": "北京"},
        }, [])
        server = FakeWeather()
        reused = apply_skill({**stale, "param_names": ["city"], "param_map": {}},
                             SHANGHAI_TASK, credential=SECRET)
        result = run_closed_loop(buildRetryRequest(reused), server, task=SHANGHAI_TASK, skill=stale,
                                 credential=SECRET)
        self.assertEqual(result["state"], LEARNING)
        self.assertEqual(result["skill"]["param_map"]["city"], "location")
        self.assertEqual(parse_http_command(server.calls[-1])["params"]["location"], "上海")

    def test_only_validated_plan_enters_stable_skill(self):
        failed_spec = parse_http_command(stale_docs_command("北京"))
        skill = generateOrUpdateSkill(None, None, [{
            "error_type": "MISSING_AUTH_HEADER", "param_names": ["city"],
        }])
        self.assertNotEqual(skill.get("param_names"), ["city"])
        success = parse_http_command(
            buildRetryRequest(repairRequestParameters(
                repairAuthentication(failed_spec, {"type": "MISSING_AUTH_HEADER"}, SECRET),
                {"type": "MISSING_PARAMETER", "name": "location"},
            ))
        )
        skill = generateOrUpdateSkill(None, success, [
            {"error_type": "MISSING_AUTH_HEADER"},
            {"error_type": "MISSING_PARAMETER"},
        ])
        self.assertEqual(skill["param_names"], ["location"])
        self.assertEqual(skill["auth_header"], "Authorization")


class ProtocolIntegrationTests(unittest.TestCase):
    def test_reasoning_repairs_without_extra_llm_round(self):
        brain = Reasoning({"enable_llm": True, "enable_treasure": False})
        first = brain.update(task_snapshot(10, BEIJING_TASK), 1, "2")
        self.assertTrue(first["prompt"])
        self.assertFalse(first["executeCmd"])
        llm = json.dumps({"kind": "command", "command": stale_docs_command("北京")}, ensure_ascii=False)
        issued = brain.update(task_snapshot(11, BEIJING_TASK, llm=llm), 1, "2")
        self.assertIn("x-api-key", issued["executeCmd"])
        repaired = brain.update(task_snapshot(
            12, BEIJING_TASK, last="[exitCode:1]\nMissing 'Authorization' header"), 1, "2")
        self.assertTrue(repaired["executeCmd"])
        self.assertFalse(repaired["prompt"])
        self.assertIn("Authorization", repaired["executeCmd"])
        self.assertNotIn("x-api-key", repaired["executeCmd"].lower())
        trace = json.dumps(brain._trace, ensure_ascii=False)
        self.assertNotIn(SECRET, trace)
        mapped = brain.update(task_snapshot(
            13, BEIJING_TASK, last="[exitCode:1]\nMissing required parameter: location"), 1, "2")
        self.assertIn("location=", mapped["executeCmd"])
        self.assertEqual(parse_http_command(mapped["executeCmd"])["params"].get("location"), "北京")
        done = brain.update(task_snapshot(
            14, BEIJING_TASK,
            last='[exitCode:0]\n{"weather":"晴","city":"北京","temp":20}'), 1, "2")
        self.assertTrue(done["taskAnswer"])
        self.assertIn("北京", done["taskAnswer"])
        self.assertIn("weather", json.dumps(brain._skills, ensure_ascii=False))
        shanghai = brain.update(task_snapshot(20, SHANGHAI_TASK), 1, "2")
        self.assertTrue(shanghai["executeCmd"])
        self.assertEqual(parse_http_command(shanghai["executeCmd"])["params"].get("location"), "上海")
        self.assertFalse(shanghai["prompt"])


if __name__ == "__main__":
    unittest.main()
