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
    ANALYZING, ANSWER_READY, API_CATEGORY, COMPLETED, FAILED_TERMINAL, FILE_CATEGORY,
    LEARNING, MAX_REQUEST_ATTEMPTS, RETRYING, TERMINAL_FAILED, TaskLedger,
    advance_execution, apply_skill, buildRetryRequest, canonicalize_api_spec,
    claimTaskCategoryOnce, classifyExecutionError, detectScriptFormat,
    executeAllTasksInCategory, extract_city, extract_credential,
    generateOrUpdateSkill, loadClaimedTasks, normalizeTypes, parse_http_command,
    persistTaskProgress, redact, repairAuthentication, repairExecutePermission,
    repairLineEndings, repairLineEndingsFile, repairRequestParameters,
    request_fingerprint, run_closed_loop, script_repair_command,
    spec_sends_both_auth_headers, spec_sends_both_city_params,
    submitPreparedAnswer, validateInterpreter, validateTaskResult,
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


HERITAGE_URL = "http://127.0.0.1/heritage"
API_TITLES = [
    "请查询北京的文化遗产",
    "请查询上海的文化遗产",
    "请查询广州的文化遗产",
]
FILE_TITLES = [
    "修复 ./check1 并提交 Token",
    "修复 ./check2 并提交 Token",
    "修复 ./check3 并提交 Token",
]


def heritage_stale_command(city):
    return (
        'curl -sS -X GET -H "x-api-key: %s" "%s?city=%s"'
        % (SECRET, HERITAGE_URL, city)
    )


class FakeHeritage:
    def __init__(self):
        self.calls = []

    def __call__(self, command):
        self.calls.append(command)
        spec = parse_http_command(command) or {}
        headers = {str(name).lower(): value for name, value in (spec.get("headers") or {}).items()}
        params = spec.get("params") or {}
        auth_names = [name for name in headers if name in ("authorization", "x-api-key", "api-key", "api_key", "token")]
        city_names = [name for name in params if str(name).lower() in ("city", "location")]
        if len(auth_names) > 1:
            return "[exitCode:1]\nHTTP 400 both auth headers"
        if len(city_names) > 1:
            return "[exitCode:1]\nHTTP 400 both city parameters"
        if "authorization" not in headers:
            return "[exitCode:1]\nMissing 'Authorization' header"
        if "location" not in params:
            return "[exitCode:1]\nMissing required parameter: location"
        city = params["location"]
        catalog = {
            "北京": ["古建筑", "石窟寺", "古建筑", " 传统技艺 ", "石窟寺", None, ""],
            "上海": ["古建筑", " 古建筑 ", "石窟寺"],
            "广州": ["石窟寺", "石窟寺"],
        }
        types = catalog.get(city, ["古建筑"])
        items = [{"type": item} for item in types if item]
        body = {
            "code": 200,
            "city": city,
            "items": items,
            "types": types,
            "next": "/page2",
        }
        return "[exitCode:0]\n" + json.dumps(body, ensure_ascii=False)


class EvolutionLoopTests(unittest.TestCase):
    def test_api_category_claimed_three_times(self):
        ledger = TaskLedger()
        first = claimTaskCategoryOnce(ledger, "自进化类1", API_CATEGORY, API_TITLES)
        second = claimTaskCategoryOnce(ledger, "自进化类1", API_CATEGORY, API_TITLES)
        self.assertEqual(ledger.claim_count[API_CATEGORY], 3)
        self.assertEqual(ledger.total_claims(), 3)
        self.assertTrue(ledger.api_claimed)
        self.assertEqual(len(first), 3)
        self.assertEqual(len(second), 3)
        self.assertEqual(len(loadClaimedTasks(ledger, API_CATEGORY)), 3)

    def test_file_category_claimed_three_times(self):
        ledger = TaskLedger()
        first = claimTaskCategoryOnce(ledger, "自进化类2", FILE_CATEGORY, FILE_TITLES)
        claimTaskCategoryOnce(ledger, "自进化类2", FILE_CATEGORY, FILE_TITLES)
        self.assertEqual(ledger.claim_count[FILE_CATEGORY], 3)
        self.assertEqual(len(first), 3)
        self.assertEqual(len(loadClaimedTasks(ledger, FILE_CATEGORY)), 3)

    def test_evolution_finishes_only_after_six_completions(self):
        ledger = TaskLedger()
        claimTaskCategoryOnce(ledger, "自进化类1", API_CATEGORY, API_TITLES)
        claimTaskCategoryOnce(ledger, "自进化类2", FILE_CATEGORY, FILE_TITLES)
        for item in loadClaimedTasks(ledger, API_CATEGORY):
            item["state"] = COMPLETED
            item["submitted"] = True
        self.assertFalse(ledger.evolution_complete())
        self.assertEqual(ledger.completed_count(API_CATEGORY), 3)
        self.assertEqual(ledger.completed_count(FILE_CATEGORY), 0)
        for item in loadClaimedTasks(ledger, FILE_CATEGORY):
            item["state"] = COMPLETED
            item["submitted"] = True
        self.assertTrue(ledger.evolution_complete())
        self.assertEqual(ledger.completed_count(), 6)

    def test_resume_does_not_reclaim(self):
        ledger = TaskLedger()
        claimTaskCategoryOnce(ledger, "自进化类1", API_CATEGORY, API_TITLES)
        snapshot = persistTaskProgress(ledger)
        restored = TaskLedger()
        restored.claimed_families = dict(snapshot["claimed_families"])
        restored.claim_count = dict(snapshot["claim_count"])
        restored.api_claimed = snapshot["api_claimed"]
        restored.file_claimed = snapshot["file_claimed"]
        restored.tasks = dict(snapshot["tasks"])
        restored.order = list(snapshot["order"])
        claimTaskCategoryOnce(restored, "自进化类1", API_CATEGORY, API_TITLES)
        self.assertEqual(restored.claim_count[API_CATEGORY], 3)
        self.assertEqual(len(loadClaimedTasks(restored, API_CATEGORY)), 3)

    def test_completed_task_is_not_rerun(self):
        ledger = TaskLedger()
        claimTaskCategoryOnce(ledger, "自进化类1", API_CATEGORY, API_TITLES)
        queries = []
        submits = []
        loadClaimedTasks(ledger, API_CATEGORY)[0].update({
            "state": COMPLETED, "submitted": True, "query_done": True, "answer": "{}",
        })

        def query(record, command):
            queries.append(record["id"])
            return FakeHeritage()(command or heritage_stale_command("上海"))

        executeAllTasksInCategory(ledger, API_CATEGORY, query, lambda ans: submits.append(ans) or True,
                                  credential=SECRET)
        self.assertNotIn(loadClaimedTasks(ledger, API_CATEGORY)[0]["id"], queries)

    def test_request_never_sends_both_auth_headers(self):
        spec = canonicalize_api_spec({
            "method": "GET", "url": HERITAGE_URL,
            "headers": {"X-API-Key": SECRET, "Authorization": SECRET},
            "params": {"location": "北京"},
        })
        self.assertFalse(spec_sends_both_auth_headers(spec))
        command = buildRetryRequest(spec)
        parsed = parse_http_command(command)
        names = {name.lower() for name in parsed["headers"]}
        self.assertIn("authorization", names)
        self.assertNotIn("x-api-key", names)

    def test_request_never_sends_both_city_params(self):
        spec = canonicalize_api_spec({
            "method": "GET", "url": HERITAGE_URL,
            "headers": {"Authorization": SECRET},
            "params": {"city": "北京", "location": "北京"},
        })
        self.assertFalse(spec_sends_both_city_params(spec))
        self.assertEqual(spec["params"], {"location": "北京"})

    def test_401_switches_to_authorization_then_succeeds(self):
        server = FakeHeritage()
        result = run_closed_loop(heritage_stale_command("北京"), server, task=API_TITLES[0],
                                 credential=SECRET)
        self.assertEqual(result["state"], LEARNING)
        final = parse_http_command(server.calls[-1])
        self.assertIn("authorization", {name.lower() for name in final["headers"]})
        self.assertNotIn("x-api-key", {name.lower() for name in final["headers"]})

    def test_400_maps_city_to_location_then_succeeds(self):
        server = FakeHeritage()
        result = run_closed_loop(heritage_stale_command("北京"), server, task=API_TITLES[0],
                                 credential=SECRET)
        self.assertEqual(result["state"], LEARNING)
        self.assertEqual(parse_http_command(server.calls[-1])["params"].get("location"), "北京")
        self.assertNotIn("city", parse_http_command(server.calls[-1])["params"])

    def test_verified_contract_reused_for_later_cities(self):
        first = run_closed_loop(heritage_stale_command("北京"), FakeHeritage(),
                                task=API_TITLES[0], credential=SECRET)
        skill = first["skill"]
        self.assertEqual(skill["auth_header"], "Authorization")
        for city, title in (("上海", API_TITLES[1]), ("广州", API_TITLES[2])):
            spec = apply_skill(skill, title, credential=SECRET)
            self.assertFalse(spec_sends_both_auth_headers(spec))
            self.assertFalse(spec_sends_both_city_params(spec))
            server = FakeHeritage()
            result = run_closed_loop(buildRetryRequest(spec), server, task=title,
                                     skill=skill, credential=SECRET)
            self.assertEqual(result["state"], LEARNING)
            self.assertEqual(len(server.calls), 1)
            self.assertEqual(parse_http_command(server.calls[0])["params"]["location"], city)

    def test_types_are_deduped(self):
        types = normalizeTypes(["古建筑", "石窟寺", "古建筑", "石窟寺"])
        self.assertEqual(types, ["古建筑", "石窟寺"])

    def test_types_are_stably_sorted(self):
        types = normalizeTypes(["古建筑", "石窟寺", "传统技艺"])
        self.assertEqual(types, ["传统技艺", "古建筑", "石窟寺"])
        self.assertEqual(types, sorted(types))

    def test_types_strip_and_drop_empty(self):
        types = normalizeTypes(["古建筑", "石窟寺", "古建筑", " 传统技艺 ", "石窟寺", None, "  "])
        self.assertEqual(types, ["传统技艺", "古建筑", "石窟寺"])

    def test_http_200_does_not_paginate_or_retry_params(self):
        server = FakeHeritage()
        result = run_closed_loop(heritage_stale_command("北京"), server, task=API_TITLES[0],
                                 credential=SECRET)
        self.assertTrue(result.get("stop_query") or result["state"] == LEARNING)
        success_calls = [cmd for cmd in server.calls
                         if "Authorization" in cmd and "location=" in cmd]
        self.assertEqual(len(success_calls), 1)

    def test_http_200_immediately_builds_and_submits_answer(self):
        result = run_closed_loop(heritage_stale_command("北京"), FakeHeritage(),
                                 task=API_TITLES[0], credential=SECRET)
        payload = json.loads(result["taskAnswer"])
        self.assertEqual(payload["types"], ["传统技艺", "古建筑", "石窟寺"])
        self.assertEqual(payload["city"], "北京")

    def test_submit_failure_retries_without_requery(self):
        ledger = TaskLedger()
        claimTaskCategoryOnce(ledger, "自进化类1", API_CATEGORY, [API_TITLES[0]])
        queries = []
        server = FakeHeritage()

        def query(record, command):
            queries.append(command)
            return server(command or heritage_stale_command("北京"))

        submits = []

        def submit(answer):
            submits.append(answer)
            return len(submits) >= 2

        executeAllTasksInCategory(ledger, API_CATEGORY, query, submit, credential=SECRET)
        record = loadClaimedTasks(ledger, API_CATEGORY)[0]
        self.assertTrue(record.get("query_done"))
        query_count = len(queries)
        submitPreparedAnswer(record, submit)
        self.assertEqual(len(queries), query_count)
        self.assertEqual(record["state"], COMPLETED)

    def test_submit_success_ends_current_subtask(self):
        record = {"state": ANSWER_READY, "answer": '{"types":["古建筑"]}', "submitted": False}
        outcome = submitPreparedAnswer(record, lambda _ans: True)
        self.assertTrue(outcome["ok"])
        self.assertEqual(record["state"], COMPLETED)

    def test_file_token_is_submitted_immediately(self):
        output = "[exitCode:0]\nTOKEN=file-token-xyz"
        action = advance_execution("./check", output, task=FILE_TITLES[0])
        self.assertEqual(action["taskAnswer"], "file-token-xyz")
        self.assertTrue(action.get("stop_query"))
        record = {"state": ANSWER_READY, "answer": action["taskAnswer"]}
        submitPreparedAnswer(record, lambda ans: ans == "file-token-xyz")
        self.assertEqual(record["state"], COMPLETED)

    def test_token_submit_failure_does_not_rerun_repair(self):
        repairs = []

        def query(record, command):
            repairs.append(command)
            return "[exitCode:0]\nTOKEN=keep-me"

        ledger = TaskLedger()
        claimTaskCategoryOnce(ledger, "自进化类2", FILE_CATEGORY, [FILE_TITLES[0]])
        executeAllTasksInCategory(ledger, FILE_CATEGORY, query, lambda _ans: False)
        record = loadClaimedTasks(ledger, FILE_CATEGORY)[0]
        repair_count = len(repairs)
        submitPreparedAnswer(record, lambda _ans: False)
        self.assertEqual(len(repairs), repair_count)
        self.assertEqual(record.get("answer"), "keep-me")

    def test_one_failure_does_not_drop_queued_tasks(self):
        ledger = TaskLedger()
        claimTaskCategoryOnce(ledger, "自进化类1", API_CATEGORY, API_TITLES)
        loadClaimedTasks(ledger, API_CATEGORY)[0].update({
            "state": TERMINAL_FAILED, "error": "timeout",
        })
        remaining = loadClaimedTasks(ledger, API_CATEGORY)
        self.assertEqual(len(remaining), 3)
        self.assertEqual(sum(1 for item in remaining if item["state"] != TERMINAL_FAILED), 2)
        claimTaskCategoryOnce(ledger, "自进化类1", API_CATEGORY, API_TITLES)
        self.assertEqual(len(loadClaimedTasks(ledger, API_CATEGORY)), 3)

    def test_completion_counts_are_three_and_three(self):
        ledger = TaskLedger()
        claimTaskCategoryOnce(ledger, "自进化类1", API_CATEGORY, API_TITLES)
        claimTaskCategoryOnce(ledger, "自进化类2", FILE_CATEGORY, FILE_TITLES)
        server = FakeHeritage()
        executeAllTasksInCategory(
            ledger, API_CATEGORY,
            lambda record, command: server(command or heritage_stale_command(extract_city(record["title"]))),
            lambda _ans: True, credential=SECRET,
        )
        tokens = {"check1": "t1", "check2": "t2", "check3": "t3"}

        def file_query(record, command):
            key = "check1" if "check1" in record["title"] else "check2" if "check2" in record["title"] else "check3"
            return "[exitCode:0]\nTOKEN=%s" % tokens[key]

        executeAllTasksInCategory(ledger, FILE_CATEGORY, file_query, lambda _ans: True)
        self.assertEqual(ledger.completed_count(API_CATEGORY), 3)
        self.assertEqual(ledger.completed_count(FILE_CATEGORY), 3)
        self.assertEqual(ledger.claim_count[API_CATEGORY], 3)
        self.assertEqual(ledger.claim_count[FILE_CATEGORY], 3)
        self.assertEqual(ledger.total_claims(), 6)
        self.assertTrue(ledger.evolution_complete())

    def test_success_writes_reusable_skill(self):
        result = run_closed_loop(heritage_stale_command("北京"), FakeHeritage(),
                                 task=API_TITLES[0], credential=SECRET)
        skill = result["skill"]
        dumped = json.dumps(skill, ensure_ascii=False)
        self.assertNotIn(SECRET, dumped)
        self.assertEqual(skill["auth_header"], "Authorization")
        self.assertEqual(skill["param_map"]["city"], "location")
        self.assertIn("types", skill.get("types_rule", "") + skill.get("success_condition", ""))
        reused = apply_skill(skill, API_TITLES[1], credential=SECRET)
        self.assertEqual(reused["params"]["location"], "上海")


if __name__ == "__main__":
    unittest.main()
