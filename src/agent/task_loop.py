"""Recoverable self-evolution task repairs for the platform sandbox.

The contestant process never talks to the weather API or the shell itself: it
emits the next `executeCmd`. These helpers classify `lastCmdResult`, rewrite
the failed request, and record a secret-free SOP for later tasks.
"""
import json
import re
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


MAX_REQUEST_ATTEMPTS = 5
RECEIVED = "RECEIVED"
ANALYZING = "ANALYZING"
EXECUTING = "EXECUTING"
FAILED_RECOVERABLE = "FAILED_RECOVERABLE"
REPAIRING = "REPAIRING"
RETRYING = "RETRYING"
VALIDATING = "VALIDATING"
LEARNING = "LEARNING"
COMPLETED = "COMPLETED"
FAILED_TERMINAL = "FAILED_TERMINAL"

MISSING_AUTH_HEADER = "MISSING_AUTH_HEADER"
INVALID_AUTH_SCHEME = "INVALID_AUTH_SCHEME"
INVALID_CREDENTIAL = "INVALID_CREDENTIAL"
PERMISSION_DENIED = "PERMISSION_DENIED"
MISSING_PARAMETER = "MISSING_PARAMETER"
BAD_INTERPRETER = "BAD_INTERPRETER"
CRLF_SHEBANG = "CRLF_SHEBANG"
NO_EXEC_PERMISSION = "NO_EXEC_PERMISSION"
MISSING_INTERPRETER = "MISSING_INTERPRETER"
UNKNOWN_ERROR = "UNKNOWN_ERROR"

RECOVERABLE = frozenset({
    MISSING_AUTH_HEADER, INVALID_AUTH_SCHEME, MISSING_PARAMETER,
    BAD_INTERPRETER, CRLF_SHEBANG, NO_EXEC_PERMISSION, MISSING_INTERPRETER,
})
TERMINAL = frozenset({INVALID_CREDENTIAL, PERMISSION_DENIED})
PARAM_ALIASES = {"city": "location", "location": "city"}
AUTH_HEADER_NAMES = ("authorization", "x-api-key", "api-key", "api_key", "token")
SECRET_RE = re.compile(
    r"(?i)((?:authorization|x-api-key|api[-_]?key|token|cookie)\s*[:=]\s*(?:bearer\s+)?)"
    r"([^\s'\"\\]+)"
)
BEARER_RE = re.compile(r"(?i)^bearer\s+")
BEARER_VALUE_RE = re.compile(r"(?i)(\bbearer\s+)([A-Za-z0-9._\-+=/]+)")


def redact(text):
    if not isinstance(text, str):
        return ""
    text = SECRET_RE.sub(r"\1<redacted>", text)
    text = BEARER_VALUE_RE.sub(r"\1<redacted>", text)
    return re.sub(
        r'(?i)("(?:authorization|x-api-key|api[-_]?key|token|cookie)"\s*:\s*")([^"\\]+)',
        r"\1<redacted>",
        text,
    )


def redact_headers(headers):
    safe = {}
    for name, value in (headers or {}).items():
        key = str(name)
        if key.lower() in AUTH_HEADER_NAMES or key.lower() == "cookie":
            safe[key] = "<redacted>"
        else:
            safe[key] = value
    return safe


def parse_command_output(raw):
    text = "" if raw is None else str(raw)
    code = None
    match = re.match(r"\[exitCode:(-?\d+)\]\s*\n?", text)
    if match:
        code = int(match.group(1))
        text = text[match.end():]
    timeout = "[TIMEOUT]" in text
    return {"exit_code": code, "body": text, "timeout": timeout, "raw": raw}


def classifyExecutionError(command, output):
    parsed = parse_command_output(output)
    body = parsed["body"]
    lowered = body.lower()
    command = command or ""
    path = command_script_path(command)
    if re.search(r"Missing\s+'Authorization'\s+header", body, flags=re.I):
        return {"type": MISSING_AUTH_HEADER, "header": "Authorization", "recoverable": True}
    if re.search(r"invalid\s+(authorization|auth)\s+scheme", body, flags=re.I):
        return {"type": INVALID_AUTH_SCHEME, "recoverable": True}
    if re.search(r"(invalid|wrong|unauthorized)\s+(api[- ]?key|token|credential)", body, flags=re.I):
        return {"type": INVALID_CREDENTIAL, "recoverable": False}
    missing = re.search(r"Missing required parameter:\s*([A-Za-z_][\w-]*)", body)
    if missing:
        return {"type": MISSING_PARAMETER, "name": missing.group(1), "recoverable": True}
    if "bad interpreter" in lowered or (path and "interpreter" in lowered):
        kind = CRLF_SHEBANG if ("\\r" in body or "\r" in body or "crlf" in lowered or "^m" in lowered) else BAD_INTERPRETER
        return {"type": kind, "path": path, "recoverable": True}
    if path and re.search(r"permission denied", lowered):
        return {"type": NO_EXEC_PERMISSION, "path": path, "recoverable": True}
    if path and re.search(r"no such file or directory", lowered) and re.search(r"(/bin/|/usr/bin/|env)", lowered):
        return {"type": MISSING_INTERPRETER, "path": path, "recoverable": True}
    if re.search(r"\b403\b|forbidden", lowered) and "interpreter" not in lowered:
        return {"type": PERMISSION_DENIED, "recoverable": False}
    if parsed["timeout"] or parsed["exit_code"] not in (None, 0):
        return {"type": UNKNOWN_ERROR, "recoverable": False, "exit_code": parsed["exit_code"]}
    return None


def isRecoverableError(error):
    if not error:
        return False
    if error.get("type") in TERMINAL:
        return False
    return bool(error.get("recoverable")) or error.get("type") in RECOVERABLE


def parseApiError(output):
    error = classifyExecutionError("", output) or {}
    return {k: error[k] for k in ("type", "name", "header") if k in error} or None


def command_script_path(command):
    if not command:
        return None
    match = re.search(
        r"(?:^|[\s;|&])(?:bash|sh|python3?|env)\s+(\./[\w./-]+)", command)
    if match:
        return match.group(1)
    match = re.search(r"(?:^|[\s;|&])(\./[\w./-]+)", command)
    return match.group(1) if match else None


def parse_http_command(command):
    """Best-effort curl/URL parser; unknown commands still keep the raw string."""
    if not isinstance(command, str) or not command.strip():
        return None
    spec = {"method": "GET", "url": "", "headers": {}, "params": {}, "raw": command}
    method = re.search(r"\b(?:curl\s+(?:-X|--request)\s+)?(GET|POST|PUT|PATCH|DELETE)\b",
                       command, flags=re.I)
    if method:
        spec["method"] = method.group(1).upper()
    elif "curl" in command:
        spec["method"] = "GET"
    for header in re.findall(r"(?:-H|--header)\s+(['\"])(.+?)\1", command):
        line = header[1]
        if ":" in line:
            name, value = line.split(":", 1)
            spec["headers"][name.strip()] = value.strip()
    for blob in re.findall(r"(?:-d|--data(?:-urlencode|--data-raw)?)\s+(['\"])(.+?)\1", command):
        spec["params"].update(dict(parse_qsl(blob[1], keep_blank_values=True)))
    url_match = re.search(r"https?://[^\s'\"\\]+", command)
    if url_match:
        spec["url"] = url_match.group(0)
        parsed = urlparse(spec["url"])
        spec["params"].update(dict(parse_qsl(parsed.query, keep_blank_values=True)))
        spec["url"] = urlunparse(parsed._replace(query=""))
    for key, value in re.findall(r"\b([A-Za-z_][\w-]*)=([^\s&'\"\\]+)", command):
        if key.lower() not in AUTH_HEADER_NAMES:
            spec["params"].setdefault(key, value)
    return spec if spec["url"] or spec["params"] or spec["headers"] else {
        "method": "GET", "url": "", "headers": {}, "params": {}, "raw": command,
    }


def extract_credential(spec):
    headers = (spec or {}).get("headers") or {}
    for name, value in headers.items():
        if not value:
            continue
        if name.lower() in AUTH_HEADER_NAMES:
            return BEARER_RE.sub("", str(value)).strip()
    return None


def repairAuthentication(spec, error, credential=None):
    spec = dict(spec or {})
    headers = dict(spec.get("headers") or {})
    secret = credential or extract_credential(spec)
    error_type = (error or {}).get("type")
    if error_type in {INVALID_CREDENTIAL, PERMISSION_DENIED}:
        return None
    if not secret:
        return None
    kept = {name: value for name, value in headers.items()
            if name.lower() not in AUTH_HEADER_NAMES}
    current = headers.get("Authorization") or headers.get("authorization") or ""
    token = BEARER_RE.sub("", str(secret)).strip()
    if error_type == INVALID_AUTH_SCHEME or (current and not current.lower().startswith("bearer ")):
        kept["Authorization"] = "Bearer " + token
    else:
        kept["Authorization"] = token
    spec["headers"] = kept
    spec["auth_format"] = "Bearer" if str(kept["Authorization"]).lower().startswith("bearer ") else "raw"
    spec["credential_ref"] = "in_memory_header"
    return spec


def repairRequestParameters(spec, error):
    spec = dict(spec or {})
    params = dict(spec.get("params") or {})
    missing = (error or {}).get("name")
    if not missing:
        return spec
    if missing in params and params[missing]:
        spec["params"] = params
        return spec
    source_name = next((alias for alias, target in PARAM_ALIASES.items()
                        if target == missing and alias in params), None)
    if source_name is None:
        source_name = next((name for name, value in params.items() if value), None)
    if source_name is None:
        return spec
    params[missing] = params[source_name]
    if source_name != missing:
        params.pop(source_name, None)
        spec.setdefault("param_map", {})[source_name] = missing
    spec["params"] = params
    return spec


def request_fingerprint(spec):
    spec = spec or {}
    if not spec.get("url") and not spec.get("headers") and not spec.get("params"):
        return ("raw", (spec.get("raw") or "").strip())
    headers = tuple(sorted(name.lower() for name in (spec.get("headers") or {})))
    params = tuple(sorted((str(k), str(v)) for k, v in (spec.get("params") or {}).items()))
    schemes = []
    for name, value in (spec.get("headers") or {}).items():
        if name.lower() in AUTH_HEADER_NAMES:
            schemes.append((name.lower(), "bearer" if str(value).lower().startswith("bearer ") else "raw"))
    return (
        str(spec.get("method") or "GET").upper(),
        spec.get("url") or "",
        headers,
        tuple(sorted(schemes)),
        params,
    )


def buildRetryRequest(spec):
    spec = spec or {}
    method = str(spec.get("method") or "GET").upper()
    url = spec.get("url") or ""
    params = spec.get("params") or {}
    headers = spec.get("headers") or {}
    if url and params:
        parsed = urlparse(url)
        url = urlunparse(parsed._replace(query=urlencode(params)))
    header_flags = " ".join(
        "-H %s" % json.dumps("%s: %s" % (name, value), ensure_ascii=False)
        for name, value in headers.items()
    )
    if url:
        return ("curl -sS -X %s %s %s" % (
            method, header_flags, json.dumps(url, ensure_ascii=False))).strip()
    return spec.get("raw") or ""


def recordAttempt(command, output, error=None, repair=None, spec=None):
    parsed = parse_command_output(output)
    return {
        "method": (spec or {}).get("method"),
        "url": (spec or {}).get("url"),
        "headers": redact_headers((spec or {}).get("headers")),
        "param_names": sorted((spec or {}).get("params") or {}),
        "command": redact(command or ""),
        "exit_code": parsed["exit_code"],
        "error_type": (error or {}).get("type"),
        "repair": repair,
        "body_excerpt": redact((parsed["body"] or "")[:2000]),
    }


def detectScriptFormat(content):
    data = content if isinstance(content, (bytes, bytearray)) else str(content or "").encode("utf-8", "replace")
    first = data.split(b"\n", 1)[0]
    crlf = b"\r\n" in data or first.endswith(b"\r")
    shebang = first.decode("utf-8", "replace") if first.startswith(b"#!") else ""
    return {
        "crlf": crlf,
        "shebang": shebang.replace("\r", ""),
        "shebang_has_cr": b"\r" in first,
        "interpreter": shebang[2:].strip().split()[0] if shebang.startswith("#!") else "",
    }


def repairLineEndings(content):
    data = content if isinstance(content, (bytes, bytearray)) else str(content or "").encode("utf-8")
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def repairLineEndingsFile(path, root="."):
    target = Path(path)
    base = Path(root).resolve()
    resolved = (base / target).resolve() if not target.is_absolute() else target.resolve()
    if base not in resolved.parents and resolved != base:
        raise ValueError("refusing to rewrite file outside the task directory")
    if not resolved.is_file():
        raise ValueError("not a regular file")
    original = resolved.read_bytes()
    repaired = repairLineEndings(original)
    if repaired != original:
        resolved.write_bytes(repaired)
    return original, repaired


def repairExecutePermission(path):
    return "chmod +x %s" % path


def validateInterpreter(shebang, available=("bash", "sh", "python3")):
    interpreter = (shebang or "").replace("\r", "").strip()
    if interpreter.startswith("#!"):
        interpreter = interpreter[2:].strip().split()[0]
    name = Path(interpreter).name if interpreter else ""
    if name in available or interpreter in ("/bin/bash", "/bin/sh", "/usr/bin/env"):
        return {"ok": True, "interpreter": interpreter, "fallback": None}
    if name in {"bash", "sh"} or "bash" in interpreter:
        return {"ok": False, "interpreter": interpreter, "fallback": "bash"}
    if "python" in name:
        return {"ok": False, "interpreter": interpreter, "fallback": "python3"}
    return {"ok": False, "interpreter": interpreter, "fallback": "bash"}


def retryCommand(original, repairs):
    steps = [step for step in (repairs or []) if step]
    if original and original not in steps:
        steps.append(original)
    return " && ".join(steps)


def script_repair_command(path, original=None):
    path = path or "./check"
    original = original or path
    converter = "python3 -c %s" % json.dumps(
        "from pathlib import Path; root=Path('.').resolve(); "
        "p=Path(%s); t=p.resolve(); "
        "ok=t==root or str(t).startswith(str(root)+'/'); "
        "data=t.read_bytes() if ok and t.is_file() else b''; "
        "cr,lf=bytes([13]),bytes([10]); "
        "t.write_bytes(data.replace(cr+lf, lf).replace(cr, lf)) if data else None"
        % json.dumps(path)
    )
    return retryCommand(None, [
        converter,
        repairExecutePermission(path),
        "%s || bash %s" % (original, path),
    ])


def looks_successful(output):
    parsed = parse_command_output(output)
    body = parsed["body"] or ""
    if parsed["timeout"] or "[JUDGER_ERROR]" in body or "[NO_RESULT]" in body:
        return False
    error = classifyExecutionError("", output)
    if error and error.get("type") != UNKNOWN_ERROR:
        return False
    if parsed["exit_code"] not in (None, 0):
        return False
    return bool(body.strip())


def validateTaskResult(output, task=""):
    if not looks_successful(output):
        return False
    body = parse_command_output(output)["body"]
    if any(token in body.lower() for token in ("missing", "error", "bad interpreter", "traceback")):
        if re.search(r"Missing required parameter|Missing 'Authorization'|bad interpreter", body, flags=re.I):
            return False
    if "天气" in task or "weather" in task.lower():
        return bool(re.search(r"weather|温度|天气|temp|晴|雨|云", body, flags=re.I))
    return True


def extract_city(task):
    if not isinstance(task, str):
        return None
    match = re.search(r"查询\s*([^\s，。,:：]{1,20})天气", task)
    if match:
        return match.group(1)
    match = re.search(r"(北京|上海|广州|深圳|成都|杭州|南京|武汉|西安|重庆)", task)
    return match.group(1) if match else None


def generateOrUpdateSkill(existing, spec, error_history, script=None):
    skill = dict(existing or {})
    if spec:
        skill["kind"] = "api"
        skill["method"] = spec.get("method") or skill.get("method") or "GET"
        skill["url"] = spec.get("url") or skill.get("url")
        headers = spec.get("headers") or {}
        auth_name = next((name for name in headers if name.lower() == "authorization"), "Authorization")
        skill["auth_header"] = auth_name
        value = headers.get(auth_name) or ""
        skill["auth_format"] = "Bearer" if BEARER_RE.match(value) else "raw"
        skill["credential_ref"] = spec.get("credential_ref") or "in_memory_header"
        skill.setdefault("param_map", {})
        skill["param_map"].update(spec.get("param_map") or {})
        if "location" in (spec.get("params") or {}) and "city" not in skill["param_map"]:
            skill["param_map"]["city"] = "location"
        skill["param_names"] = sorted((spec.get("params") or {}).keys())
        skill["value_source"] = "task_city"
        skill["success_condition"] = "exit 0 and response contains weather fields"
        skill["summary"] = (
            "输入：城市名称；参数：%s=<城市名称>；鉴权：使用 %s 请求头，凭据从安全配置读取；"
            "输出：从成功响应中提取天气信息"
            % ((skill["param_names"][0] if skill.get("param_names") else "location"),
               skill["auth_header"])
        )
        skill["recovery"] = [
            "缺少 Authorization 时改用 Authorization 请求头，凭据从已有安全引用读取",
            "缺少 location 时将 city 映射为 location，保留原城市值",
            "禁止在日志或 SOP 中输出完整凭据",
        ]
    if script:
        skill.setdefault("script", {})
        skill["script"].update({
            "command": script.get("command") or "./check",
            "crlf_detect": "file contains \\r\\n or shebang ends with \\r",
            "crlf_repair": "python replace \\r\\n then \\r, or sed -i 's/\\r$//'",
            "chmod": "chmod +x",
            "fallback": "bash ./check",
        })
    history = [item.get("error_type") for item in (error_history or []) if item.get("error_type")]
    if history:
        skill["seen_errors"] = history[-8:]
    return skill


def apply_skill(skill, task, credential=None):
    if not skill or skill.get("kind") != "api" or not skill.get("url"):
        return None
    city = extract_city(task)
    params = {}
    names = skill.get("param_names") or ["location"]
    mapped = skill.get("param_map") or {}
    target = mapped.get("city") or ("location" if "location" in names else (names[0] if names else "location"))
    if city:
        params[target] = city
    headers = {}
    if credential:
        if skill.get("auth_format") == "Bearer":
            headers[skill.get("auth_header") or "Authorization"] = "Bearer " + credential
        else:
            headers[skill.get("auth_header") or "Authorization"] = credential
    spec = {
        "method": skill.get("method") or "GET",
        "url": skill.get("url"),
        "headers": headers,
        "params": params,
        "param_map": dict(mapped),
        "credential_ref": skill.get("credential_ref"),
    }
    return spec


def skill_text(skill):
    if not skill:
        return ""
    return json.dumps({k: v for k, v in skill.items() if k != "credential"},
                      ensure_ascii=False, indent=2)


def extract_task_answer(output, task=""):
    body = parse_command_output(output)["body"].strip()
    if not body:
        return ""
    decoder = json.JSONDecoder()
    for index, char in enumerate(body):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(body[index:])
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        weather_keys = ("weather", "temperature", "temp", "description", "answer", "result")
        if "天气" in (task or "") or "weather" in (task or "").lower():
            city = extract_city(task) or obj.get("city") or obj.get("location")
            parts = []
            if city:
                parts.append(str(city))
            for key in weather_keys:
                if obj.get(key) not in (None, ""):
                    parts.append(str(obj[key]))
                    break
            extra = obj.get("temp") or obj.get("temperature")
            if extra is not None and str(extra) not in parts:
                parts.append(str(extra))
            if len(parts) >= 2:
                return " ".join(parts)
            return json.dumps(obj, ensure_ascii=False)
        for key in weather_keys:
            if obj.get(key) not in (None, ""):
                return str(obj[key])
        return json.dumps(obj, ensure_ascii=False)
    if "天气" in (task or "") or "weather" in (task or "").lower():
        for line in body.splitlines():
            text = line.strip()
            if text and not text.startswith("["):
                return text[:500]
    return ""


def script_repair_plan(path, error):
    path = path or "./check"
    error_type = (error or {}).get("type")
    steps = []
    if error_type in {BAD_INTERPRETER, CRLF_SHEBANG, MISSING_INTERPRETER}:
        steps.append(script_repair_command(path, original=path))
    elif error_type == NO_EXEC_PERMISSION:
        steps.append(retryCommand(path, [repairExecutePermission(path), "%s || bash %s" % (path, path)]))
    else:
        steps.append(script_repair_command(path, original=path))
    return steps


def new_task_session(skill=None, credential=None):
    return {
        "state": RECEIVED,
        "attempts": [],
        "failed": set(),
        "credential": credential,
        "skill": skill,
        "spec": None,
        "repair_count": 0,
        "last_repair": None,
    }


def _command_fingerprint(command, spec):
    payload = dict(spec or {})
    payload.setdefault("raw", command or "")
    return request_fingerprint(payload)


def advance_execution(command, output, session=None, task="", max_attempts=MAX_REQUEST_ATTEMPTS):
    """Classify sandbox output and, when recoverable, emit a changed retry command."""
    session = dict(session or new_task_session())
    session.setdefault("attempts", [])
    session.setdefault("failed", set())
    command = command or ""
    spec = parse_http_command(command) or {"method": "GET", "url": "", "headers": {}, "params": {}, "raw": command}
    credential = extract_credential(spec) or session.get("credential")
    session["credential"] = credential
    session["spec"] = spec
    error = classifyExecutionError(command, output)
    result = {
        "state": session.get("state") or EXECUTING,
        "executeCmd": "",
        "taskAnswer": "",
        "prompt_needed": False,
        "reason": "",
        "spec": spec,
        "skill": session.get("skill"),
        "attempts": session["attempts"],
        "failed": session["failed"],
        "credential": credential,
        "repair": None,
        "error": error,
        "session": session,
    }

    if looks_successful(output) and validateTaskResult(output, task):
        script_meta = None
        path = command_script_path(command)
        if path or (command and "chmod" in command and "./" in command):
            script_meta = {"command": path or "./check"}
        skill = generateOrUpdateSkill(session.get("skill"), spec if spec.get("url") else None,
                                      session["attempts"], script=script_meta)
        answer = extract_task_answer(output, task)
        session["state"] = LEARNING
        session["skill"] = skill
        result.update({
            "state": LEARNING,
            "skill": skill,
            "taskAnswer": answer,
            "prompt_needed": not bool(answer),
            "reason": "validated",
            "session": session,
        })
        return result

    failed = session["failed"]
    failed.add(_command_fingerprint(command, spec))
    attempts = session["attempts"]
    attempts.append(recordAttempt(command, output, error, None, spec))

    if not isRecoverableError(error):
        if error and error.get("type") in {INVALID_CREDENTIAL, PERMISSION_DENIED}:
            session["state"] = FAILED_TERMINAL
            result.update({"state": FAILED_TERMINAL, "reason": error.get("type"), "prompt_needed": False})
            return result
        session["state"] = ANALYZING
        result.update({"state": ANALYZING, "prompt_needed": True,
                       "reason": (error or {}).get("type") or "unclassified"})
        return result

    if session.get("repair_count", 0) >= max_attempts or len(attempts) >= max_attempts:
        session["state"] = FAILED_TERMINAL
        result.update({"state": FAILED_TERMINAL, "reason": "max_attempts", "prompt_needed": False})
        return result

    session["state"] = FAILED_RECOVERABLE
    error_type = error.get("type")
    next_spec = spec
    next_cmd = ""
    repair = None
    session["state"] = REPAIRING
    if error_type in {MISSING_AUTH_HEADER, INVALID_AUTH_SCHEME}:
        if not credential:
            session["state"] = ANALYZING
            result.update({"state": ANALYZING, "prompt_needed": True, "reason": "missing_credential"})
            return result
        next_spec = repairAuthentication(spec, error, credential)
        if not next_spec:
            session["state"] = FAILED_TERMINAL
            result.update({"state": FAILED_TERMINAL, "reason": error_type, "prompt_needed": False})
            return result
        repair = "repair_authentication"
        next_cmd = buildRetryRequest(next_spec)
    elif error_type == MISSING_PARAMETER:
        next_spec = repairRequestParameters(spec, error)
        repair = "repair_parameters"
        next_cmd = buildRetryRequest(next_spec)
    elif error_type in {BAD_INTERPRETER, CRLF_SHEBANG, NO_EXEC_PERMISSION, MISSING_INTERPRETER}:
        path = error.get("path") or command_script_path(command) or "./check"
        repair = {
            BAD_INTERPRETER: "repair_script_interpreter",
            CRLF_SHEBANG: "repair_crlf",
            NO_EXEC_PERMISSION: "repair_execute_permission",
            MISSING_INTERPRETER: "repair_missing_interpreter",
        }[error_type]
        next_cmd = script_repair_plan(path, error)[0]
        next_spec = {
            "method": "SCRIPT", "url": path, "headers": {},
            "params": {"repair": repair}, "raw": next_cmd,
        }
    else:
        session["state"] = ANALYZING
        result.update({"state": ANALYZING, "prompt_needed": True, "reason": error_type})
        return result

    if not next_cmd or next_cmd.strip() == command.strip():
        session["state"] = FAILED_TERMINAL
        result.update({"state": FAILED_TERMINAL, "reason": "unchanged_request", "prompt_needed": False})
        return result
    next_fp = _command_fingerprint(next_cmd, next_spec)
    if next_fp in failed:
        session["state"] = FAILED_TERMINAL
        result.update({"state": FAILED_TERMINAL, "reason": "duplicate_request", "prompt_needed": False})
        return result

    attempts[-1]["repair"] = repair
    session["repair_count"] = session.get("repair_count", 0) + 1
    session["last_repair"] = repair
    session["state"] = RETRYING
    session["spec"] = next_spec
    result.update({
        "state": RETRYING,
        "executeCmd": next_cmd,
        "spec": next_spec,
        "repair": repair,
        "reason": error_type,
        "session": session,
        "attempts": attempts,
        "failed": failed,
    })
    return result


def run_closed_loop(initial_command, executor, task="", skill=None, credential=None,
                    max_attempts=MAX_REQUEST_ATTEMPTS):
    """Drive execute → repair → retry until success, terminal failure, or analysis."""
    session = new_task_session(skill=skill, credential=credential)
    command = initial_command
    output = ""
    last = None
    for _ in range(max_attempts):
        session["state"] = EXECUTING
        output = executor(command)
        last = advance_execution(command, output, session=session, task=task, max_attempts=max_attempts)
        session = last["session"]
        if last["state"] == RETRYING and last.get("executeCmd"):
            command = last["executeCmd"]
            continue
        return last
    if last is None:
        last = advance_execution(command, output, session=session, task=task, max_attempts=max_attempts)
    last["state"] = FAILED_TERMINAL
    last["reason"] = last.get("reason") or "max_attempts"
    return last
