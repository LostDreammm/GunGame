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
SUBTASKS_PER_CATEGORY = 3
TOTAL_EVOLUTION_TASKS = 6
RECEIVED = "RECEIVED"
ANALYZING = "ANALYZING"
EXECUTING = "EXECUTING"
FAILED_RECOVERABLE = "FAILED_RECOVERABLE"
REPAIRING = "REPAIRING"
RETRYING = "RETRYING"
VALIDATING = "VALIDATING"
LEARNING = "LEARNING"
API_RESPONSE_RECEIVED = "API_RESPONSE_RECEIVED"
ANSWER_READY = "ANSWER_READY"
SUBMITTING = "SUBMITTING"
PENDING = "PENDING"
RUNNING = "RUNNING"
RETRYABLE_FAILED = "RETRYABLE_FAILED"
TERMINAL_FAILED = "TERMINAL_FAILED"
COMPLETED = "COMPLETED"
FAILED_TERMINAL = "FAILED_TERMINAL"

API_CATEGORY = "api"
FILE_CATEGORY = "file"

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
# Only map the documented stale name onto the verified parameter. Never map
# location back to city: that oscillation is what produced the 400/401 loop.
PARAM_ALIASES = {"city": "location"}
AUTH_HEADER_NAMES = ("authorization", "x-api-key", "api-key", "api_key", "token")
CITY_PARAM_NAMES = ("location", "city")
TOKEN_RE = re.compile(
    r"(?im)^\s*(?:token|令牌|check[_-]?token)\s*[:=：]\s*([A-Za-z0-9._\-+=/]{6,})\s*$"
)
TOKEN_PLACEHOLDERS = frozenset({
    "xxx", "xxxx", "token", "your_token", "your-token", "todo", "none",
})
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


def _http_status_code(body):
    text = body or ""
    match = re.search(r"HTTP/\d(?:\.\d)?\s+(\d{3})", text)
    if match:
        return int(match.group(1))
    match = re.search(r'"(?:status|statusCode|http_code)"\s*:\s*(\d{3})', text)
    if match:
        return int(match.group(1))
    match = re.search(r'"code"\s*:\s*(\d{3})\b', text)
    if match:
        value = int(match.group(1))
        if 100 <= value <= 599:
            return value
    return None


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
    if re.search(r"missing\s+(required\s+)?parameter.*\blocation\b|\blocation\b.*required", lowered):
        return {"type": MISSING_PARAMETER, "name": "location", "recoverable": True}
    status = _http_status_code(body)
    if status == 401 or re.search(r"\b401\b", body):
        if re.search(r"invalid|wrong|forbidden credential", lowered):
            return {"type": INVALID_CREDENTIAL, "recoverable": False}
        return {"type": MISSING_AUTH_HEADER, "header": "Authorization", "recoverable": True}
    if status == 400 and re.search(r"\blocation\b", lowered):
        return {"type": MISSING_PARAMETER, "name": "location", "recoverable": True}
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


def classifyApiError(output, command=""):
    return classifyExecutionError(command, output)


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


def classifyTaskKind(task):
    text = task or ""
    lowered = text.lower()
    file_markers = ("check", "token", "chmod", "shebang", "crlf", "修复", "脚本",
                    "权限", "配置文件", "newline", "interpreter", "./check")
    api_markers = ("api", "天气", "weather", "文化遗产", "heritage", "文物",
                   "location", "http", "curl", "x-api-key", "authorization")
    if any(marker in text or marker in lowered for marker in (
            "文化遗产", "文物", "天气", "weather", "http://", "https://", "curl ")):
        return API_CATEGORY
    if any(marker in text or marker in lowered for marker in file_markers):
        return FILE_CATEGORY
    if any(marker in lowered for marker in api_markers):
        return API_CATEGORY
    return API_CATEGORY


def split_subtasks(text):
    if not isinstance(text, str) or not text.strip():
        return []
    numbered = re.findall(r"任务\s*[123一二三][:：.、)\s]+([^\n]+)", text)
    titles = [item.strip() for item in numbered if item and item.strip()]
    if len(titles) >= 2:
        return titles[:SUBTASKS_PER_CATEGORY]
    cities = re.findall(r"查询\s*([^\s，。,：:]+?)\s*(?:的)?(?:天气|文化遗产|文物)", text)
    if len(cities) >= 2:
        kind = "文化遗产" if ("文化遗产" in text or "文物" in text) else "天气"
        return ["请查询%s%s" % (city, kind) for city in cities[:SUBTASKS_PER_CATEGORY]]
    return [text.strip()]


def _json_from_output(output):
    body = parse_command_output(output)["body"]
    if not body:
        return None
    decoder = json.JSONDecoder()
    for index, char in enumerate(body):
        if char not in "{[":
            continue
        try:
            obj, _ = decoder.raw_decode(body[index:])
        except ValueError:
            continue
        if isinstance(obj, (dict, list)):
            return obj
    return None


def looks_http_ok(output):
    parsed = parse_command_output(output)
    body = parsed["body"] or ""
    status = _http_status_code(body)
    if status is not None:
        return status == 200
    return looks_successful(output)


def should_paginate(task, output):
    text = task or ""
    if not re.search(r"分页|全部页|all pages|every page", text, flags=re.I):
        return False
    obj = _json_from_output(output)
    if not isinstance(obj, dict):
        return False
    if obj.get("next") or obj.get("nextPage") or obj.get("has_more") is True:
        return True
    page = obj.get("page") or obj.get("pageNo") or obj.get("current")
    pages = obj.get("pages") or obj.get("totalPages") or obj.get("total_pages")
    try:
        return int(page) < int(pages)
    except (TypeError, ValueError):
        return False


def _auth_header_items(headers):
    return [(name, value) for name, value in (headers or {}).items()
            if str(name).lower() in AUTH_HEADER_NAMES]


def canonicalize_api_spec(spec, contract=None):
    """Keep exactly one auth header and one city parameter name."""
    spec = dict(spec or {})
    headers = dict(spec.get("headers") or {})
    params = dict(spec.get("params") or {})
    verified = bool(contract and contract.get("verified"))
    preferred_auth = (contract or {}).get("auth_header") or "Authorization"
    preferred_param = (contract or {}).get("location_parameter") or "location"

    auth_items = _auth_header_items(headers)
    if auth_items:
        chosen_name, chosen_value = auth_items[0]
        for name, value in auth_items:
            if str(name).lower() == str(preferred_auth).lower():
                chosen_name, chosen_value = name, value
                break
        if verified:
            chosen_name = preferred_auth
        elif any(str(name).lower() == "authorization" for name, _ in auth_items):
            chosen_name = next(name for name, _ in auth_items if str(name).lower() == "authorization")
            chosen_value = headers.get(chosen_name) or chosen_value
        headers = {name: value for name, value in headers.items()
                   if str(name).lower() not in AUTH_HEADER_NAMES}
        headers[chosen_name] = chosen_value

    city_keys = [name for name in params if str(name).lower() in CITY_PARAM_NAMES]
    if len(city_keys) > 1 or (verified and city_keys):
        keep = preferred_param if verified else (
            "location" if any(str(name).lower() == "location" for name in city_keys) else city_keys[0]
        )
        value = params.get(keep)
        if value in (None, ""):
            value = next((params[name] for name in city_keys if params.get(name) not in (None, "")), None)
        for name in list(params):
            if str(name).lower() in CITY_PARAM_NAMES:
                params.pop(name, None)
        if value not in (None, ""):
            params[keep] = value
    spec["headers"] = headers
    spec["params"] = params
    return spec


def resolveVerifiedApiContract(skill_or_contract):
    if not skill_or_contract:
        return None
    names = skill_or_contract.get("param_names") or []
    mapped = skill_or_contract.get("param_map") or {}
    param = mapped.get("city") or ("location" if "location" in names else None)
    proven = bool(skill_or_contract.get("verified") or param == "location")
    if not proven:
        return None
    return {
        "verified": True,
        "auth_header": skill_or_contract.get("auth_header") or "Authorization",
        "auth_format": skill_or_contract.get("auth_format") or "raw",
        "location_parameter": param or "location",
        "method": skill_or_contract.get("method") or "GET",
        "url": skill_or_contract.get("url"),
    }


def contract_from_spec(spec):
    spec = canonicalize_api_spec(spec)
    headers = spec.get("headers") or {}
    params = spec.get("params") or {}
    auth_name = next((name for name in headers if str(name).lower() in AUTH_HEADER_NAMES), "Authorization")
    loc = "location" if "location" in params else ("city" if "city" in params else "location")
    value = headers.get(auth_name) or ""
    return {
        "verified": True,
        "auth_header": auth_name,
        "auth_format": "Bearer" if str(value).lower().startswith("bearer ") else "raw",
        "location_parameter": loc,
        "method": spec.get("method") or "GET",
        "url": spec.get("url"),
    }


def buildApiRequest(contract, city, credential=None):
    contract = resolveVerifiedApiContract(contract) or dict(contract or {})
    headers = {}
    if credential:
        name = contract.get("auth_header") or "Authorization"
        if contract.get("auth_format") == "Bearer":
            headers[name] = "Bearer " + credential
        else:
            headers[name] = credential
    param_name = contract.get("location_parameter") or "location"
    params = {param_name: city} if city else {}
    spec = {
        "method": contract.get("method") or "GET",
        "url": contract.get("url") or "",
        "headers": headers,
        "params": params,
        "param_map": {"city": param_name} if param_name != "city" else {},
        "credential_ref": "in_memory_header",
    }
    return canonicalize_api_spec(spec, contract)


def spec_sends_both_auth_headers(spec):
    return len(_auth_header_items((spec or {}).get("headers"))) > 1


def spec_sends_both_city_params(spec):
    params = (spec or {}).get("params") or {}
    return sum(1 for name in params if str(name).lower() in CITY_PARAM_NAMES) > 1


def normalizeTypes(types):
    cleaned = []
    seen = set()
    for item in types or []:
        if item is None:
            continue
        if isinstance(item, (list, dict, bool)):
            continue
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    normalized = sorted(cleaned)
    assert len(normalized) == len(set(normalized))
    assert normalized == sorted(normalized)
    return normalized


def _heritage_records(obj):
    if isinstance(obj, list):
        return [item for item in obj if isinstance(item, (dict, str))]
    if not isinstance(obj, dict):
        return []
    for key in ("items", "records", "heritages", "results", "list", "data"):
        value = obj.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            nested = _heritage_records(value)
            if nested:
                return nested
    return []


def _types_from_payload(obj, records):
    collected = []
    if isinstance(obj, dict):
        raw = obj.get("types") or obj.get("type")
        if isinstance(raw, list):
            collected.extend(raw)
        elif raw not in (None, ""):
            collected.append(raw)
        nested = obj.get("data")
        if isinstance(nested, dict):
            collected.extend(_types_from_payload(nested, []))
    for record in records or []:
        if isinstance(record, dict):
            value = record.get("type") or record.get("types") or record.get("category")
            if isinstance(value, list):
                collected.extend(value)
            elif value not in (None, ""):
                collected.append(value)
        elif isinstance(record, str):
            collected.append(record)
    return collected


def buildApiAnswer(output, task=""):
    obj = _json_from_output(output)
    if obj is None:
        token = extractToken(output)
        return token or ""
    if isinstance(obj, dict):
        status = str(obj.get("status") or "").lower()
        if status in {"error", "fail", "failed"} or obj.get("error"):
            return ""
    records = _heritage_records(obj)
    payload = obj if isinstance(obj, dict) else {}
    types = normalizeTypes(_types_from_payload(payload, records))
    city = extract_city(task)
    if not city and isinstance(payload, dict):
        city = payload.get("city") or payload.get("location")
        if not city and isinstance(payload.get("data"), dict):
            city = payload["data"].get("city") or payload["data"].get("location")
    count = len(records) if records else payload.get("count")
    if count is None:
        count = len(types)
    heritage = bool(types) or "文化遗产" in (task or "") or "文物" in (task or "")
    if heritage:
        answer = {"count": count, "types": types}
        if city:
            answer["city"] = city
        assert answer["types"] == normalizeTypes(answer["types"])
        return json.dumps(answer, ensure_ascii=False)
    return ""


def extractToken(output):
    body = parse_command_output(output)["body"]
    if not body:
        return ""
    match = TOKEN_RE.search(body)
    if match:
        value = match.group(1).strip()
        if value.lower() not in TOKEN_PLACEHOLDERS:
            return value
    obj = _json_from_output(output)
    if isinstance(obj, dict):
        status = str(obj.get("status") or obj.get("error") or "").lower()
        if status in {"error", "fail", "failed"}:
            return ""
        for key in ("token", "Token", "checkToken", "令牌"):
            value = str(obj.get(key) or "").strip()
            if value and value.lower() not in TOKEN_PLACEHOLDERS and len(value) >= 6:
                return value
    return ""


def extract_check_path(task):
    match = re.search(r"(\./[\w./-]+)", task or "")
    return match.group(1) if match else "./check"


def diagnoseFileTask(output, command="", task=""):
    return classifyExecutionError(command, output)


def repairFileEnvironment(path, error, original=None):
    path = path or extract_check_path("") or "./check"
    return script_repair_plan(path, error or {})[0]


def normalize_submission(answer, task=""):
    if not isinstance(answer, str) or not answer.strip():
        return answer or ""
    text = answer.strip()
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return text
    if isinstance(obj, dict) and "types" in obj:
        obj["types"] = normalizeTypes(obj.get("types"))
        return json.dumps(obj, ensure_ascii=False)
    if isinstance(obj, list) and ("types" in (task or "") or "文化遗产" in (task or "")):
        return json.dumps(normalizeTypes(obj), ensure_ascii=False)
    return text


def assert_types_normalized(types):
    normalized = normalizeTypes(types)
    if list(types or []) != normalized:
        raise AssertionError("types must be unique and stably sorted before submit")
    return normalized


def submitPreparedAnswer(record, submit_fn):
    record = record or {}
    if record.get("state") == COMPLETED or record.get("submitted"):
        return {"ok": True, "skipped": True, "answer": record.get("answer", "")}
    answer = record.get("answer") or ""
    if not answer:
        return {"ok": False, "reason": "no_answer"}
    record["state"] = SUBMITTING
    ok = bool(submit_fn(answer))
    if ok:
        record["state"] = COMPLETED
        record["submitted"] = True
        record["error"] = None
        return {"ok": True, "answer": answer}
    record["state"] = RETRYABLE_FAILED
    record["retries"] = int(record.get("retries") or 0) + 1
    record["error"] = "submit_failed"
    return {"ok": False, "retry": True, "answer": answer}


def _stable_task_id(family, title):
    return "%s|%s" % (family or "unknown", (title or "").strip())


class TaskLedger:
    def __init__(self):
        self.claimed_families = {}
        self.claim_count = {API_CATEGORY: 0, FILE_CATEGORY: 0}
        self.api_claimed = False
        self.file_claimed = False
        self.tasks = {}
        self.order = []
        self.current_id = None
        self.verified_api = None
        self.file_skill = None
        self.pending_claim_family = None
        self.query_complete = {}

    def total_claims(self):
        return int(self.claim_count.get(API_CATEGORY, 0)) + int(self.claim_count.get(FILE_CATEGORY, 0))

    def note_claim_attempt(self, family):
        family = str(family or "")
        if not family or self.total_claims() >= TOTAL_EVOLUTION_TASKS:
            return False
        if self.pending_claim_family:
            return False
        self.pending_claim_family = family
        return True

    def family_claimed(self, family):
        family = str(family or "")
        return family in self.claimed_families or family == self.pending_claim_family

    def _mark_category_claimed(self, category):
        if category == API_CATEGORY:
            self.api_claimed = True
        elif category == FILE_CATEGORY:
            self.file_claimed = True

    def _find_task(self, family, title):
        title = (title or "").strip()
        for item in self.tasks.values():
            if item.get("family") == family and item.get("title") == title:
                return item
        return None

    def _add_task(self, family, category, title):
        title = (title or "").strip()
        if not title:
            return None
        existing = self._find_task(family, title)
        if existing:
            return existing
        if self.category_total(category) >= SUBTASKS_PER_CATEGORY:
            return None
        if self.total_claims() >= TOTAL_EVOLUTION_TASKS:
            return None
        seq = self.total_claims() + 1
        tid = "%s|#%s|%s" % (family, seq, title)
        record = {
            "id": tid,
            "family": family,
            "category": category,
            "title": title,
            "state": PENDING,
            "answer": "",
            "submitted": False,
            "retries": 0,
            "error": None,
            "query_done": False,
        }
        self.tasks[tid] = record
        self.order.append(tid)
        self.claim_count[category] = int(self.claim_count.get(category, 0)) + 1
        self._mark_category_claimed(category)
        self.claimed_families[family] = category
        return record

    def register_from_phase(self, family, text):
        family = str(family or self.pending_claim_family or "unknown")
        title = (text or "").strip()
        if not title:
            return None
        category = classifyTaskKind(text)
        record = claimTask(self, family, category, title)
        self.pending_claim_family = None
        if record:
            self.current_id = record["id"]
        return self.current()

    def current(self):
        if self.current_id and self.current_id in self.tasks:
            return self.tasks[self.current_id]
        for tid in self.order:
            item = self.tasks[tid]
            if item["state"] != COMPLETED:
                return item
        return None

    def completed_count(self, category=None):
        return sum(
            1 for item in self.tasks.values()
            if item["state"] == COMPLETED and (category is None or item["category"] == category)
        )

    def family_completed(self, family):
        return sum(1 for item in self.tasks.values()
                   if item.get("family") == family and item["state"] == COMPLETED)

    def category_total(self, category):
        return sum(1 for item in self.tasks.values() if item.get("category") == category)

    def evolution_complete(self):
        return self.completed_count() >= TOTAL_EVOLUTION_TASKS or (
            self.completed_count(API_CATEGORY) >= SUBTASKS_PER_CATEGORY
            and self.completed_count(FILE_CATEGORY) >= SUBTASKS_PER_CATEGORY
        )

    def mark_current_completed(self):
        record = self.current()
        if not record:
            return None
        record["state"] = COMPLETED
        record["submitted"] = True
        record["error"] = None
        self.pending_claim_family = None
        return record

    def fail_current(self, state=RETRYABLE_FAILED, error=""):
        record = self.current()
        if not record:
            return None
        record["state"] = state
        record["error"] = error or record.get("error")
        self.pending_claim_family = None
        return record

    def store_answer(self, answer, query_done=True):
        record = self.current()
        if not record:
            return None
        record["answer"] = answer
        record["query_done"] = query_done
        record["state"] = ANSWER_READY
        return record

    def should_defer_new_claim(self):
        # Only block a second accept while the previous accept is still pending
        # or the current sub-task has not finished. After completion, claim again.
        if self.pending_claim_family:
            return True
        record = self.current()
        if record and record.get("state") not in {COMPLETED, TERMINAL_FAILED}:
            if record.get("state") in {RUNNING, ANSWER_READY, SUBMITTING, API_RESPONSE_RECEIVED}:
                return True
        return False

    def family_exhausted(self, family, meta=None):
        if self.family_completed(family) >= SUBTASKS_PER_CATEGORY:
            return True
        if self.total_claims() >= TOTAL_EVOLUTION_TASKS and self.family_completed(family) >= self.category_total(
                self.claimed_families.get(family)):
            return True
        if not meta:
            return False
        cooldown = int(meta.get("coldDownRounds") or 0)
        if not meta.get("isValid") and cooldown == 0 and self.family_completed(family) > 0:
            return True
        return False

    def to_dict(self):
        return {
            "claimed_families": dict(self.claimed_families),
            "claim_count": dict(self.claim_count),
            "api_claimed": self.api_claimed,
            "file_claimed": self.file_claimed,
            "tasks": dict(self.tasks),
            "order": list(self.order),
            "current_id": self.current_id,
            "verified_api": dict(self.verified_api) if self.verified_api else None,
            "pending_claim_family": self.pending_claim_family,
        }


def claimTask(ledger, family, category, title):
    """Claim one self-evolution sub-task. Duplicate titles reuse the existing record."""
    ledger = ledger or TaskLedger()
    family = str(family or category)
    title = (title or "").strip()
    if not title:
        return None
    existing = ledger._find_task(family, title)
    if existing:
        ledger.current_id = existing["id"]
        ledger.pending_claim_family = None
        return existing
    record = ledger._add_task(family, category, title)
    ledger.pending_claim_family = None
    if record:
        ledger.current_id = record["id"]
    return record


def claimTaskCategoryOnce(ledger, family, category, titles):
    """Claim each title separately. Six self-evolution tasks require six claims."""
    ledger = ledger or TaskLedger()
    for title in titles or []:
        claimTask(ledger, family, category, title)
    return loadClaimedTasks(ledger, category)


def loadClaimedTasks(ledger, category=None):
    if not ledger:
        return []
    items = [ledger.tasks[tid] for tid in ledger.order]
    if category:
        items = [item for item in items if item.get("category") == category]
    return items


def persistTaskProgress(ledger):
    return ledger.to_dict() if ledger else {}


def executeAllTasksInCategory(ledger, category, query_fn, submit_fn, skill=None, credential=None):
    """Run already-claimed sub-tasks: query once per task, then submit."""
    results = []
    for record in loadClaimedTasks(ledger, category):
        if record.get("state") == COMPLETED or record.get("submitted"):
            results.append(record)
            continue
        ledger.current_id = record["id"]
        if record.get("answer") and record.get("query_done"):
            submitPreparedAnswer(record, submit_fn)
            results.append(record)
            continue
        record["state"] = RUNNING
        command = ""
        if category == API_CATEGORY:
            contract = ledger.verified_api or skill
            city = extract_city(record.get("title"))
            spec = buildApiRequest(contract, city, credential) if contract else None
            command = buildRetryRequest(spec) if spec and spec.get("url") else ""
        else:
            command = extract_check_path(record.get("title"))
        if not command and category == API_CATEGORY:
            city = extract_city(record.get("title")) or "北京"
            command = 'curl -sS -X GET -H "x-api-key: %s" "http://127.0.0.1/api?city=%s"' % (
                credential or "missing", city)

        def executor(cmd, rec=record):
            return query_fn(rec, cmd)

        action = run_closed_loop(
            command, executor, task=record.get("title") or "",
            skill=skill or ledger.verified_api, credential=credential,
        )
        if action.get("skill") and category == API_CATEGORY:
            ledger.verified_api = resolveVerifiedApiContract(action["skill"]) or ledger.verified_api
            skill = action["skill"]
        if action.get("taskAnswer"):
            record["answer"] = action["taskAnswer"]
            record["query_done"] = True
            record["state"] = ANSWER_READY
            submitPreparedAnswer(record, submit_fn)
        else:
            record["state"] = RETRYABLE_FAILED if action.get("state") != FAILED_TERMINAL else TERMINAL_FAILED
            record["error"] = action.get("reason")
        results.append(record)
    return results


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
        if str(missing).lower() in CITY_PARAM_NAMES:
            spec["params"] = params
            return spec
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
    spec = canonicalize_api_spec(spec)
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
    status = _http_status_code(body)
    if status is not None and status != 200:
        return False
    if parsed["exit_code"] not in (None, 0):
        return False
    return bool(body.strip())


def validateTaskResult(output, task=""):
    if not looks_successful(output):
        return False
    body = parse_command_output(output)["body"]
    if re.search(r"Missing required parameter|Missing 'Authorization'|bad interpreter", body, flags=re.I):
        return False
    if classifyTaskKind(task) == FILE_CATEGORY:
        return bool(extractToken(output) or looks_successful(output))
    if "天气" in task or "weather" in (task or "").lower():
        return bool(re.search(r"weather|温度|天气|temp|晴|雨|云", body, flags=re.I))
    if "文化遗产" in (task or "") or "文物" in (task or "") or "types" in (task or "").lower():
        return bool(buildApiAnswer(output, task))
    return True


def extract_city(task):
    if not isinstance(task, str):
        return None
    match = re.search(r"查询\s*([^\s，。,:：]{1,20}?)(?:的)?(?:天气|文化遗产|文物)", task)
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
        skill["success_condition"] = (
            "HTTP 200 and parseable payload; types stripped/deduped/sorted; submit immediately"
        )
        skill["types_rule"] = "sorted(unique(strip(types)))"
        skill["verified"] = True
        skill["summary"] = (
            "输入：城市名称；参数：%s=<城市名称>；鉴权：使用 %s 请求头，凭据从安全配置读取；"
            "输出：统计后提交，types 去空白去重后按字典序排序"
            % ((skill["param_names"][0] if skill.get("param_names") else "location"),
               skill["auth_header"])
        )
        skill["recovery"] = [
            "缺少 Authorization 时改用 Authorization 请求头，凭据从已有安全引用读取",
            "缺少 location 时将 city 映射为 location，保留原城市值",
            "禁止同时发送 X-API-Key 与 Authorization，禁止同时发送 city 与 location",
            "HTTP 200 且数据完整后立即提交，不换协议、不无依据翻页",
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
            "token_extract": "parse token/令牌 from check output",
            "submit": "submit token immediately; retry submit only on failure",
        })
    history = [item.get("error_type") for item in (error_history or []) if item.get("error_type")]
    if history:
        skill["seen_errors"] = history[-8:]
    return skill


def apply_skill(skill, task, credential=None):
    if not skill:
        return None
    if skill.get("kind") == "script":
        return None
    contract = resolveVerifiedApiContract(skill)
    url = (contract or {}).get("url") or skill.get("url")
    if not url:
        return None
    city = extract_city(task)
    spec = buildApiRequest(contract or skill, city, credential)
    if not spec.get("url"):
        spec["url"] = url
    spec["param_map"] = dict(skill.get("param_map") or spec.get("param_map") or {})
    spec["credential_ref"] = skill.get("credential_ref") or "in_memory_header"
    return canonicalize_api_spec(spec, contract)


def skill_text(skill):
    if not skill:
        return ""
    return json.dumps({k: v for k, v in skill.items() if k != "credential"},
                      ensure_ascii=False, indent=2)


def extract_task_answer(output, task=""):
    body = parse_command_output(output)["body"].strip()
    if not body:
        return ""
    token = extractToken(output)
    if classifyTaskKind(task) == FILE_CATEGORY and token:
        return token
    heritage = buildApiAnswer(output, task)
    if heritage:
        return heritage
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
    if token:
        return token
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
    contract = resolveVerifiedApiContract(session.get("verified_contract") or session.get("skill"))
    spec = canonicalize_api_spec(spec, contract if contract and contract.get("verified") else None)
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
        "stop_query": False,
    }

    if session.get("prepared_answer") and session.get("query_complete"):
        session["state"] = ANSWER_READY
        result.update({
            "state": LEARNING,
            "taskAnswer": session["prepared_answer"],
            "reason": "reuse_prepared_answer",
            "stop_query": True,
            "session": session,
            "skill": session.get("skill"),
        })
        return result

    http_ok = looks_http_ok(output)
    payload_ok = validateTaskResult(output, task) if looks_successful(output) or http_ok else False
    if http_ok and payload_ok and not should_paginate(task, output):
        session["state"] = API_RESPONSE_RECEIVED
        script_meta = None
        path = command_script_path(command)
        if path or (command and "chmod" in command and "./" in command):
            script_meta = {"command": path or extract_check_path(task)}
        locked = spec if spec.get("url") else None
        if locked:
            session["verified_contract"] = contract_from_spec(spec)
        skill = generateOrUpdateSkill(session.get("skill"), locked, session["attempts"], script=script_meta)
        answer = extract_task_answer(output, task)
        if classifyTaskKind(task) == FILE_CATEGORY:
            answer = extractToken(output) or answer
        if answer and classifyTaskKind(task) == API_CATEGORY and "types" in str(answer):
            answer = normalize_submission(answer, task)
        session["prepared_answer"] = answer
        session["query_complete"] = True
        session["state"] = ANSWER_READY
        session["skill"] = skill
        result.update({
            "state": LEARNING,
            "skill": skill,
            "taskAnswer": answer,
            "prompt_needed": not bool(answer),
            "reason": "validated",
            "stop_query": True,
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
        if contract and contract.get("verified"):
            city = extract_city(task) or (spec.get("params") or {}).get(
                contract.get("location_parameter") or "location"
            ) or (spec.get("params") or {}).get("city")
            next_spec = buildApiRequest(contract, city, credential)
        else:
            next_spec = repairAuthentication(spec, error, credential)
        if not next_spec:
            session["state"] = FAILED_TERMINAL
            result.update({"state": FAILED_TERMINAL, "reason": error_type, "prompt_needed": False})
            return result
        next_spec = canonicalize_api_spec(next_spec, contract)
        repair = "repair_authentication"
        next_cmd = buildRetryRequest(next_spec)
    elif error_type == MISSING_PARAMETER:
        if contract and contract.get("verified"):
            city = extract_city(task) or (spec.get("params") or {}).get("city") or (
                spec.get("params") or {}).get("location")
            next_spec = buildApiRequest(contract, city, credential)
        else:
            next_spec = repairRequestParameters(spec, error)
        next_spec = canonicalize_api_spec(next_spec, contract)
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
