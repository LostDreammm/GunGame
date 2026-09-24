"""L3 TaskPlanner：自进化任务（issue#21 7 阶段 + 健壮沙箱探索）。

关键修复（2026-09-22 issue#23/#24 复盘）：
1. **任务期 LLM 不限次**：接口文档 errorCode=5 明确"自进化任务执行期间 LLM 不限次且
   不计入每日 3 次额度"。brain 不再按日限额拦截任务 prompt（此前任务1用光 3 次额度 →
   任务2 的 prompt 被静默丢弃，FSM 在空 llmResp 上空转后强制结束 = "读完文档就停摆"）。
2. **健壮探索**：单条命令定位任务文件 + 读取同目录 README/API_DOCS/ws_*/spec.md +
   列出目录权限，替代易失的 find→cat→find→cat 多步（用户提供的模板）。
3. **命令执行加固**：工程类任务自动锚定工作目录（cd "$ws"）+ 主动去 CRLF + ./check；
   从任意命令输出提取 TOKEN 直接提交；LLM 命令自动补 cd（修 teamB r17 漏 cd 的失败）。

阶段：EXPLORE_FILES → (ENGINEER_PROBE) → LLM_LOOP → WAIT_CMD_RESULT/WAIT_LLM
      → SUBMIT_ANSWER → COMPLETED
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..protocol import Turn, distance, submit_answer_command
from ._harvest_data import HARVEST_B64

# ---- 阶段 ----
ST_EXPLORE = "EXPLORE_FILES"
ST_PROBE = "ENGINEER_PROBE"
ST_API_PROBE = "API_PROBE"
ST_LLM = "LLM_LOOP"
ST_WAIT_CMD = "WAIT_CMD_RESULT"
ST_WAIT_LLM = "WAIT_LLM"
ST_SUBMIT = "SUBMIT_ANSWER"
ST_DONE = "COMPLETED"
# 兼容旧名（外部引用）
ST_FIND = ST_EXPLORE
ST_READ = ST_EXPLORE

MAX_NON_JSON = 3       # 连续非 JSON 上限 → 强制结束
MAX_LLM_LOOPS = 8      # LLM 循环上限默认值（实际按任务 timeoutRounds 收紧，见 _loop_limit）
EXPLORE_LIMIT = 8000   # 探索输出保留字符数（需容纳 API_DOCS 全文/密钥）

_FILE_NAME = re.compile(r"[A-Za-z0-9_\-/]+\.(?:md|txt)", re.I)
_JSON_BLOCK = re.compile(r"\{.*\}", re.S)
# 行首锚定（re.M）：只认 check 脚本输出里的独立行 `TOKEN: xxx`，
# 不匹配任务书示例里的 `"token": "xxx"`（issue#26 误提交占位符的根因）。
_TOKEN = re.compile(r"^\s*TOKEN\s*[:：]\s*([A-Za-z0-9_\-]{6,})\s*$", re.M)
_TOKEN_PLACEHOLDERS = {"xxx", "xxxx", "token", "your_token", "your-token", "todo", "none"}
_WS_DIR = re.compile(r"__WS:(\S+)")
_DIR = re.compile(r"__DIR:(\S*)")
_FILE = re.compile(r"__FILE:(\S*)")


def _extract_file_names(text: str) -> list[str]:
    out = []
    for m in _FILE_NAME.findall(text or ""):
        name = m.rsplit("/", 1)[-1]      # 只要文件名
        if name not in out:
            out.append(name)
    return out


# 兼容旧名
_extract_md_names = _extract_file_names


def _parse_llm_json(text: str) -> dict | None:
    """JSON 解析容忍：先直接 loads，失败再用正则提 {...} 块。"""
    t = (text or "").strip()
    if not t:
        return None
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        pass
    for blk in _JSON_BLOCK.findall(t):
        try:
            obj = json.loads(blk)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            continue
    return None


@dataclass
class TaskSession:
    stage: str = ST_EXPLORE
    task_text: str = ""
    task_key: str = ""                                   # SOP 匹配键
    target_name: str = "task_*.md"                       # 待定位的任务文件名
    task_dir: str = ""                                   # 工作目录（工程类=ws 目录）
    explore_output: str = ""                             # 健壮探索的完整输出
    explore_sent: bool = False
    engineer: bool = False
    probe_sent: bool = False
    crlf_fixed: bool = False
    api_probe_sent: bool = False                        # API 类确定性探测已发
    city: str = ""                                       # API 任务城市（探测参数值）
    files: dict = field(default_factory=dict)            # 文件名 → 内容（兼容旧引用）
    pending_cmd: str | None = None
    cmd_history: list = field(default_factory=list)      # [(cmd, result)]
    llm_pending: bool = False
    last_llm_raw: str = ""
    non_json: int = 0
    llm_loops: int = 0
    answer: Any = None
    submitted: bool = False
    need_refine: bool = False
    max_loops: int = MAX_LLM_LOOPS                          # 本任务 LLM 循环上限（按 timeout 收紧）

    def reset(self) -> None:
        self.__init__()


@dataclass
class PlannerOutput:
    prompt: str = ""
    execute_cmd: str = ""
    submit: dict | None = None


class TaskPlanner:
    def __init__(self) -> None:
        self.sop: dict[str, str] = {}       # task_key → SOP 文本（跨任务复用）
        self.completed: int = 0             # 已完成任务数（用于提取 SOP）

    # ---- 主入口 ----
    def work(self, turn: Turn, session: TaskSession) -> PlannerOutput:
        out = PlannerOutput()
        if not turn.phase_task:
            session.reset()
            return out
        if turn.phase_task != session.task_text:      # 新任务 → 重置
            session.reset()
            session.task_text = turn.phase_task
            session.task_key = self._task_key(turn.phase_task)
            names = _extract_file_names(turn.phase_task)
            session.target_name = names[0] if names else "task_*.md"
            session.max_loops = self._loop_limit(turn)
            session.stage = ST_EXPLORE

        # 1. 回收异步结果
        if session.pending_cmd is not None:
            result = turn.last_cmd_result or ""
            session.cmd_history.append((session.pending_cmd, result))
            self._on_cmd_result(session, session.pending_cmd, result)
            session.pending_cmd = None
        if session.llm_pending:
            self._on_llm_result(session, turn.llm_resp or "")
            session.llm_pending = False
        if any(code == 2 for code, _ in turn.errors):   # 答案错 → 重新作答
            session.need_refine = True
            session.submitted = False
            session.answer = None

        # 2. 阶段推进
        return self._advance(turn, session, out)

    def _advance(self, turn: Turn, session: TaskSession, out: PlannerOutput) -> PlannerOutput:
        # LLM 给的命令 → 下发沙盒执行（自动锚定工作目录）
        pend = getattr(session, "_pending_llm_cmd", None)
        if pend:
            session._pending_llm_cmd = None  # type: ignore[attr-defined]
            cmd = self._anchor(pend, session)
            out.execute_cmd = cmd
            session.pending_cmd = cmd
            session._pending_kind = "llm"  # type: ignore[attr-defined]
            session.stage = ST_WAIT_CMD
            return out
        # 提交
        if session.stage == ST_SUBMIT:
            if session.answer is None:
                session.stage = ST_LLM
            else:
                out.submit = submit_answer_command(
                    session.answer if isinstance(session.answer, str)
                    else json.dumps(session.answer, ensure_ascii=False)
                )
                session.submitted = True
                session.stage = ST_DONE
                self.completed += 1
                self._maybe_extract_sop(session)
                return out
        if session.stage == ST_DONE:
            return out
        # 等待中
        if session.stage in (ST_WAIT_CMD, ST_WAIT_LLM):
            return out
        # 健壮探索（单条命令）
        if session.stage == ST_EXPLORE:
            if not session.explore_sent:
                session.explore_sent = True
                cmd = self._explore_cmd(session)
                out.execute_cmd = cmd
                session.pending_cmd = cmd
                session._pending_kind = "explore"  # type: ignore[attr-defined]
                session.stage = ST_WAIT_CMD
                return out
            session.stage = ST_LLM
            return self._advance(turn, session, out)
        # 工程类确定性探测（去 CRLF + ./check）
        if session.stage == ST_PROBE:
            session.probe_sent = True
            cmd = self._probe_cmd(session)
            out.execute_cmd = cmd
            session.pending_cmd = cmd
            session._pending_kind = "probe"  # type: ignore[attr-defined]
            session.stage = ST_WAIT_CMD
            return out
        # API 类确定性探测（认证×参数矩阵 + 收割）
        if session.stage == ST_API_PROBE:
            session.api_probe_sent = True
            cmd = self._api_probe_cmd(session)
            out.execute_cmd = cmd
            session.pending_cmd = cmd
            session._pending_kind = "api_probe"  # type: ignore[attr-defined]
            session.stage = ST_WAIT_CMD
            return out
        # LLM_LOOP
        if session.stage == ST_LLM:
            if session.llm_loops >= session.max_loops:
                session.stage = ST_DONE
                return out
            out.prompt = self._build_prompt(session)
            session.llm_pending = True
            session.llm_loops += 1
            session.stage = ST_WAIT_LLM
            return out
        return out

    # ---- 命令构造 ----
    def _explore_cmd(self, session: TaskSession) -> str:
        """健壮探索（用户模板）：定位任务文件 + 读同目录文档 + 列目录权限。"""
        name = session.target_name or "task_*.md"
        return (
            f'f=$(find /tmp/selfEvolutionTask -type f -iname "{name}" -print -quit 2>/dev/null); '
            f'[ -n "$f" ] || f=$(find / -maxdepth 10 -type f -iname "{name}" -print -quit 2>/dev/null); '
            'd=$(dirname "$f"); echo "__FILE:$f"; echo "=== TASK ==="; cat "$f" 2>/dev/null; '
            'for p in "$d/README.md" "$d/API_DOCS.md" "$d"/ws_*/spec.md; do '
            '[ -f "$p" ] && { echo "=== FILE:$p ==="; cat "$p"; }; done; '
            'echo "=== LIST ==="; find "$d" -maxdepth 3 -type f -printf \'%p %m\\n\' 2>/dev/null; '
            'echo "__DIR:$d"'
        )

    def _probe_cmd(self, session: TaskSession) -> str:
        """工程类确定性探测：定位 check → 去 CRLF → chmod → ls → ./check（拿 FAIL 清单）。"""
        d = session.task_dir or "."
        return (
            f'c=$(find "{d}" -maxdepth 3 -type f -name check -print -quit 2>/dev/null); '
            'w=$(dirname "$c"); echo "__WS:$w"; cd "$w" && sed -i \'s/\\r$//\' check 2>/dev/null; '
            'chmod +x check 2>/dev/null; ls -la; echo "=== CHECK ==="; ./check 2>&1'
        )

    def _api_probe_cmd(self, session: TaskSession) -> str:
        """API 类确定性探测（issue#26）：沙盒跑内嵌 harvester，探测认证×参数矩阵并收割。"""
        d = session.task_dir or "."
        city = session.city or ""
        return (
            f'python3 -c "import base64;exec(base64.b64decode(\'{HARVEST_B64}\'))" '
            f'"{d}" "{city}"'
        )

    @staticmethod
    def _anchor(cmd: str, session: TaskSession) -> str:
        """LLM 命令锚定工作目录：未显式 cd 且未用绝对路径时补 `cd "$dir" && `。"""
        c = (cmd or "").strip()
        if not c or not session.task_dir:
            return c
        if re.search(r"(^|&&|;|\|)\s*cd\s", c) or c.startswith("/"):
            return c
        return f'cd "{session.task_dir}" && {c}'

    # ---- 命令结果 ----
    def _on_cmd_result(self, session: TaskSession, cmd: str, result: str) -> None:
        kind = getattr(session, "_pending_kind", None)
        session._pending_kind = None  # type: ignore[attr-defined]
        if kind == "explore":
            self._on_explore(session, result)
            return
        if kind == "probe":
            self._on_probe(session, result)
            return
        if kind == "api_probe":
            self._on_api_probe(session, result)
            return
        # LLM / 工程修复命令结果
        if self._try_token_submit(session, result):
            return
        if self._maybe_crlf_fix(session, result):
            return
        session.stage = ST_LLM

    def _on_explore(self, session: TaskSession, result: str) -> None:
        session.explore_output = (result or "")[:EXPLORE_LIMIT]
        file_m = _FILE.search(result or "")
        dir_m = _DIR.search(result or "")
        path = (file_m.group(1) if file_m else "").strip()
        if dir_m:
            session.task_dir = dir_m.group(1).strip()
        if path:
            session.files[path.rsplit("/", 1)[-1]] = session.explore_output
        out = session.explore_output
        # 工程类识别：存在 ws_N 工作区且含 check 脚本（题面或探索输出）
        session.engineer = bool(
            re.search(r"ws_\d+", out) and re.search(r"\bcheck\b", out)
        ) or (
            "ws_" in (session.task_text or "") and "check" in (session.task_text or "")
        )
        # API 类识别：文档含 localhost/http + /api/ 路径（issue#26：探测认证×参数矩阵）
        session.city = self._extract_city(session, out)
        api = bool(
            re.search(r"http://localhost:\d+", out)
            or ("/api/" in out and "API_DOCS" in out.upper())
        )
        if self._try_token_submit(session, result):
            return
        if session.engineer and not session.probe_sent:
            session.stage = ST_PROBE
            return
        if api and not session.api_probe_sent:
            session.stage = ST_API_PROBE
            return
        session.stage = ST_LLM

    # 城市名映射：文件名拼音 + 任务文本中的中文城市
    _CITY_PINYIN = {
        "beijing": "北京", "shanghai": "上海", "guangzhou": "广州", "shenzhen": "深圳",
        "xian": "西安", "nanjing": "南京", "hangzhou": "杭州", "chengdu": "成都",
        "tianjin": "天津", "chongqing": "重庆", "wuhan": "武汉", "suzhou": "苏州",
        "xianyang": "咸阳", "luoyang": "洛阳", "kaifeng": "开封", "datong": "大同",
    }
    _CITY_CN = (
        "北京", "上海", "广州", "深圳", "西安", "南京", "杭州", "成都",
        "天津", "重庆", "武汉", "苏州", "洛阳", "开封", "大同", "沈阳",
    )

    @classmethod
    def _extract_city(cls, session: TaskSession, explore: str) -> str:
        """从任务文件名(拼音)或探索输出(中文)提取 API 任务的城市参数值。"""
        for py, cn in cls._CITY_PINYIN.items():
            if py in (session.target_name or "").lower() or py in (session.task_text or "").lower():
                return cn
        for cn in cls._CITY_CN:
            if cn in (explore or "") or cn in (session.task_text or ""):
                return cn
        m = re.search(r"([一-龥]{2,4})市", explore or "")
        return m.group(1) if m else ""

    def _on_probe(self, session: TaskSession, result: str) -> None:
        ws = _WS_DIR.search(result or "")
        if ws and ws.group(1).strip() not in ("", "."):
            session.task_dir = ws.group(1).strip()
        if self._try_token_submit(session, result):
            return
        session.stage = ST_LLM

    def _on_api_probe(self, session: TaskSession, result: str) -> None:
        """API 探测结果 → 事实行喂给 LLM（认证/参数已确定，LLM 只需组答案）。"""
        session.explore_output = (session.explore_output + "\n=== API 探测事实 ===\n"
                                  + (result or "")[:3000])[:EXPLORE_LIMIT]
        session.stage = ST_LLM

    def _try_token_submit(self, session: TaskSession, result: str) -> bool:
        """命令输出里**独立行** `TOKEN: xxx`（真实 check 通过标志）→ 直接提交。

        防误收（issue#26）：任务书示例 `"token": "xxx"` 含占位符，不得当答案。
        """
        m = _TOKEN.search(result or "")
        if not m:
            return False
        value = m.group(1).strip()
        if value.lower() in _TOKEN_PLACEHOLDERS or len(value) < 6:
            return False
        session.answer = json.dumps({"token": value}, ensure_ascii=False)
        session.stage = ST_SUBMIT
        return True

    def _maybe_crlf_fix(self, session: TaskSession, result: str) -> bool:
        """工程类命令因 CRLF/相对路径失败 → 确定性重试（去 CRLF + chmod + ./check）。"""
        if not session.engineer or session.crlf_fixed:
            return False
        low = (result or "").lower()
        if "bad interpreter" not in low and "^m" not in low and "\\r" not in result:
            return False
        session.crlf_fixed = True
        session.stage = ST_PROBE
        return True

    # ---- LLM 结果 ----
    def _on_llm_result(self, session: TaskSession, text: str) -> None:
        session.last_llm_raw = text or ""
        obj = _parse_llm_json(text)
        if obj is None:                                  # 非 JSON
            session.non_json += 1
            if session.non_json >= MAX_NON_JSON:
                session.stage = ST_DONE                  # 强制结束（不提交）
            else:
                session.stage = ST_LLM
            return
        session.non_json = 0
        cmd = str(obj.get("cmd") or "").strip()
        answer = obj.get("answer")
        finished = bool(obj.get("isFinished"))
        if answer not in (None, "", {}):
            session.answer = answer
            session.stage = ST_SUBMIT
            return
        if cmd and not cmd.startswith("cat "):
            session.stage = ST_WAIT_CMD
            session._pending_llm_cmd = cmd  # type: ignore[attr-defined]
            return
        if finished:
            session.stage = ST_DONE
            return
        session.stage = ST_LLM

    # ---- prompt 组装 ----
    def _build_prompt(self, session: TaskSession) -> str:
        parts = ["=== 任务描述 ===", session.task_text or ""]
        if session.explore_output:
            parts += ["=== 沙箱探索结果（任务文件/文档/目录权限） ===", session.explore_output[:6000]]
        for name, content in list(session.files.items())[-4:]:
            if name.startswith("/") or content == session.explore_output:
                continue
            parts += [f"=== 文件: {name} ===", content[:2000]]
        if session.cmd_history:
            cmd, result = session.cmd_history[-1]
            parts += ["=== 上回合命令结果 ===", f"命令: {cmd}", f"结果: {(result or '')[:2500]}"]
        if session.task_dir:
            parts.append(f"工作目录: {session.task_dir}（命令已自动 cd 到此目录）")
        sop = self.sop.get(session.task_key)
        if sop:
            parts += ["=== 参考 SOP（历史任务沉淀） ===", sop[:1500]]
        if session.non_json > 0:
            parts.append("你上一次的返回未按要求仅返回JSON，请勿再犯。")
        parts.append(
            "请只返回 JSON: {\"cmd\": \"\", \"answer\": \"\", \"isFinished\": true|false}\n"
            "说明：cmd=要执行的 shell 命令（单行，如 curl API 调用，为空则不执行）；"
            "answer=最终答案(JSON字符串，为空则未完成)；isFinished=任务是否结束。\n"
            "工程修复类：按 spec.md/check 的 FAIL 清单修复（mkdir -p/chmod/sed 第N行），"
            "完成后运行 ./check，输出含 `TOKEN: xxx` 即代表通过（直接作为 token 答案提交）。\n"
            "API 类：用 curl 调 http://localhost:8899（文档字段可能过期，以实测为准），"
            "认证头与参数名以文档/实测为准；分页用 limit/offset。"
        )
        return "\n".join(parts)

    # ---- SOP 提取（完成前 2 个任务后）----
    def _maybe_extract_sop(self, session: TaskSession) -> None:
        if self.completed > 2:
            return
        cmds = " ; ".join(c for c, _ in session.cmd_history[-5:])
        ans = (session.answer if isinstance(session.answer, str)
               else json.dumps(session.answer, ensure_ascii=False))
        self.sop[session.task_key] = (
            f"任务：{session.task_text[:200]}\n命令序列：{cmds[:600]}\n最终答案：{ans[:400]}"
        )

    @staticmethod
    def _loop_limit(turn: Turn) -> int:
        """按任务 timeoutRounds 收紧 LLM 循环上限（留 2 回合给提交/收尾）。"""
        pioneer = turn.pioneer()
        timeout = 0
        if pioneer is not None:
            for t in turn.tasks:
                if t.timeout_rounds > 0 and distance(pioneer.pos, t.pos) <= 1:
                    timeout = t.timeout_rounds
                    break
        if timeout <= 0:
            return MAX_LLM_LOOPS
        return max(2, min(MAX_LLM_LOOPS, timeout - 2))

    @staticmethod
    def _task_key(task_text: str) -> str:
        """任务类型指纹（用于 SOP 匹配）：取任务文本里的关键词/文件名。"""
        names = _extract_file_names(task_text)
        if names:
            return names[0].rsplit("_", 1)[-1].rsplit(".", 1)[0]  # task_1_alpha.md → alpha
        return (task_text or "")[:20]


def _iter_json_objects(text: str):
    """迭代文本里所有平衡的 {...} 子串并尝试解析。"""
    depth, start = 0, -1
    for i, ch in enumerate(text or ""):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        yield json.loads(text[start:i + 1])
                    except (json.JSONDecodeError, ValueError):
                        pass


_ANSWER_KEYS = {
    "token", "total_count", "totalcount", "world_heritage_count", "types",
    "oldest_era", "city", "answer", "result", "value", "count", "total",
}
_ERROR_KEYS = {"status", "error", "message", "code"}


def _looks_like_answer(obj, task_text: str) -> bool:
    """命令输出里的 JSON 是否"像答案"（防误收 API 原始记录/错误报文）。"""
    if not isinstance(obj, dict) or not obj:
        return False
    keys = {str(k).lower() for k in obj}
    if "status" in keys and str(obj.get("status", "")).lower() in ("error", "fail", "failed"):
        return False
    if keys & _ERROR_KEYS and "message" in keys and not (keys & _ANSWER_KEYS):
        return False
    if keys & _ANSWER_KEYS:
        return True
    low = (task_text or "").lower()
    return any(len(k) >= 2 and k in low for k in keys)
