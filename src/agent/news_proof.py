"""Literal sentence facts used by TreasureMemory consistency checks."""
import re


ITEM_ALIASES = {
    "AcientTablet": "古符石板",
    "StarSand": "星辰之沙",
    "FlameBreath": "烈焰之息",
    "FrostPotion": "寒霜药剂",
    "ThornAmulet": "荆棘护符",
    "IronWhistle": "回音铁哨",
}
_COUNT_WORDS = {"两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6}


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _sentences(text):
    if not isinstance(text, str) or not text.strip():
        return []
    parts = re.split(r"(?<=[。！？.!?])\s*", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _intersect(values, operation, operand):
    values = set(values)
    if operation in ("eq", "exact"):
        if isinstance(operand, (set, frozenset, list, tuple)):
            return values == set(operand)
        return operand in values
    if operation in ("in", "pool"):
        if isinstance(operand, (set, frozenset, list, tuple)):
            return bool(values) and values <= set(operand)
        return operand in values
    if operation == "ge":
        return bool(values) and min(values) >= operand
    if operation == "le":
        return bool(values) and max(values) <= operand
    if operation == "range" and isinstance(operand, (list, tuple)) and len(operand) == 2:
        low, high = operand
        return bool(values) and all(low <= item <= high for item in values)
    if operation == "exclude":
        if isinstance(operand, (set, frozenset, list, tuple)):
            return not (values & set(operand))
        return operand not in values
    return True


def _parse(sentence, day, origin):
    """Return (field, operation, operand) or None for one source sentence."""
    if not isinstance(sentence, str) or not sentence.strip():
        return None
    text = sentence.strip()
    negated = bool(re.search(r"(?:不是|并非|不要|禁止|否认|谣言|没有|并未)", text))
    coord = re.search(r"[（(]\s*(\d+)\s*[,，]\s*(\d+)\s*[)）]", text)
    if coord and not negated:
        return ("pos", "in", {(int(coord.group(1)), int(coord.group(2)))})
    found = set()
    for ident, alias in ITEM_ALIASES.items():
        if ident in text or alias in text:
            found.add(ident)
    if found:
        if negated:
            return ("items", "exclude", found)
        if re.search(r"(?:仅需|只需|恰好|且仅|全部祭品|完整祭品)", text):
            return ("items", "exact", found)
        if re.search(r"(?:之一|任意|或者)", text):
            return ("items", "pool", found)
        count = None
        match = re.search(r"([两二三四五六])\s*(?:件|钥|种)", text)
        if match:
            count = _COUNT_WORDS.get(match.group(1))
        if count:
            return ("items", "count", count)
        return ("items", "require", found)
    day_match = re.search(r"第\s*([1-9]|10)\s*天", text)
    if day_match and re.search(r"(?:开启|打开|献祭|祭坛)", text) and not negated:
        open_day = int(day_match.group(1))
        start = origin if _integer(origin) else 1
        return ("open_round", "eq", {start + (open_day - 1) * 130})
    return None
