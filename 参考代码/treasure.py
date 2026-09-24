"""L3 TreasurePlanner：长上下文类任务（民间传闻 → 祭坛宝藏）。

机制（任务书 §5.2）：民间传闻逐日累积；开拓者据线索推断**宝藏地点/开启条件(祭品)/开启时间**，
携带任务用品到祭坛 `summonTreasure`。全图唯一，成功开启后不重复得奖。

地点/祭品无法确定性推断 → 本实现为**框架 + LLM 推断**：
- 逐日累积 folkLegends（去重）；
- 线索足够且无 ready 计划时，用 LLM 推断（任务期免费；非任务期用每日少量额度）；
- **只有得到 `ready` 计划才行动**：备齐祭品（缺则去商店买）→ 走到祭坛旁 → summonTreasure。

边界：不做任何无把握的献祭（避免浪费金币/用品）；未 ready 时返回 None。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..path import step_toward
from ..protocol import (
    Pos,
    Turn,
    Unit,
    buy_command,
    distance,
    move_command,
    summon_treasure_command,
)

# 任务用品英文名（任务书 §4.6.3）——LLM 返回值须落在此集合内
TREASURE_ITEMS = (
    "AcientTablet", "StarSand", "FlameBreath", "FrostPotion", "ThornAmulet", "IronWhistle",
)
_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


@dataclass
class TreasurePlan:
    location: Pos | None = None
    items: tuple[str, ...] = ()
    day: int = 0
    ready: bool = False
    raw: str = ""

    def dump(self) -> dict:
        return {
            "location": self.location.dump() if self.location else None,
            "items": list(self.items),
            "day": self.day,
            "ready": self.ready,
        }


class TreasurePlanner:
    def __init__(self) -> None:
        self.legends: list[str] = []
        self.plan = TreasurePlan()
        self.attempted = False

    # ---- 线索累积 ----
    def observe(self, folk: str) -> bool:
        text = (folk or "").strip()
        if not text or text in self.legends:
            return False
        self.legends.append(text)
        return True

    def needs_inference(self) -> bool:
        return bool(self.legends) and not self.plan.ready and not self.attempted

    # ---- LLM 推断 ----
    def prompt(self) -> str:
        return (
            "=== 民间传闻（逐日累积）===\n" + "\n".join(self.legends[-12:]) +
            "\n请据线索推断宝藏：祭坛坐标(x,y)、需献祭的任务用品(英文名)、开启天数。"
            '只返回 JSON：{"x":<int>,"y":<int>,"items":["AcientTablet",...],"day":<int>,"ready":<bool>}。'
            "信息不足时 ready=false。可用用品：" + ", ".join(TREASURE_ITEMS)
        )

    def apply_llm(self, response: str) -> bool:
        obj = None
        t = (response or "").strip()
        for blob in [t, *_JSON_BLOCK.findall(t)]:
            try:
                parsed = json.loads(blob)
                if isinstance(parsed, dict):
                    obj = parsed
                    break
            except (json.JSONDecodeError, ValueError):
                continue
        if not obj:
            return False
        try:
            x, y = int(obj.get("x")), int(obj.get("y"))
        except (TypeError, ValueError):
            return False
        items = tuple(
            str(i) for i in (obj.get("items") or []) if str(i) in TREASURE_ITEMS
        )
        try:
            day = int(obj.get("day") or 0)
        except (TypeError, ValueError):
            day = 0
        ready = bool(obj.get("ready")) and bool(items)
        self.plan = TreasurePlan(Pos(x, y), items, day, ready, (response or "")[:200])
        return True

    def record_attempt(self) -> None:
        self.attempted = True

    # ---- 行动 ----
    def cmd(self, turn: Turn, pioneer: Unit, shop: Pos | None, ctx) -> dict[str, Any] | None:
        """ready 计划才行动：备祭品 → 到祭坛 → summonTreasure。"""
        if not self.plan.ready or self.attempted or self.plan.location is None:
            return None
        loc = self.plan.location
        missing = [it for it in self.plan.items if it not in pioneer.backpack]
        if missing:
            if shop is None:
                return None
            item = missing[0]
            price = turn.shop_prices.get(item, 15)
            if turn.gold < price:
                return None
            if pioneer.pos != shop and distance(pioneer.pos, shop) <= 1:
                return buy_command(item, 1)
            step = step_toward(turn, pioneer, shop, ctx.reserved)
            return move_command(step) if step is not None else None
        if distance(pioneer.pos, loc) <= 1:
            self.attempted = True
            return summon_treasure_command(loc, list(self.plan.items))
        step = step_toward(turn, pioneer, loc, ctx.reserved)
        return move_command(step) if step is not None else None