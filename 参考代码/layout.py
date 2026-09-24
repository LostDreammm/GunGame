"""L3 BaseLayoutPlanner：动态 FRONT 的基地环形布局。

布局规范（STRATEGY_DECISIONS #8，以 FRONT=WEST 为例）：
    |空白|围墙|围墙|围墙|围墙|围墙|
    |空白|空白|空白|空白|空白|围墙|
    |空白|炮台|基地|基地|空白|围墙|
    |空白|人物|基地|基地|空白|围墙|
    |空白|炮台|炮台|空白|空白|围墙|
    |空白|围墙|围墙|围墙|围墙|围墙|
- 基地 2×2；墙在距基地切比雪夫距离 2 的三侧成环（≈14 格，开口侧不建）。
- **FRONT = 开口/炮台/通道侧，背向机器人来向；墙环朝向来敌**：
  challenger 左上基地 → 机器人从右(东)来 → 墙在 右/上/下，FRONT=西；
  defender 右下基地 → 机器人从左(西)来 → 墙在 左/上/下，FRONT=东。
  （实战反馈纠正：此前 FRONT 朝向来敌，炮塔直接挨打，方向建反。）
- 3 炮台与控制点(CP)两两相邻，开拓者一站控三炮；火箭无视阻挡跨墙输出。
- 坐标全部由变换生成，零硬编码。
"""
from __future__ import annotations

from dataclasses import dataclass

from ..buildable_map import BuildableMap
from ..protocol import Pos, Turn, distance, in_bounds, station_footprint

# 规范坐标系：基地锚点 (xmin, ymin) = (station.x, station.y-1)，
# 基地四格 {(0,0),(1,0),(0,1),(1,1)}，FRONT=WEST。
CANONICAL_TURRETS = (Pos(-1, 1), Pos(-1, -1), Pos(0, -1))
CANONICAL_CP = Pos(-1, 0)

# 各 FRONT 朝向的坐标变换（把规范 (dx,dy) 映射到目标朝向，基地四格映射到自身）
TRANSFORMS = {
    "W": lambda dx, dy: (dx, dy),
    "E": lambda dx, dy: (1 - dx, dy),
    "N": lambda dx, dy: (dy, 1 - dx),
    "S": lambda dx, dy: (1 - dy, dx),
}


def _canonical_walls() -> list[Pos]:
    """距离基地 2 的墙环，去掉 FRONT(dx=-2) 开口列，共 14 格。"""
    cells = []
    for dx in range(-2, 4):
        for dy in range(-2, 4):
            if -1 <= dx <= 2 and -1 <= dy <= 2:
                continue  # 内圈
            if dx == -2:
                continue  # FRONT 开口
            cells.append(Pos(dx, dy))
    return cells


CANONICAL_WALLS = tuple(_canonical_walls())


def choose_front(turn: Turn) -> str:
    """FRONT（开口侧）默认朝向：**背向**地图中心的 dominant axis。

    机器人出生点全局唯一且每晚固定（已确认），实战证实来自图心方向：
    开口/炮台置于背侧（角落方向），墙环朝图心迎敌。
    """
    station = turn.station()
    if station is None:
        return "S"
    cx = station.pos.x + 0.5
    cy = station.pos.y - 0.5
    dx = (turn.width - 1) / 2 - cx
    dy = (turn.height - 1) / 2 - cy
    if abs(dx) >= abs(dy):
        return "W" if dx > 0 else "E"  # 中心在东侧 → 开口朝西（背向）
    return "S" if dy > 0 else "N"


@dataclass(frozen=True, slots=True)
class BaseLayout:
    front: str
    turret_cells: tuple[Pos, ...]
    control_point: Pos
    wall_cells: tuple[Pos, ...]  # 按距 CP 由近到远（开口侧优先建造）
    repair_post: Pos | None = None  # 修理工夜间抢修就位点（内圈、邻墙最多、非炮台/CP）


def _ring_cells(xmin: int, ymin: int, dist: int) -> list[Pos]:
    """距 2×2 基地锚点切比雪夫距离恰好为 dist 的全部格子（按 x,y 排序）。"""
    base = {(0, 0), (1, 0), (0, 1), (1, 1)}
    cells = []
    for dx in range(-dist, 2 + dist):
        for dy in range(-dist, 2 + dist):
            if min(max(abs(dx - bx), abs(dy - by)) for bx, by in base) == dist:
                cells.append(Pos(xmin + dx, ymin + dy))
    cells.sort(key=lambda p: (p.x, p.y))
    return cells


def compute_layout(
    turn: Turn,
    front: str,
    buildable: BuildableMap | None = None,
) -> BaseLayout | None:
    station = turn.station()
    if station is None:
        return None
    xmin, ymin = station.pos.x, station.pos.y - 1
    tf = TRANSFORMS[front]
    base_cells = set(station_footprint(station.pos))
    enemy_cells = {cell for unit in turn.enemy for cell in turn.footprint(unit)}
    our_weapon_pos = {w.pos for w in turn.weapons()}
    our_wall_pos = {w.pos for w in turn.walls()}

    def to_abs(p: Pos) -> Pos:
        dx, dy = tf(p.x, p.y)
        return Pos(xmin + dx, ymin + dy)

    def usable(pos: Pos, kind: str) -> bool:
        if not in_bounds(pos, turn.width, turn.height):
            return False
        if not turn.land(pos):  # 中立元素（矿/小贩/商店/任务点）不可建
            return False
        if pos in base_cells or pos in enemy_cells:
            return False
        if kind == "weapon" and pos in our_wall_pos:
            return False
        if kind == "wall" and pos in our_weapon_pos:
            return False
        if buildable is not None and not buildable.is_usable(pos, kind, turn.round_no):
            return False
        return True

    # 蓝区（距基地1格）= CP/炮台候选；规范位优先，凑不齐 3 炮则迁移 CP
    ring1 = [
        pos for pos in _ring_cells(xmin, ymin, 1)
        if pos not in enemy_cells and turn.land(pos)
    ]
    canonical_cp = to_abs(CANONICAL_CP)
    cp_candidates = []
    if canonical_cp in ring1:
        cp_candidates.append(canonical_cp)
    others = [p for p in ring1 if p != canonical_cp]
    others.sort(key=lambda p: (distance(p, canonical_cp), p.x, p.y))
    cp_candidates += others

    def turrets_for(cp: Pos) -> list[Pos]:
        cands: list[Pos] = []
        if cp == canonical_cp:
            cands.extend(to_abs(p) for p in CANONICAL_TURRETS)
        cands.extend(p for p in ring1 if p != cp and distance(p, cp) <= 1)
        seen: set[Pos] = set()
        out: list[Pos] = []
        for pos in cands:
            if pos in seen or pos == cp or not usable(pos, "weapon"):
                continue
            seen.add(pos)
            out.append(pos)
            if len(out) == 3:
                break
        return out

    cp: Pos | None = None
    turrets: list[Pos] = []
    for cand in cp_candidates:
        picked = turrets_for(cand)
        if len(picked) == 3:
            cp, turrets = cand, picked
            break
        if len(picked) > len(turrets):
            cp, turrets = cand, picked
    if cp is None:
        return None

    walls = [p for p in (to_abs(c) for c in CANONICAL_WALLS) if usable(p, "wall")]
    # 建造优先级：正面（迎敌侧，离 CP 最远）优先，再到侧面（实战复盘：正面必须先封）
    walls.sort(key=lambda p: (-distance(p, cp), p.x, p.y))
    # 修理工夜间抢修就位点（issue#26）：内圈（ring1）中非炮台/非 CP 的可站格，
    # 选在"邻接墙最多"的位置（=贴墙侧，上下走动即可覆盖整圈墙）；候选为空则退化为 CP。
    occupied_role_cells = set(turrets) | {cp}
    repair_post: Pos | None = None
    best_score = -1
    for cell in ring1:
        if cell in occupied_role_cells or cell in base_cells:
            continue
        if not turn.land(cell):
            continue
        adj = sum(1 for nb in cell.neighbours() if nb in set(walls))
        if adj > best_score:
            best_score = adj
            repair_post = cell
    if repair_post is None:
        repair_post = cp
    return BaseLayout(front, tuple(turrets), cp, tuple(walls), repair_post)
