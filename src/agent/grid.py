"""Navigation and official weapon commands; no environment simulation.

Coordinates use the taskbook's lower-left origin. A station's position is its
top-left cell. Ballistic cell membership is not specified textually by v1.0:
this planner conservatively intersects the center-to-center segment with robot
cell squares, including corner contacts. The judger's rasterization may differ.
Only robots intercept bullets in the supplied textual weapon rules. Damage is
settled at turn end, so projected kills never remove ballistic obstructions.
"""

import logging
from collections import deque
from itertools import combinations


LOG = logging.getLogger(__name__)
PEOPLE = frozenset(("worker", "pioneer"))
GUNS = frozenset(("gatling", "railgun", "rocket"))
ROBOT_POINTS = {"smallRobot": 1, "middleRobot": 2, "largeRobot": 4, "bossRobot": 10}
ROBOT_POWER = {"smallRobot": 5, "middleRobot": 10, "largeRobot": 20, "bossRobot": 40}

# Taskbook coordinates: origin at lower-left, +x right, +y up.
# A station's recorded position is the top-left cell of its 2x2 footprint.
ROCKET_TURRET_COUNT = 3
TARGET_WALL_COUNT = 12
WALL_STONE_COST = 1
BASE_REGION_TOP_LEFT = "top_left"
BASE_REGION_BOTTOM_RIGHT = "bottom_right"
DAY_LENGTH = 130
DAYTIME_LENGTH = 70


def pos(value):
    """Accept a role/zone, a Pos dict, or an (x, y) pair."""
    if isinstance(value, dict):
        point = value.get("pos", value)
        return int(point["x"]), int(point["y"])
    return int(value[0]), int(value[1])


def distance(a, b):
    a, b = pos(a), pos(b)
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def footprint(role):
    x, y = pos(role)
    if role.get("roleType") == "station":
        return {(x, y), (x + 1, y), (x, y - 1), (x + 1, y - 1)}
    return {(x, y)}


def _living(roles):
    return [r for r in roles if isinstance(r, dict) and int(r.get("health", 1)) > 0]


class World:
    def __init__(self, data, config):
        self.data = data
        self.config = config
        info = data.get("mapInfo") or {}
        self.width, self.height = int(info.get("width", 0)), int(info.get("height", 0))
        if not (1 <= self.width <= 256 and 1 <= self.height <= 256):
            raise ValueError("Map dimensions must be between 1 and 256")
        our_team = data.get("teamOur") or {}
        self.ours = _living(our_team.get("roles") or [])
        self.enemy = _living((data.get("teamEnemy") or {}).get("roles") or [])
        self.robots = _living((data.get("robot") or {}).get("roles") or [])
        self.zones = info.get("zones") or []
        self.people = sorted((r for r in self.ours if r.get("roleType") in PEOPLE), key=lambda r: str(r["id"]))
        self.guns = sorted((r for r in self.ours if r.get("roleType") in GUNS), key=lambda r: str(r["id"]))
        self.base = next((r for r in self.ours if r.get("roleType") == "station"), None)
        self.blocked = set()
        for entity in self.ours + self.enemy + self.robots:
            self.blocked.update(footprint(entity))
        for zone in self.zones:
            self.blocked.add(pos(zone))
        # Task entries reinforce blocking if a platform observation omits a zone.
        for task in our_team.get("playerTasks") or []:
            if task.get("taskPosition"):
                self.blocked.add(pos(task["taskPosition"]))
        phase = max(0, int(data.get("roundNo", 1)) - int(config.get("round_origin", 1)))
        self.day = phase // DAY_LENGTH + 1
        self.phase_in_day = phase % DAY_LENGTH
        self.night = self.phase_in_day >= DAYTIME_LENGTH
        self.remaining_day = max(0, DAYTIME_LENGTH - self.phase_in_day)
        self.remaining_night = max(0, DAY_LENGTH - self.phase_in_day) if self.night else (
            DAY_LENGTH - DAYTIME_LENGTH)
        self.remaining_cycle = max(0, DAY_LENGTH - self.phase_in_day)
        self._paths = {}

    def in_bounds(self, point):
        x, y = pos(point)
        return 0 <= x < self.width and 0 <= y < self.height

    def neighbors(self, point):
        x, y = pos(point)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                target = (x + dx, y + dy)
                if (dx or dy) and self.in_bounds(target):
                    yield target

    def interaction_cells(self, target_cells):
        cells = set(pos(p) for p in target_cells)
        return {n for cell in cells for n in self.neighbors(cell)} - cells

    def path(self, role, goals, reserved=()):
        """Return a shortest eight-direction path including start, or None.

        Current other units are conservative obstacles; no ally swaps or
        movement chains are assumed. Reserve already selected destinations.
        Diagonal movement through two orthogonally adjacent obstacles is legal.
        """
        start = pos(role)
        reserved = frozenset(pos(p) for p in reserved)
        goals = {pos(p) for p in goals if self.in_bounds(p)}
        blocked = (self.blocked - {start}) | reserved
        goals -= blocked
        if not goals or not self.in_bounds(start):
            return None
        if start in goals:
            return [start]
        key = (start, reserved)
        if key not in self._paths:
            parents = {start: None}
            queue = deque([start])
            order = []
            while queue:
                point = queue.popleft()
                order.append(point)
                for nxt in self.neighbors(point):
                    if nxt not in parents and nxt not in blocked:
                        parents[nxt] = point
                        queue.append(nxt)
            self._paths[key] = parents, order
        parents, order = self._paths[key]
        target = next((point for point in order if point in goals), None)
        if target is None:
            return None
        route = [target]
        while parents[route[-1]] is not None:
            route.append(parents[route[-1]])
        return list(reversed(route))


def defense_assignments(world, people):
    """Match up to three live controllers to reachable guns.

    Lexicographic objective: maximize distinct guns covered, minimize total
    walking distance, then stable person/gun IDs. A final movement reservation
    pass by the caller prevents two controllers choosing the same square.
    """
    live = {str(r["id"]): r for r in world.people}
    people = sorted((live[str(p["id"])] for p in people if str(p["id"]) in live), key=lambda r: str(r["id"]))[:3]
    guns = world.guns[:3]
    costs = {}
    for person in people:
        for gun in guns:
            route = world.path(person, world.interaction_cells(footprint(gun)))
            if route is not None:
                costs[(str(person["id"]), str(gun["id"]))] = len(route) - 1
    best = [None, {}]

    def visit(index, chosen, used, cost):
        if index == len(people):
            signature = tuple(sorted((p, str(g["id"])) for p, g in chosen.items()))
            score = (-len(chosen), cost, signature)
            if best[0] is None or score < best[0]:
                best[:] = [score, dict(chosen)]
            return
        ident = str(people[index]["id"])
        for gun in guns:
            gid = str(gun["id"])
            if gid not in used and (ident, gid) in costs:
                chosen[ident] = gun
                visit(index + 1, chosen, used | {gid}, cost + costs[(ident, gid)])
                del chosen[ident]
        visit(index + 1, chosen, used, cost)

    visit(0, {}, set(), 0)
    return best[1]


def _segment_entry(start, end, cell):
    """First t in [0,1] where the ray intersects a closed robot cell."""
    lo, hi = 0.0, 1.0
    for axis in (0, 1):
        direction = end[axis] - start[axis]
        lower, upper = cell[axis] - 0.5, cell[axis] + 0.5
        if direction == 0:
            if not lower <= start[axis] <= upper:
                return None
        else:
            a, b = (lower - start[axis]) / direction, (upper - start[axis]) / direction
            lo, hi = max(lo, min(a, b)), min(hi, max(a, b))
            if lo > hi:
                return None
    return lo if hi > 0 else None


def _ray_robots(start, target, robots):
    hits = []
    for robot in robots:
        entry = _segment_entry(start, target, pos(robot))
        if entry is not None:
            hits.append((entry, str(robot["id"]), robot))
    return [hit[2] for hit in sorted(hits, key=lambda item: (item[0], item[1]))]


def _damage(gun, target, robots, position_index=None):
    kind = gun["roleType"]
    if position_index is not None:
        if kind == "rocket":
            cells = ((target[0] + dx, target[1] + dy)
                     for dx in (-1, 0, 1) for dy in (-1, 0, 1))
        else:
            start = pos(gun)
            cells = ((x, y) for x in range(min(start[0], target[0]), max(start[0], target[0]) + 1)
                     for y in range(min(start[1], target[1]), max(start[1], target[1]) + 1))
        robots = [r for point in cells for r in position_index.get(point, ())]
    if kind == "rocket":
        return {str(r["id"]): 20 if pos(r) == target else 10 for r in robots if distance(r, target) <= 1}
    hits = _ray_robots(pos(gun), target, robots)
    if kind == "gatling":
        return {str(hits[0]["id"]): 10} if hits else {}
    energy = max(0, int(gun.get("attackPower", 0)))
    result = {}
    for robot in hits:
        ident = str(robot["id"])
        amount = min(energy, int(robot["health"]))
        result[ident] = amount
        energy -= amount
        if energy <= 0:
            break
    return result


def _robot_weights(world):
    base_cells = footprint(world.base) if world.base else set()
    team = (world.data.get("teamOur") or {}).get("type")
    weights = {}
    for robot in world.robots:
        near = min((distance(robot, point) for point in base_cells), default=20)
        target_team = robot.get("targetTeam")
        ours = team in ("challenger", "defender") and target_team == team
        unknown = target_team not in ("challenger", "defender")
        # Near enemies with large attacks deserve more attention, while kill
        # bonuses preserve the direct scoring value of damaged small robots.
        weight = 1.0 + (0.7 if ours else 0.0)
        if ours:
            weight += max(0, 10 - near) * 0.2
            if near <= 4 and robot.get("abnormalState") != "dizzy":
                # A robot already in attack range can decide the match before
                # a distant high-point target becomes relevant.
                weight += 4.0 + ROBOT_POWER.get(robot.get("roleType"), 5) / 8.0
        elif unknown:
            # Some supplied observations omit targetTeam. Proximity remains
            # evidence of danger, with lower confidence than a confirmed target.
            weight += 0.25 + max(0, 10 - near) * 0.08
            if near <= 4 and robot.get("abnormalState") != "dizzy":
                weight += ROBOT_POWER.get(robot.get("roleType"), 5) / 30.0
        weights[str(robot["id"])] = weight
    return weights


def _target_confidence(world, robot):
    """How strongly the observation says this robot threatens our team."""
    team = (world.data.get("teamOur") or {}).get("type")
    target = robot.get("targetTeam")
    if target == team and team in ("challenger", "defender"):
        return 1.0
    if target in ("challenger", "defender"):
        return 0.0
    # Old samples omit targetTeam. Keep a conservative defensive allowance.
    return 0.5


def projected_base_damage(world, horizon=5):
    """Conservative short-horizon pressure estimate used for policy gating.

    Robot path and target selection are not exposed by the API, so this is an
    observable proxy rather than a simulator. It intentionally ignores robots
    confirmed to target the other team.
    """
    if not world.base or horizon <= 0:
        return 0.0
    cells = footprint(world.base)
    total = 0.0
    for robot in world.robots:
        confidence = _target_confidence(world, robot)
        if confidence <= 0:
            continue
        near = min(distance(robot, cell) for cell in cells)
        turns_to_range = max(0, near - 3)
        attack_turns = max(0, min(horizon, world.remaining_night) - turns_to_range)
        # No remaining-stun field is exposed. Credit only one certain skipped
        # action instead of assuming the full original five turns remain.
        if robot.get("abnormalState") == "dizzy":
            attack_turns = max(0, attack_turns - 1)
        total += confidence * attack_turns * ROBOT_POWER.get(robot.get("roleType"), 5)
    return total


def best_area_effect(world, item_name):
    """Return (target, defensive value, affected ids) for Bomb/DizzyWeapon."""
    if item_name not in ("Bomb", "DizzyWeapon") or not world.base:
        return None, 0.0, set()
    base_cells = footprint(world.base)
    candidates = set()
    for robot in world.robots:
        if _target_confidence(world, robot) <= 0:
            continue
        rx, ry = pos(robot)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                point = (rx + dx, ry + dy)
                if world.in_bounds(point):
                    candidates.add(point)
    best = (None, 0.0, set())
    for target in sorted(candidates):
        affected = [robot for robot in world.robots
                    if distance(robot, target) <= 1 and _target_confidence(world, robot) > 0]
        value = 0.0
        ids = set()
        for robot in affected:
            confidence = _target_confidence(world, robot)
            near = min(distance(robot, cell) for cell in base_cells)
            if near > 8:
                continue
            ident = str(robot["id"])
            ids.add(ident)
            power = ROBOT_POWER.get(robot.get("roleType"), 5)
            imminent = max(0, 6 - near)
            if item_name == "Bomb":
                damage = min(100, max(0, int(robot.get("health", 0))))
                value += confidence * (damage + imminent * power)
                if 0 < int(robot.get("health", 0)) <= 100:
                    value += 10 * ROBOT_POINTS.get(robot.get("roleType"), 1)
            elif robot.get("abnormalState") != "dizzy":
                prevented_turns = max(0, 5 - max(0, near - 3))
                value += confidence * power * prevented_turns
        candidate = (target, value, ids)
        if candidate[1] > best[1] or (candidate[1] == best[1] and candidate[0] and
                                      (best[0] is None or candidate[0] < best[0])):
            best = candidate
    return best


def _value(damage, projected, weights, robots_by_id):
    result = 0.0
    for ident, amount in damage.items():
        remaining = projected.get(ident, 0)
        result += min(remaining, amount) * weights[ident]
        if 0 < remaining <= amount:
            result += 10 * ROBOT_POINTS.get(robots_by_id[ident].get("roleType"), 1)
    return result


def _apply_projection(projected, damage):
    for ident, amount in damage.items():
        projected[ident] = max(0, projected.get(ident, 0) - amount)


def _cone_compatible(origin, target, previous):
    vector = target[0] - origin[0], target[1] - origin[1]
    return all(vector[0] * (p[0] - origin[0]) + vector[1] * (p[1] - origin[1]) >= 0 for p in previous)


def plan_attacks(world, assignments):
    """Produce official weapon-keyed commands plus occupied controller IDs."""
    if not world.night:
        return {}, set()
    people = {str(p["id"]): p for p in world.people}
    guns = {str(g["id"]): g for g in world.guns}
    robots_by_id = {str(r["id"]): r for r in world.robots}
    projected = {ident: int(r["health"]) for ident, r in robots_by_id.items()}
    weights = _robot_weights(world)
    position_index = {}
    for robot in world.robots:
        position_index.setdefault(pos(robot), []).append(robot)
    commands, used = {}, set()
    ordered = sorted(assignments.items(), key=lambda item: (str(item[1]["id"]), str(item[0])))
    for controller_id, assigned in ordered:
        controller_id, gid = str(controller_id), str(assigned["id"])
        controller, gun = people.get(controller_id), guns.get(gid)
        if not controller or not gun or gid in commands or controller_id in used:
            continue
        if distance(controller, gun) != 1 or int(gun.get("cooldown", 0)) > 0:
            continue
        kind, origin = gun["roleType"], pos(gun)
        attack_range = max(0, int(gun.get("attackRange", 0)))
        level = max(1, min(3, int(gun.get("level", 1))))
        count = 1 if kind == "railgun" else level
        candidates = {pos(r) for r in world.robots if 0 < distance(gun, r) <= attack_range}
        if kind == "rocket":
            # Empty centers can catch several robots, or splash a robot one
            # cell outside direct range. Never aim at the weapon's own square.
            for robot in world.robots:
                if distance(gun, robot) <= attack_range + 1:
                    for point in world.neighbors(pos(robot)):
                        if 0 < distance(origin, point) <= attack_range:
                            candidates.add(point)
        candidates = sorted(candidates)
        if not candidates:
            continue
        damage_cache = {target: _damage(gun, target, world.robots, position_index) for target in candidates}
        # Different first shots explore the legal gatling cones. Other guns
        # have no cone restriction and use the best marginal damage per shot.
        anchors = candidates if kind == "gatling" else [None]
        best_value, best_targets, best_projection = 0.0, [], None
        for anchor in anchors:
            local, selected, total = dict(projected), [], 0.0
            for index in range(count):
                available = [anchor] if anchor is not None and index == 0 else candidates
                chosen, value = None, -1.0
                for target in available:
                    if kind == "gatling" and not _cone_compatible(origin, target, selected):
                        continue
                    gain = _value(damage_cache[target], local, weights, robots_by_id)
                    if gain > value:
                        chosen, value = target, gain
                if chosen is None:
                    break
                selected.append(chosen)
                total += value
                _apply_projection(local, damage_cache[chosen])
            if len(selected) == count and total > best_value:
                best_value, best_targets, best_projection = total, selected, local
        if best_targets:
            commands[gid] = {"action": "attack", "controllerId": controller_id,
                             "targetPos": [{"x": x, "y": y} for x, y in best_targets]}
            used.add(controller_id)
            projected = best_projection
    return commands, used


"""Construction from the confirmed concentric rings around the observed base.

The station's position is its top-left cell in a lower-left-origin coordinate
system. Its surrounding 4 x 4 ring has twelve weapon cells; the next 6 x 6
ring has twenty wall cells. The HTTP snapshot supplies the current base and
occupants, so no absolute build coordinates or map-color image are required.
"""

import math
from collections import Counter



WEAPON_GOLD = 25
DEFAULT_WEAPON_PLAN = ('rocket', 'rocket', 'rocket')


def building_rings(base, width, height):
    """Return in-bounds (blue weapon cells, yellow wall cells)."""
    if not base:
        return set(), set()
    x, y = pos(base)
    green = {(x, y), (x + 1, y), (x, y - 1), (x + 1, y - 1)}
    inner = {(px, py) for px in range(x - 1, x + 3)
             for py in range(y - 2, y + 2)}
    outer = {(px, py) for px in range(x - 2, x + 4)
             for py in range(y - 3, y + 3)}
    bounds = lambda cell: 0 <= cell[0] < width and 0 <= cell[1] < height
    return ({cell for cell in inner - green if bounds(cell)},
            {cell for cell in outer - inner if bounds(cell)})


def base_region(base, width, height):
    """Classify the station as top-left or bottom-right using map centre.

    Station centre is (x+0.5, y-0.5) because `pos` is the top-left cell and +y is up.
    """
    if not base or width <= 0 or height <= 0:
        return BASE_REGION_TOP_LEFT
    sx, sy = pos(base)
    cx, cy = sx + 0.5, sy - 0.5
    mx, my = (width - 1) / 2.0, (height - 1) / 2.0
    if cx <= mx and cy >= my:
        return BASE_REGION_TOP_LEFT
    if cx >= mx and cy <= my:
        return BASE_REGION_BOTTOM_RIGHT
    return BASE_REGION_TOP_LEFT if cx <= mx else BASE_REGION_BOTTOM_RIGHT


def _in_bounds(cell, width, height):
    return 0 <= cell[0] < width and 0 <= cell[1] < height


def _cluster_connected(cells):
    cells = [pos(cell) for cell in cells]
    if len(cells) <= 1:
        return True
    remaining = set(cells[1:])
    stack = [cells[0]]
    while stack:
        current = stack.pop()
        for other in list(remaining):
            if distance(current, other) <= 1:
                remaining.remove(other)
                stack.append(other)
    return not remaining


def _eight_neighbors(cell, width, height):
    x, y = pos(cell)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx or dy:
                nxt = (x + dx, y + dy)
                if _in_bounds(nxt, width, height):
                    yield nxt


def _station_cells(base):
    return footprint(base) if isinstance(base, dict) and base.get("roleType") == "station" else footprint(
        {"roleType": "station", "pos": {"x": pos(base)[0], "y": pos(base)[1]}}
    )


def _layout_occupied(base, width, height, extra_blocked, rockets=()):
    occupied = set(_station_cells(base))
    occupied.update(pos(cell) for cell in extra_blocked)
    occupied.update(template_wall_cells(base, width, height))
    occupied.update(pos(cell) for cell in rockets)
    return occupied


def walkable_operator_pads(gun, occupied, width, height):
    """Empty 8-neighbour cells a controller can stand on to fire `gun`."""
    return [cell for cell in _eight_neighbors(gun, width, height) if cell not in occupied]


def _bfs_reachable(seeds, occupied, width, height):
    seen = set()
    queue = deque()
    for seed in seeds:
        cell = pos(seed)
        if cell in occupied or not _in_bounds(cell, width, height) or cell in seen:
            continue
        seen.add(cell)
        queue.append(cell)
    while queue:
        current = queue.popleft()
        for nxt in _eight_neighbors(current, width, height):
            if nxt not in occupied and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def _access_seeds(base, region, occupied, width, height):
    """Walkable cells on the unwalled side of the C, used as pathfinding starts."""
    sx, sy = pos(base)
    if region == BASE_REGION_TOP_LEFT:
        candidates = [(sx + dx, sy + dy) for dx in range(-4, 0) for dy in range(-5, 5)]
    else:
        candidates = [(sx + dx, sy + dy) for dx in range(2, 6) for dy in range(-5, 5)]
    return [cell for cell in candidates
            if _in_bounds(cell, width, height) and cell not in occupied]


def cluster_is_operable(base, cluster, width, height, extra_blocked=()):
    """True if each rocket has a reachable standing cell after walls go up."""
    cluster = [pos(cell) for cell in cluster]
    if len(cluster) != len(set(cluster)):
        return False
    occupied = _layout_occupied(base, width, height, extra_blocked, cluster)
    pads_by_gun = [walkable_operator_pads(gun, occupied, width, height) for gun in cluster]
    if any(not pads for pads in pads_by_gun):
        return False
    region = base_region(base, width, height)
    seeds = _access_seeds(base, region, occupied, width, height)
    if not seeds:
        seeds = [pad for pads in pads_by_gun for pad in pads]
    reachable = _bfs_reachable(seeds, occupied, width, height)
    return all(any(pad in reachable for pad in pads) for pads in pads_by_gun)


def _preferred_rocket_clusters(base, region):
    """Right-column (top-left) / left-column (bottom-right) lines of three.

    These sit on the enemy-facing blue edge but leave the top and bottom
    corridors along the station free, so a controller can walk in with
    8-direction movement after the C-shaped wall is up.
    """
    sx, sy = pos(base)
    if region == BASE_REGION_TOP_LEFT:
        return [
            ((sx + 2, sy + 1), (sx + 2, sy), (sx + 2, sy - 1)),
            ((sx + 2, sy), (sx + 2, sy - 1), (sx + 2, sy - 2)),
            ((sx + 2, sy + 1), (sx + 2, sy - 1), (sx + 2, sy - 2)),
        ]
    return [
        ((sx - 1, sy + 1), (sx - 1, sy), (sx - 1, sy - 1)),
        ((sx - 1, sy), (sx - 1, sy - 1), (sx - 1, sy - 2)),
        ((sx - 1, sy + 1), (sx - 1, sy - 1), (sx - 1, sy - 2)),
    ]


def _cluster_sort_key(cluster, base, region):
    cells = [pos(cell) for cell in cluster]
    sx, _ = pos(base)
    diameter = max(distance(a, b) for a in cells for b in cells)
    if region == BASE_REGION_TOP_LEFT:
        facing = 0 if all(x >= sx + 2 for x, _ in cells) else 1
        pull = -sum(x for x, _ in cells)
    else:
        facing = 0 if all(x <= sx - 1 for x, _ in cells) else 1
        pull = sum(x for x, _ in cells)
    return (0 if _cluster_connected(cells) else 1, facing, diameter, pull,
            sum(distance(cell, base) for cell in cells), tuple(sorted(cells)))


def generate_rocket_positions(base, width, height, blocked=()):
    """Pick up to three clustered rockets a character can walk in and fire."""
    if not base:
        LOG.info("rocket layout skipped: no station")
        return []
    blue, _ = building_rings(base, width, height)
    blocked = {pos(cell) for cell in blocked}
    legal = {cell for cell in blue if cell not in blocked and _in_bounds(cell, width, height)}
    if not legal:
        LOG.info("rocket layout skipped: no legal inner-ring cells")
        return []
    region = base_region(base, width, height)
    for cluster in _preferred_rocket_clusters(base, region):
        chosen = [cell for cell in cluster if cell in legal]
        if len(chosen) == ROCKET_TURRET_COUNT and cluster_is_operable(
                base, chosen, width, height, blocked):
            LOG.info("rocket layout preferred %s", chosen)
            return chosen
    operable = []
    for combo in combinations(sorted(legal), ROCKET_TURRET_COUNT):
        if not _cluster_connected(combo):
            continue
        if cluster_is_operable(base, combo, width, height, blocked):
            operable.append(combo)
    if operable:
        best = min(operable, key=lambda cluster: _cluster_sort_key(cluster, base, region))
        LOG.info("rocket layout fallback %s", best)
        return list(best)
    # Safe degrade: nearest connected legal cells, even if a pad is currently blocked.
    chosen = []
    remaining = set(legal)
    while len(chosen) < ROCKET_TURRET_COUNT and remaining:
        anchor = chosen[-1] if chosen else pos(base)
        nxt = min(remaining, key=lambda cell: (
            0 if chosen and min(distance(cell, item) for item in chosen) <= 1 else 1,
            distance(cell, anchor),
            distance(cell, base),
            cell,
        ))
        chosen.append(nxt)
        remaining.remove(nxt)
    if len(chosen) < ROCKET_TURRET_COUNT:
        LOG.info("rocket layout degraded: only %s legal cells", len(chosen))
    else:
        LOG.info("rocket layout degraded: no fully operable cluster, using %s", chosen)
    return chosen[:ROCKET_TURRET_COUNT]


def template_wall_cells(base, width, height):
    """Twelve unique yellow-ring cells forming a C facing the map centre."""
    if not base:
        return []
    sx, sy = pos(base)
    region = base_region(base, width, height)
    cells = set()
    if region == BASE_REGION_TOP_LEFT:
        for dy in range(-3, 3):
            cells.add((sx + 3, sy + dy))
        for dx in range(0, 4):
            cells.add((sx + dx, sy + 2))
            cells.add((sx + dx, sy - 3))
    else:
        for dy in range(-3, 3):
            cells.add((sx - 2, sy + dy))
        for dx in range(-2, 2):
            cells.add((sx + dx, sy + 2))
            cells.add((sx + dx, sy - 3))
    return [cell for cell in sorted(cells) if _in_bounds(cell, width, height)]


def _nearest_open_cell(anchor, legal, used):
    available = [cell for cell in legal if cell not in used]
    if not available:
        return None
    return min(available, key=lambda cell: (distance(cell, anchor), cell))


def generate_wall_positions(base, width, height, blocked=(), preferred=()):
    """Return twelve legal wall cells, substituting the nearest yellow cell."""
    if not base:
        LOG.info("wall layout skipped: no station")
        return []
    _, yellow = building_rings(base, width, height)
    blocked = {pos(cell) for cell in blocked}
    yellow = {cell for cell in yellow if _in_bounds(cell, width, height)}
    chosen = []
    used = set()
    for cell in preferred or ():
        point = pos(cell)
        if point in yellow and point not in used:
            chosen.append(point)
            used.add(point)
    for cell in template_wall_cells(base, width, height):
        if len(chosen) >= TARGET_WALL_COUNT:
            break
        if cell in used:
            continue
        if cell in yellow and cell not in blocked:
            chosen.append(cell)
            used.add(cell)
            continue
        substitute = _nearest_open_cell(cell, yellow, used | blocked)
        if substitute is None:
            LOG.info("wall layout: no substitute for %s", cell)
            continue
        LOG.info("wall layout: %s blocked, using %s", cell, substitute)
        chosen.append(substitute)
        used.add(substitute)
    if len(chosen) < TARGET_WALL_COUNT:
        for cell in sorted(yellow - used, key=lambda item: (distance(item, base), item)):
            if cell in blocked:
                continue
            chosen.append(cell)
            used.add(cell)
            if len(chosen) >= TARGET_WALL_COUNT:
                break
    if len(chosen) < TARGET_WALL_COUNT:
        LOG.info("wall layout degraded: only %s legal cells", len(chosen))
    return chosen[:TARGET_WALL_COUNT]


def ring_order(base, width, height, bias=(0, 0)):
    """Order the outer wall ring so that every prefix is one connected arc.

    Cells are swept by angle around the station centre, which for a convex
    ring equals perimeter order, then re-indexed by cyclic distance from the
    cell that best faces `bias`. The last entry is the single gate that is
    never walled, so a prefix of the result is a continuous wall with one exit.
    """
    if not base:
        return []
    _, yellow = building_rings(base, width, height)
    if not yellow:
        return []
    x, y = pos(base)
    centre = (x + 0.5, y - 0.5)
    cycle = sorted(yellow, key=lambda cell: (
        math.atan2(cell[1] - centre[1], cell[0] - centre[0]), cell))
    anchor = 0
    if tuple(bias) != (0, 0):
        anchor = max(range(len(cycle)), key=lambda index: (
            bias[0] * (cycle[index][0] - centre[0]) +
            bias[1] * (cycle[index][1] - centre[1]), -index))
    total = len(cycle)
    order = [cycle[anchor]]
    seen = {cycle[anchor]}
    for step in range(1, total):
        for cell in (cycle[(anchor + step) % total], cycle[(anchor - step) % total]):
            if cell not in seen:
                seen.add(cell)
                order.append(cell)
    return order


class AutoConstruction:
    """Propose worker moves/builds, reserving this turn's planned weapon types.

The caller reserves emitted destinations and deducts exactly 25 gold for each
build. Claims prevent two workers planning the same missing weapon this turn.
Every new snapshot resets claims; only living observed guns count as completed
construction. A valid-action flag alone never invents a completed building.
"""

    def __init__(self, config):
        self.config = config or {}
        self._round = None
        self._claims = {}
        self._failed = {}

    def _begin_round(self, round_no):
        if round_no != self._round:
            self._claims = {}
            if self._round is not None and round_no < self._round:
                self._failed = {}
            self._round = round_no
            self._failed = {cell: expiry for cell, expiry in self._failed.items()
                            if expiry > round_no}

    def observe(self, data, previous_commands, previous_round):
        """Use only contiguous, per-role failure feedback for short avoidance."""
        round_no = int(data.get('roundNo', 1))
        self._begin_round(round_no)
        if previous_round is None or round_no != previous_round + 1:
            return
        results = {str(key): value for key, value in
                   (data.get('lastRoundRoleActionResults') or {}).items()}
        observed = {(pos(role), role.get('roleType')) for role in
                    (data.get('teamOur') or {}).get('roles', [])
                    if role.get('roleType') in GUNS and int(role.get('health', 1)) > 0}
        for ident, command in (previous_commands or {}).items():
            if command.get('action') != 'build' or command.get('name') not in GUNS:
                continue
            targets = command.get('targetPos') or []
            if len(targets) != 1:
                continue
            target = pos(targets[0])
            if (target, command['name']) in observed:
                self._failed.pop(target, None)
            elif results.get(str(ident)) is False:
                self._failed[target] = round_no + 4

    def propose(self, world, role, reserved, available_gold, targets=None):
        """Return one official move/build command, or None if no legal job."""
        self._begin_round(int(world.data.get('roundNo', 1)))
        cfg = self.config.get('construction') or {}
        ident = str(role['id'])
        if (cfg.get('mode', 'auto') == 'off' or world.night or world.base is None
                or role.get('roleType') != 'worker' or int(role.get('health', 1)) <= 0
                or available_gold < WEAPON_GOLD or ident in self._claims
                or len(world.guns) + len(self._claims) >= ROCKET_TURRET_COUNT):
            return None
        plan = cfg.get('weapon_plan', DEFAULT_WEAPON_PLAN)
        if not isinstance(plan, (list, tuple)):
            plan = DEFAULT_WEAPON_PLAN
        plan = [kind for kind in plan if kind in GUNS][:ROCKET_TURRET_COUNT]
        counts = Counter(gun['roleType'] for gun in world.guns)
        counts.update(claim[0] for claim in self._claims.values())
        kind = None
        for requested in plan:
            if counts[requested]:
                counts[requested] -= 1
            else:
                kind = requested
                break
        if kind is None:
            return None
        blue, _ = building_rings(world.base, world.width, world.height)
        if targets:
            focus = {pos(cell) for cell in targets} & blue
            if not focus:
                LOG.info("planned rocket cells unavailable; degrading to inner ring")
                focus = blue
        else:
            focus = blue
        reserved = {pos(cell) for cell in reserved}
        reserved.update(claim[1] for claim in self._claims.values())
        candidates = []
        for target in sorted(focus - world.blocked - reserved - set(self._failed)):
            # The future building is also blocked while routing its builder.
            future_reserved = reserved | {target}
            cells = world.interaction_cells([target])
            future_blocked = (world.blocked - {pos(role)}) | future_reserved
            # Preserve an exit from the operating cell after the gun appears.
            cells = {cell for cell in cells if cell not in future_blocked
                     and any(n not in future_blocked for n in world.neighbors(cell))}
            route = world.path(role, cells, future_reserved)
            if route is None:
                continue
            candidates.append((len(route) - 1, target, route))
        if not candidates:
            return None
        _, target, route = min(candidates, key=lambda item: (item[0], item[1]))
        self._claims[ident] = (kind, target)
        if len(route) > 1:
            destination = route[1]
            return {'action': 'move', 'targetPos': [{'x': destination[0], 'y': destination[1]}]}
        return {'action': 'build', 'name': kind,
                'targetPos': [{'x': target[0], 'y': target[1]}]}

    def propose_replacement(self, world, role, reserved, available_gold, desired_plan):
        """Replace one surplus gun when the observed wave justifies rebalancing."""
        self._begin_round(int(world.data.get('roundNo', 1)))
        ident = str(role['id'])
        if (world.night or role.get('roleType') != 'worker' or available_gold < WEAPON_GOLD
                or ident in self._claims or len(world.guns) != 3 or self._claims):
            return None
        desired = [kind for kind in desired_plan if kind in GUNS][:3]
        if len(desired) != 3:
            return None
        have, want = Counter(gun['roleType'] for gun in world.guns), Counter(desired)
        missing = next((kind for kind in desired if have[kind] < want[kind]), None)
        surplus = [gun for gun in world.guns if have[gun['roleType']] > want[gun['roleType']]]
        if missing is None or not surplus:
            return None
        reserved = {pos(cell) for cell in reserved}
        routes = []
        for gun in surplus:
            target = pos(gun)
            if target in reserved or target in self._failed:
                continue
            route = world.path(role, world.interaction_cells([target]), reserved)
            if route is not None:
                routes.append((len(route) - 1, str(gun['id']), target, route))
        if not routes:
            return None
        _, _, target, route = min(routes)
        self._claims[ident] = (missing, target)
        if len(route) > 1:
            destination = route[1]
            return {'action': 'move', 'targetPos': [{'x': destination[0], 'y': destination[1]}]}
        return {'action': 'build', 'name': missing,
                'targetPos': [{'x': target[0], 'y': target[1]}]}
