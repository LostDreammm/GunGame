"""Navigation and official weapon commands; no environment simulation.

Coordinates use the taskbook's lower-left origin. A station's position is its
top-left cell. Ballistic cell membership is not specified textually by v1.0:
this planner conservatively intersects the center-to-center segment with robot
cell squares, including corner contacts. The judger's rasterization may differ.
Only robots intercept bullets in the supplied textual weapon rules. Damage is
settled at turn end, so projected kills never remove ballistic obstructions.
"""

import logging
import math
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
TARGET_WALL_COUNT = 14
WALL_STONE_COST = 1
BASE_REGION_TOP_LEFT = "top_left"
BASE_REGION_BOTTOM_RIGHT = "bottom_right"
FRONTS = ("W", "E", "N", "S")
DAY_LENGTH = 130
DAYTIME_LENGTH = 70
GATE_HYSTERESIS = 0.34
ROCKET_MIN_VALUE = 20
ATTACK_OBS_RANGE = 18
ATTACK_NEAR_RANGE = 9
ATTACK_NEAR_WEIGHT = 3
ROCKET_SPREAD = 2
NEAR_MINE_RADIUS = 16
CORRIDOR_HALF_WIDTH = 5
CORRIDOR_SAFE_BASE = 8
APPROACH_RANGE = 12


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


def base_center(base):
    x, y = pos(base)
    return (x + 0.5, y - 0.5)


def _unit(dx, dy):
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        return (0.0, 0.0)
    return (dx / length, dy / length)


def facing_score(cell, center, direction):
    vx, vy = cell[0] - center[0], cell[1] - center[1]
    length = math.hypot(vx, vy) or 1.0
    dnorm = math.hypot(direction[0], direction[1]) or 1.0
    return (vx * direction[0] + vy * direction[1]) / (length * dnorm)


def default_attack_direction(base, width, height, enemy_base=None):
    center = base_center(base)
    if enemy_base:
        other = base_center(enemy_base)
        vector = _unit(other[0] - center[0], other[1] - center[1])
        if vector != (0.0, 0.0):
            return vector
    return _unit((width - 1) / 2.0 - center[0], (height - 1) / 2.0 - center[1])


def accumulate_attack_votes(base, robots, team, votes):
    """Add night observations: robots charging our base within 18 cells."""
    if not base:
        return votes
    center = base_center(base)
    cells = footprint(base)
    for robot in robots:
        target = robot.get("targetTeam")
        if target not in (None, "", team):
            continue
        near = min(distance(robot, cell) for cell in cells)
        if near > ATTACK_OBS_RANGE:
            continue
        weight = ATTACK_NEAR_WEIGHT if near <= ATTACK_NEAR_RANGE else 1
        dx = pos(robot)[0] - center[0]
        dy = pos(robot)[1] - center[1]
        qx = 0 if abs(dx) < 0.5 else (1 if dx > 0 else -1)
        qy = 0 if abs(dy) < 0.5 else (1 if dy > 0 else -1)
        key = (qx, qy)
        votes[key] = votes.get(key, 0) + weight
    return votes


def direction_from_votes(votes, base, width, height, enemy_base=None):
    if votes:
        (qx, qy), _ = max(votes.items(), key=lambda item: (item[1], item[0]))
        vector = _unit(qx, qy)
        if vector != (0.0, 0.0):
            return vector
    return default_attack_direction(base, width, height, enemy_base)


def threat_robots(world, ours_only=False):
    """Robots that are alive and not confirmed to be attacking the other team."""
    team = (world.data.get("teamOur") or {}).get("type")
    result = []
    for robot in world.robots:
        if int(robot.get("health", 0)) <= 0:
            continue
        target = robot.get("targetTeam")
        if target not in (None, "", team):
            continue
        if ours_only and target != team:
            continue
        result.append(robot)
    return result


def our_wave(world, robots=None):
    """Prefer robots explicitly charging us; fall back to unlabelled threats."""
    team = (world.data.get("teamOur") or {}).get("type")
    pool = list(robots if robots is not None else world.robots)
    confirmed = [robot for robot in pool
                 if int(robot.get("health", 0)) > 0 and robot.get("targetTeam") == team]
    if confirmed:
        return confirmed
    return threat_robots(world)


def segment_distance(point, start, end):
    """Chebyshev distance from a cell to the spawn-to-base corridor segment."""
    px, py = pos(point)
    ax, ay = pos(start)
    bx, by = pos(end)
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return max(abs(px - ax), abs(py - ay))
    length2 = float(dx * dx + dy * dy)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length2))
    qx, qy = ax + t * dx, ay + t * dy
    return max(abs(px - qx), abs(py - qy))


def in_march_corridor(cell, spawn, base, half_width=CORRIDOR_HALF_WIDTH,
                      safe_base=CORRIDOR_SAFE_BASE):
    """True if a mine sits on the night-wave path and is far enough from base."""
    if spawn is None or base is None:
        return False
    cells = footprint(base) if isinstance(base, dict) else {pos(base)}
    near_base = min(distance(cell, item) for item in cells)
    if near_base <= safe_base:
        return False
    anchor = min(cells, key=lambda item: distance(item, spawn))
    return segment_distance(cell, spawn, anchor) <= half_width


def on_our_side(cell, base, enemy_base=None, width=0, height=0):
    """Reject mines that sit closer to the enemy station than to ours."""
    if not base:
        return True
    ours = min(distance(cell, item) for item in footprint(base))
    if enemy_base:
        theirs = min(distance(cell, item) for item in footprint(enemy_base))
        return ours <= theirs
    if width > 0 and height > 0:
        mid_x, mid_y = (width - 1) / 2.0, (height - 1) / 2.0
        cx, cy = base_center(base)
        px, py = pos(cell)
        if cx <= mid_x and cy >= mid_y:
            return px <= mid_x + 2 or py >= mid_y - 2
        if cx >= mid_x and cy <= mid_y:
            return px >= mid_x - 2 or py <= mid_y + 2
    return True


def pair_controllers(people, guns):
    """Greedy person-to-gun matching by Chebyshev distance."""
    assigned, used_people, used_guns = {}, set(), set()
    pairs = []
    for person in people:
        for gun in guns:
            pairs.append((distance(person, gun), str(person["id"]), str(gun["id"]), person, gun))
    for _, pid, gid, person, gun in sorted(pairs):
        if pid in used_people or gid in used_guns:
            continue
        assigned[pid] = gun
        used_people.add(pid)
        used_guns.add(gid)
    return assigned


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


def cluster_consumable_target(world):
    """Large/BOSS centred cluster: at least two undizzy robots within one cell."""
    threats = [robot for robot in threat_robots(world)
               if robot.get("abnormalState") != "dizzy"]
    best = None
    for hub in threats:
        if hub.get("roleType") not in ("largeRobot", "bossRobot"):
            continue
        nearby = [robot for robot in threats if distance(robot, hub) <= 1]
        if len(nearby) < 2:
            continue
        score = (len(nearby), ROBOT_POINTS.get(hub.get("roleType"), 0), -pos(hub)[0], -pos(hub)[1])
        if best is None or score > best[0]:
            best = (score, pos(hub))
    return None if best is None else best[1]


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


def _rocket_landings(robots):
    cells = {pos(robot) for robot in robots}
    for left, right in combinations(robots, 2):
        if distance(left, right) <= 2:
            ax, ay = pos(left)
            bx, by = pos(right)
            cells.add(((ax + bx) // 2, (ay + by) // 2))
    return cells


def _rocket_value(target, robots, projected, world=None):
    """Score a rocket landing: center 20 + splash 10 + lethal 15 + approach + our-side."""
    value = 0
    team = ((world.data.get("teamOur") or {}).get("type") if world else None)
    base_cells = footprint(world.base) if world and world.base else set()
    for robot in robots:
        ident = str(robot["id"])
        remaining = projected.get(ident, int(robot.get("health", 0)))
        if remaining <= 0:
            continue
        gap = distance(robot, target)
        if gap == 0:
            damage = 20
            value += 20
        elif gap == 1:
            damage = 10
            value += 10
        else:
            continue
        if damage >= remaining:
            value += 15
        if base_cells:
            near = min(distance(robot, cell) for cell in base_cells)
            if near <= APPROACH_RANGE:
                value += max(1, APPROACH_RANGE - near)
        if team and robot.get("targetTeam") == team:
            value += 5
    return value


def _base_threatened(world, robots, radius=3):
    if not world.base:
        return False
    cells = footprint(world.base)
    return any(min(distance(robot, cell) for cell in cells) <= radius for robot in robots)


def plan_attacks(world, assignments):
    """Produce official weapon-keyed commands plus occupied controller IDs."""
    if not world.night:
        return {}, set()
    people = {str(p["id"]): p for p in world.people}
    guns = {str(g["id"]): g for g in world.guns}
    labelled = our_wave(world)
    threats = labelled if labelled else threat_robots(world)
    robots_by_id = {str(r["id"]): r for r in threats}
    projected = {ident: int(r["health"]) for ident, r in robots_by_id.items()}
    weights = _robot_weights(world)
    position_index = {}
    for robot in threats:
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
        in_range = [robot for robot in threats
                    if 0 < distance(gun, robot) <= (attack_range or 10 ** 9)]
        if kind == "rocket":
            landings = []
            for cell in _rocket_landings(in_range):
                if not world.in_bounds(cell) or cell == origin:
                    continue
                if attack_range and distance(origin, cell) > attack_range:
                    continue
                landings.append(cell)
            selected, total = [], 0
            local = dict(projected)
            for _ in range(count):
                chosen, best = None, -1
                for target in landings:
                    if any(distance(target, previous) < ROCKET_SPREAD for previous in selected):
                        continue
                    gain = _rocket_value(target, in_range, local, world)
                    if gain > best:
                        chosen, best = target, gain
                if chosen is None or best < ROCKET_MIN_VALUE:
                    break
                selected.append(chosen)
                total += best
                for robot in in_range:
                    gap = distance(robot, chosen)
                    if gap == 0:
                        hit = 20
                    elif gap == 1:
                        hit = 10
                    else:
                        continue
                    ident = str(robot["id"])
                    local[ident] = max(0, local.get(ident, 0) - hit)
            if selected and total >= ROCKET_MIN_VALUE:
                commands[gid] = {"action": "attack", "controllerId": controller_id,
                                 "targetPos": [{"x": x, "y": y} for x, y in selected]}
                used.add(controller_id)
                projected = local
            continue
        candidates = sorted({pos(robot) for robot in in_range})
        if not candidates:
            continue
        damage_cache = {target: _damage(gun, target, threats, position_index) for target in candidates}
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


def layout_front(base, width, height, direction=None):
    """Opening side, opposite the dominant incoming direction."""
    if direction is None or direction == (0.0, 0.0):
        direction = default_attack_direction(base, width, height)
    dx, dy = direction
    if abs(dx) >= abs(dy):
        return "W" if dx >= 0 else "E"
    return "S" if dy >= 0 else "N"


def _front_transform(front, dx, dy):
    if front == "E":
        return 1 - dx, dy
    if front == "N":
        return dy, 1 - dx
    if front == "S":
        return 1 - dy, dx
    return dx, dy


def _front_from_region(region):
    if region in FRONTS:
        return region
    return "W" if region == BASE_REGION_TOP_LEFT else "E"


def _relative_to_top_left(front, cell):
    """Transform canonical coordinates anchored at station bottom-left."""
    dx, dy = _front_transform(front, cell[0], cell[1])
    return dx, dy - 1


def specified_rocket_offsets(region):
    """Three inner-ring cells sharing one stand, transformed to any FRONT."""
    front = _front_from_region(region)
    canonical = ((-1, 1), (-1, -1), (0, -1))
    return tuple(_relative_to_top_left(front, cell) for cell in canonical)


def specified_control_point(base, region=None):
    """The one stand that neighbours all three specified rockets."""
    if not base:
        return None
    sx, sy = pos(base)
    front = _front_from_region(region or BASE_REGION_TOP_LEFT)
    dx, dy = _relative_to_top_left(front, (-1, 0))
    return sx + dx, sy + dy


def _cp_covers(cell, rockets):
    return all(distance(cell, rocket) == 1 for rocket in rockets)


def choose_control_point(rockets, base, width, height, blocked=(), preferred=None):
    """Pick a walkable pad adjacent to every rocket; migrate if the template fails."""
    rockets = [pos(cell) for cell in rockets]
    if len(rockets) < 2:
        return None
    occupied = set(_station_cells(base))
    occupied.update(pos(cell) for cell in rockets)
    occupied.update(pos(cell) for cell in blocked)
    if preferred:
        cand = pos(preferred)
        if (_in_bounds(cand, width, height) and cand not in occupied
                and _cp_covers(cand, rockets)):
            return cand
    region = base_region(base, width, height)
    specified = specified_control_point(base, region)
    if (specified and _in_bounds(specified, width, height) and specified not in occupied
            and _cp_covers(specified, rockets)):
        return specified
    pads = []
    for rocket in rockets:
        pads.extend(_eight_neighbors(rocket, width, height))
    best = None
    for cell in sorted(set(pads)):
        if cell in occupied or not _cp_covers(cell, rockets):
            continue
        cover = sum(1 for rocket in rockets if distance(cell, rocket) == 1)
        key = (-cover, distance(cell, base_center(base)), cell)
        if best is None or key < best[0]:
            best = (key, cell)
    return None if best is None else best[1]


def choose_repair_post(base, walls, rockets, control_point, width, height, blocked=()):
    """Inner-ring stand next to the most planned walls, never a gun or CP."""
    blue, _ = building_rings(base, width, height)
    forbidden = set(_station_cells(base))
    forbidden.update(pos(cell) for cell in rockets)
    if control_point:
        forbidden.add(pos(control_point))
    forbidden.update(pos(cell) for cell in blocked)
    wall_set = {pos(cell) for cell in walls}
    best = None
    for cell in sorted(blue):
        if cell in forbidden or not _in_bounds(cell, width, height):
            continue
        adjacent = sum(1 for wall in wall_set if distance(cell, wall) == 1)
        key = (-adjacent, distance(cell, control_point or base_center(base)), cell)
        if best is None or key < best[0]:
            best = (key, cell)
    if best:
        return best[1]
    if control_point and _in_bounds(pos(control_point), width, height):
        return pos(control_point)
    return None


def front_wall_cells(walls, control_point):
    """Enemy-facing walls: farthest from the opening / CP."""
    if not walls or not control_point:
        return set()
    ranked = sorted(walls, key=lambda cell: (-distance(cell, control_point), cell))
    if not ranked:
        return set()
    cutoff = max(distance(ranked[0], control_point) - 1, 1)
    return {cell for cell in ranked if distance(cell, control_point) >= cutoff}


def _step_to_blue(sx, sy, dx, dy, legal, used):
    """Honor the requested offset; if it lands on the 2x2, keep stepping out."""
    cell = (sx + dx, sy + dy)
    if cell in legal and cell not in used:
        return cell
    step_x = 0 if dx == 0 else (1 if dx > 0 else -1)
    step_y = 0 if dy == 0 else (1 if dy > 0 else -1)
    x, y = cell
    for _ in range(4):
        x += step_x
        y += step_y
        cand = (x, y)
        if cand in legal and cand not in used:
            return cand
    remaining = [item for item in legal if item not in used]
    if not remaining:
        return None
    return min(remaining, key=lambda item: (distance(item, cell), item))


def _specified_rocket_cells(base, legal, region):
    sx, sy = pos(base)
    chosen = []
    for dx, dy in specified_rocket_offsets(region):
        cell = _step_to_blue(sx, sy, dx, dy, legal, chosen)
        if cell is None:
            break
        chosen.append(cell)
    return chosen


def _preferred_rocket_clusters(base, region):
    """Single specified trio used when a listed cell is blocked."""
    sx, sy = pos(base)
    return [tuple((sx + dx, sy + dy) for dx, dy in specified_rocket_offsets(region))]


def _cluster_sort_key(cluster, base, region):
    cells = [pos(cell) for cell in cluster]
    sx, sy = pos(base)
    if region == BASE_REGION_TOP_LEFT:
        facing = 0 if all(x <= sx for x, _ in cells) else 1
    else:
        facing = 0 if all(x >= sx + 1 for x, _ in cells) else 1
    spread = max(distance(a, b) for a in cells for b in cells) if cells else 0
    return (facing, spread, tuple(sorted(cells)))


def assemble_base_plot(base, width, height, blocked=(), direction=None):
    """Pick a stand first, then three neighbouring guns; migrate the stand if needed."""
    if not base:
        return [], None
    blue, _ = building_rings(base, width, height)
    blocked = {pos(cell) for cell in blocked}
    pad = {cell for cell in blue if cell not in blocked and _in_bounds(cell, width, height)}
    if not pad:
        return [], None
    facing = layout_front(base, width, height, direction)
    ideal_cp = specified_control_point(base, facing)
    ordered = []
    if ideal_cp in pad:
        ordered.append(ideal_cp)
    ordered.extend(sorted((cell for cell in pad if cell != ideal_cp),
                          key=lambda cell: (distance(cell, ideal_cp or base_center(base)), cell)))
    named = []
    sx, sy = pos(base)
    for dx, dy in specified_rocket_offsets(facing):
        cell = (sx + dx, sy + dy)
        if cell in pad:
            named.append(cell)

    def guns_around(stand):
        picked, seen = [], set()
        if stand == ideal_cp:
            for cell in named:
                if cell != stand and cell not in seen:
                    seen.add(cell)
                    picked.append(cell)
        for cell in sorted(pad, key=lambda item: (distance(item, stand), item)):
            if cell == stand or cell in seen or distance(cell, stand) != 1:
                continue
            seen.add(cell)
            picked.append(cell)
            if len(picked) == ROCKET_TURRET_COUNT:
                break
        return picked[:ROCKET_TURRET_COUNT]

    best_stand, best_guns = None, []
    for stand in ordered:
        guns = guns_around(stand)
        if len(guns) == ROCKET_TURRET_COUNT:
            LOG.info("base plot stand=%s guns=%s front=%s", stand, guns, facing)
            return guns, stand
        if len(guns) > len(best_guns):
            best_stand, best_guns = stand, guns
    LOG.info("base plot incomplete stand=%s guns=%s", best_stand, best_guns)
    return best_guns, best_stand


def generate_rocket_positions(base, width, height, blocked=(), direction=None):
    """Three inner-ring rockets that share one walkable control stand."""
    guns, _ = assemble_base_plot(base, width, height, blocked, direction)
    return list(guns)


def template_wall_cells(base, width, height, front=None):
    """Fourteen yellow-ring cells: three sides, opening on the gun/CP side."""
    if not base:
        return []
    sx, sy = pos(base)
    front = front or layout_front(base, width, height)
    canonical = []
    base_cells = ((0, 0), (1, 0), (0, 1), (1, 1))
    for dx in range(-2, 4):
        for dy in range(-2, 4):
            ring_distance = min(max(abs(dx - bx), abs(dy - by))
                                for bx, by in base_cells)
            if ring_distance == 2 and dx != -2:
                canonical.append((dx, dy))
    cells = set()
    for cell in canonical:
        ox, oy = _relative_to_top_left(front, cell)
        cells.add((sx + ox, sy + oy))
    return [cell for cell in sorted(cells) if _in_bounds(cell, width, height)]


def _nearest_open_cell(anchor, legal, used):
    available = [cell for cell in legal if cell not in used]
    if not available:
        return None
    return min(available, key=lambda cell: (distance(cell, anchor), cell))


def choose_wall_gate(yellow, center, direction, old_gate=None):
    """Leave one exit opposite the attack vector; migrate only if clearly better."""
    if not yellow:
        return None
    new_gate = max(yellow, key=lambda cell: (-facing_score(cell, center, direction), cell))
    if old_gate and old_gate in yellow:
        improved = facing_score(old_gate, center, direction) - facing_score(new_gate, center, direction)
        if improved <= GATE_HYSTERESIS:
            return old_gate
    return new_gate


def opening_side_cells(base, width, height, front=None):
    """Yellow cells on the gun/CP side that stay open as the FRONT gate."""
    if not base:
        return []
    sx, sy = pos(base)
    front = front or layout_front(base, width, height)
    cells = []
    for dy in range(-2, 4):
        ox, oy = _relative_to_top_left(front, (-2, dy))
        cells.append((sx + ox, sy + oy))
    return [cell for cell in cells if _in_bounds(cell, width, height)]


def wall_build_order(base, width, height, direction=None, old_gate=None, blocked=()):
    """14-cell C-ring: enemy-facing front first, then the two flanks."""
    if not base:
        return [], None
    _, yellow = building_rings(base, width, height)
    yellow = {cell for cell in yellow if _in_bounds(cell, width, height)}
    if not yellow:
        return [], None
    front = layout_front(base, width, height, direction)
    opening = set(opening_side_cells(base, width, height, front))
    planned = [cell for cell in template_wall_cells(base, width, height, front) if cell in yellow]
    control = specified_control_point(base, front)
    planned.sort(key=lambda cell: (
        -distance(cell, control or base_center(base)),
        cell,
    ))
    blocked = {pos(cell) for cell in blocked}
    legal = yellow - opening
    chosen, used = [], set()
    for cell in planned:
        if cell not in blocked:
            chosen.append(cell)
            used.add(cell)
            continue
        substitute = _nearest_open_cell(cell, legal, used | blocked)
        if substitute is None:
            LOG.info("wall layout: no substitute for %s", cell)
            continue
        LOG.info("wall layout: %s blocked, using %s", cell, substitute)
        chosen.append(substitute)
        used.add(substitute)
    if not chosen:
        LOG.info("wall layout skipped: no legal yellow cells")
    gate = old_gate if old_gate in opening else (sorted(opening)[len(opening) // 2] if opening else None)
    return chosen, gate


def generate_wall_positions(base, width, height, blocked=(), preferred=(),
                            direction=None, old_gate=None):
    """Ring of walls two cells out, leaving one gate opposite the attack."""
    if not base:
        LOG.info("wall layout skipped: no station")
        return []
    if preferred:
        kept = [pos(cell) for cell in preferred]
        if len(kept) >= 3:
            return kept
    walls, _ = wall_build_order(base, width, height, direction, old_gate, blocked)
    return walls


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
