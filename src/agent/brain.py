"""Official v1.0 contestant policy: scheme B, three-role balanced play."""
import copy
import logging
from collections import Counter

from .grid import (AutoConstruction, World, accumulate_attack_votes,
                   base_center, building_rings, choose_control_point,
                   choose_repair_post, default_attack_direction, direction_from_votes,
                   distance, facing_score, footprint, front_wall_cells,
                   assemble_base_plot, generate_rocket_positions, in_march_corridor,
                   on_our_side, our_wave, plan_attacks, pos, ROCKET_TURRET_COUNT,
                   TARGET_WALL_COUNT, WALL_STONE_COST, wall_build_order)
from .market import MarketPlanner
from .protocol import Reasoning

LOG = logging.getLogger(__name__)
ORES = ('stone', 'iron', 'copper')
MONEY_ORES = ('copper', 'iron')
GUNS = ('gatling', 'railgun', 'rocket')
SUMMON_ORDERS = ('SmallRobotSummonOrder', 'MiddleRobotSummonOrder',
                 'LargeRobotSummonOrder', 'BossRobotSummonOrder')
WALL_REPAIR_KIT = 'WallFixer'
WALL_REPAIR_KIT_TARGET_STOCK = 2
MEDICINE_TARGET_STOCK = 2
BASE_LOW_HEALTH_RATIO = 0.5
STONE_KEEP = 5
STONE_BATCH_DAY1 = 14
STONE_BATCH_LATER = 6
STONE_DEMAND_CAP = 14
BAG_SELL_RATIO = 0.60
NEAR_MINE_RADIUS = 16
WORKER_DANGER = 3
WORKER_RESUME = 5
SPAWN_AVOID = 4
DUSK_LEAD = 12
PIONEER_RETREAT = 1
PIONEER_MEDICINE_HP = 120
PIONEER_HOME_BUFFER = 2
EMERGENCY_GOLD = 30
BASE_UPGRADE_GOLD = 250
WALL_FIX_RATIO = 0.50
WALL_CRITICAL_RATIO = 0.30
WALL_HURT_RATIO = 0.60
FRONT_WALL_L2_CAP = 6
SELL_NEAR_DIST = 3
SELL_NEAR_VALUE = 8
SELL_TRIP_VALUE = 25
SELL_TRIP_DIST = 20
ROBOT_BASE_DANGER = 4
NIGHT_CLEAR_ROUNDS = 30
MINE_YIELD_CAP = 10
STALE_COLLECT_HITS = 3
STALE_BLACKLIST = 20
COLLECT_FAIL_ROUNDS = 10
COLLECT_REPEAT_ROUNDS = 40
BUILD_FAIL_ROUNDS = 40
FAIL_HITS = 2
FAIL_ROUNDS = 10
REPAIRER_HOME_GIVEUP = 40
MINER_HELP_STONES = 20
WEAPON_GOLD = 25
UPGRADE_WEAPON = 'weapon'
UPGRADE_WALL = 'wall'
UPGRADE_STATION = 'station'


def point(p):
    return {'x': int(p[0]), 'y': int(p[1])}


def command(action, target=None, **fields):
    result = {'action': action}
    if target is not None:
        result['targetPos'] = [point(target)]
    result.update(fields)
    return result


def empty_response():
    return {'roleCommandMap': {}, 'prompt': '', 'executeCmd': ''}


def wall_stone_deficit(effective_walls, available_stone,
                       target=TARGET_WALL_COUNT, cost=WALL_STONE_COST):
    missing = max(0, int(target) - int(effective_walls or 0))
    needed = missing * max(0, int(cost))
    return min(STONE_DEMAND_CAP, max(0, needed - max(0, int(available_stone or 0))))


def upgrade_group_order(base_health_ratio=1.0):
    return (UPGRADE_WEAPON, UPGRADE_WALL, UPGRADE_STATION)


def fixer_target_stock(day, l3_walls=0):
    if int(day or 0) < 3:
        return 0
    if int(day or 0) >= 4 or int(l3_walls or 0) > 0:
        return min(3, max(2, int(l3_walls or 0)))
    return WALL_REPAIR_KIT_TARGET_STOCK


def fixer_restock_count(stock, target=WALL_REPAIR_KIT_TARGET_STOCK):
    return max(0, int(target) - max(0, int(stock or 0)))


def can_complete_same_day(remaining_rounds, flow_cost):
    if remaining_rounds is None or flow_cost is None:
        return False
    return int(remaining_rounds) >= int(flow_cost)


def preferred_ores(stone_deficit, prices, blocked=(), money=False):
    blocked = set(blocked or ())
    if int(stone_deficit or 0) > 0 and 'stone' not in blocked and not money:
        return ('stone',)
    valued = [ore for ore in MONEY_ORES + ('stone',) if ore not in blocked]
    if money:
        valued = [ore for ore in valued if ore != 'stone'] or valued
    return tuple(sorted(valued, key=lambda ore: (-int((prices or {}).get(ore, 0)), ore)))


def command_fingerprint(cmd):
    targets = tuple((item.get('x'), item.get('y')) for item in (cmd.get('targetPos') or [])
                    if isinstance(item, dict))
    return (cmd.get('action'), cmd.get('name'), targets)


class WallRegistry:
    def __init__(self):
        self.cells = {}

    def observe(self, planned, walls_by_pos, front, round_no):
        for cell in planned:
            wall = walls_by_pos.get(cell)
            prev = self.cells.get(cell, {})
            if wall:
                rebuilt = prev.get('rebuilt_round')
                if prev and not prev.get('exists'):
                    rebuilt = round_no
                self.cells[cell] = {
                    'exists': True,
                    'level': int(wall.get('level', 1)),
                    'health': int(wall.get('health', 0)),
                    'front': cell in front,
                    'breached_round': prev.get('breached_round'),
                    'rebuilt_round': rebuilt,
                }
            else:
                breached = prev.get('breached_round')
                if prev.get('exists'):
                    breached = round_no
                self.cells[cell] = {
                    'exists': False,
                    'level': 0,
                    'health': 0,
                    'front': cell in front,
                    'breached_round': breached,
                    'rebuilt_round': prev.get('rebuilt_round'),
                }

    def reset(self):
        self.cells = {}


class Strategy:
    def __init__(self, config=None):
        self.config = copy.deepcopy(config or {})
        self._identity = None
        self._round = None
        self._cached = None
        self.reasoning = Reasoning(self.config)
        self.auto_construction = AutoConstruction(self.config)
        self.failed_cells = {}
        self.treasure_attempted = set()
        self._previous = {}
        self._health_caps = {}
        self._summon_day = None
        self._summons_used = 0
        self._wall_slots = []
        self._wall_gate = None
        self._seen_walls = set()
        self._rocket_cells = []
        self._control_point = None
        self._repair_post = None
        self._spawn_point = None
        self._mine_yield = Counter()
        self._attack_votes = {}
        self.shop_runs = {}
        self.mine_claims = {}
        self._fail_hits = {}
        self._blacklist = {}
        self._collect_fail_hits = {}
        self._stale_collect = {}
        self._last_bags = {}
        self._banned_tasks = set()
        self._pending_accept = None
        self._gun_cursor = 0
        self._repairer_return = False
        self._repairer_return_round = None
        self._evading = {}
        self._wall_registry = WallRegistry()
        self._upgrade_dispatched = False
        self.market = MarketPlanner(self.config)

    def _reset_match(self, identity):
        knowledge = self.reasoning.export_knowledge() if self.reasoning else {}
        self.reasoning = Reasoning(self.config)
        self.reasoning.import_knowledge(knowledge)
        self.auto_construction = AutoConstruction(self.config)
        self.failed_cells = {}
        self.treasure_attempted = set()
        self._previous = {}
        self._health_caps = {}
        self._summon_day = None
        self._summons_used = 0
        self._wall_slots = []
        self._wall_gate = None
        self._seen_walls = set()
        self._rocket_cells = []
        self._control_point = None
        self._repair_post = None
        self._spawn_point = None
        self._mine_yield = Counter()
        self._attack_votes = {}
        self.shop_runs = {}
        self.mine_claims = {}
        self._fail_hits = {}
        self._blacklist = {}
        self._collect_fail_hits = {}
        self._stale_collect = {}
        self._last_bags = {}
        self._banned_tasks = set()
        self._pending_accept = None
        self._gun_cursor = 0
        self._repairer_return = False
        self._repairer_return_round = None
        self._evading = {}
        self._wall_registry = WallRegistry()
        self._upgrade_dispatched = False
        self.market = MarketPlanner(self.config)
        self._cached = None
        self._round = None
        self._identity = identity

    def callback(self, data):
        if not isinstance(data, dict) or not isinstance(data.get('teamOur'), dict):
            return empty_response()
        try:
            return self._callback(data)
        except Exception:
            LOG.exception("decide failed")
            return empty_response()

    def _callback(self, data):
        team = data['teamOur']
        identity = (team.get('teamId'), team.get('type'))
        round_no = int(data.get('roundNo', 1))
        if identity != self._identity or (self._round is not None and round_no < self._round):
            self._reset_match(identity)
        if round_no == self._round and self._cached is not None:
            return copy.deepcopy(self._cached)
        self.auto_construction.observe(data, self._previous, self._round)
        if self._round is not None and round_no == self._round + 1:
            roles = {str(role.get('id')): role for role in (team.get('roles') or [])
                     if isinstance(role, dict)}
            for rid, result in (data.get('lastRoundRoleActionResults') or {}).items():
                old = self._previous.get(str(rid), {})
                fingerprint = (str(rid), command_fingerprint(old)) if old else None
                target = (old.get('targetPos') or [{}])[0] if old else {}
                cell = (target['x'], target['y']) if isinstance(target, dict) and 'x' in target and 'y' in target else None
                if result is False and old.get('action') == 'build' and cell:
                    self.failed_cells[cell] = round_no + BUILD_FAIL_ROUNDS
                if result is False and old.get('action') == 'collect' and cell:
                    hits = self._collect_fail_hits.get(cell, 0) + 1
                    self._collect_fail_hits[cell] = hits
                    self.failed_cells[cell] = round_no + (
                        COLLECT_REPEAT_ROUNDS if hits >= 2 else COLLECT_FAIL_ROUNDS)
                    self._unlock_mine(str(rid), cell)
                if result is False and old.get('action') == 'acceptTask':
                    banned = self._pending_accept or cell
                    if banned:
                        self._banned_tasks.add(banned)
                    ledger = getattr(self.reasoning, 'ledger', None)
                    if ledger:
                        ledger.fail_current(error='acceptTask')
                if result is False and fingerprint:
                    self._fail_hits[fingerprint] = self._fail_hits.get(fingerprint, 0) + 1
                    if self._fail_hits[fingerprint] >= FAIL_HITS:
                        self._blacklist[fingerprint] = round_no + FAIL_ROUNDS
                elif result is True and fingerprint:
                    self._fail_hits.pop(fingerprint, None)
                if result is True and old.get('action') == 'collect' and cell:
                    ore = next((zone.get('neutralType') for zone in
                                (data.get('mapInfo') or {}).get('zones', [])
                                if pos(zone) == cell), None)
                    role = roles.get(str(rid))
                    prev_bag = self._last_bags.get(str(rid), Counter())
                    now_bag = Counter((role or {}).get('backpack', []))
                    grew = sum(now_bag[name] for name in ORES) > sum(prev_bag[name] for name in ORES)
                    if ore in ORES and grew:
                        self._mine_yield[(cell, ore)] += 1
                        self._stale_collect.pop((str(rid), cell), None)
                    else:
                        key = (str(rid), cell)
                        self._stale_collect[key] = self._stale_collect.get(key, 0) + 1
                        if self._stale_collect[key] >= STALE_COLLECT_HITS:
                            self.failed_cells[cell] = round_no + STALE_BLACKLIST
                            self._unlock_mine(str(rid), cell)
                            self._stale_collect.pop(key, None)
        self.failed_cells = {cell: expiry for cell, expiry in self.failed_cells.items()
                             if expiry > round_no}
        self._blacklist = {key: expiry for key, expiry in self._blacklist.items()
                           if expiry > round_no}
        response = self._decide(data)
        self._round, self._cached = round_no, copy.deepcopy(response)
        self._previous = copy.deepcopy(response['roleCommandMap'])
        self._last_bags = {str(role.get('id')): Counter(role.get('backpack', []))
                           for role in (data.get('teamOur') or {}).get('roles') or []
                           if isinstance(role, dict)}
        return response

    def _safe(self, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            LOG.exception("role action failed")
            return False

    def _decide(self, data):
        config = dict(self.config)
        if data.get('roundNo') == 0:
            self.config['round_origin'] = config['round_origin'] = 0
            self.reasoning.config['round_origin'] = 0
        w = World(data, config)
        self.w = w
        self._upgrade_dispatched = False
        self._observe_world()
        self.commands, self.reserved, self.build_jobs = {}, set(), set()
        self._wall_plan = None
        self.gold = int(data['teamOur'].get('goldNum', 0))
        self.prices = {x['name']: max(0, int(x['price'])) for x in data.get('vendorShopList', [])
                       if isinstance(x, dict) and 'name' in x and 'price' in x}
        self.shop = {x['name']: max(0, int(x['price'])) for x in data.get('weaponShopList', [])
                     if isinstance(x, dict) and 'name' in x and 'price' in x}
        self.market.observe(w.day, self.prices, self.reasoning)
        self.reasoning.blocked_ores = set(self.reasoning.blocked_ores) | set(self.market.blocked_ores)
        pioneers = [role for role in w.people if role['roleType'] == 'pioneer']
        pioneer = pioneers[0] if pioneers else None
        active = bool(data.get('phaseTask'))
        self._refresh_layout()
        thought = self.reasoning.update(data, w.day, str(pioneer['id']) if pioneer else None)
        response = {'roleCommandMap': self.commands, 'prompt': thought.get('prompt', ''),
                    'executeCmd': thought.get('executeCmd', '')}
        if active and pioneer and isinstance(thought.get('taskAnswer'), str):
            self.commands[str(pioneer['id'])] = command('submitAnswer', taskAnswer=thought['taskAnswer'])
        available = [role for role in w.people if not (active and role['roleType'] == 'pioneer')]
        LOG.info(
            "round=%s night=%s phase=%s walls=%s/%s stone_gap=%s cp=%s",
            data.get('roundNo'), w.night, w.phase_in_day,
            self._effective_walls(), self._wall_target(), self._stone_deficit(),
            self._control_point)
        for role in available:
            if w.night and role.get('roleType') == 'pioneer':
                continue
            self._safe(self._use_emergency, role)
        if w.night:
            self._safe(self._night, available, pioneer)
        else:
            self._safe(self._day, available, pioneer)
        self._filter_blacklisted()
        return response

    def _filter_blacklisted(self):
        blocked = []
        for rid, cmd in list(self.commands.items()):
            key = (str(rid), command_fingerprint(cmd))
            if key in self._blacklist:
                blocked.append(rid)
        for rid in blocked:
            self.commands.pop(rid, None)

    def _roles(self, available):
        pioneers = [role for role in available if role.get('roleType') == 'pioneer']
        workers = sorted(
            (role for role in available if role.get('roleType') == 'worker'),
            key=lambda role: (int(role['id']) if str(role['id']).isdigit() else 10 ** 9, str(role['id'])))
        return (pioneers[0] if pioneers else None,
                workers[0] if workers else None,
                workers[1] if len(workers) > 1 else None)

    def _observe_world(self):
        if self.w.night and self.w.base:
            team = (self.w.data.get('teamOur') or {}).get('type')
            accumulate_attack_votes(self.w.base, self.w.robots, team, self._attack_votes)
            if self._spawn_point is None and self.w.robots:
                farthest = max(
                    self.w.robots,
                    key=lambda robot: min(distance(robot, cell) for cell in footprint(self.w.base)))
                if min(distance(farthest, cell) for cell in footprint(self.w.base)) >= 8:
                    self._spawn_point = pos(farthest)
                    LOG.info("spawn locked %s", self._spawn_point)
        if self._summon_day != self.w.day:
            self._summon_day, self._summons_used = self.w.day, 0
        for building in self.w.ours:
            if building.get('roleType') not in ('station', 'wall') + GUNS:
                continue
            key = (building.get('roleType'), int(building.get('level', 1)))
            baseline = 1500 if building.get('roleType') == 'station' else 1000
            self._health_caps[key] = max(
                baseline, self._health_caps.get(key, 0), int(building.get('health', 0)))
        visible = {(pos(zone), zone.get('neutralType')) for zone in self.w.zones
                   if zone.get('neutralType') in ORES}
        self._mine_yield = Counter({key: value for key, value in self._mine_yield.items()
                                    if key in visible})
        for building in self.w.ours:
            if building.get('roleType') == 'wall':
                self._seen_walls.add(pos(building))
        live = {claim_id: claim for claim_id, claim in self.mine_claims.items()
                if (claim.get('pos'), claim.get('ore')) in visible
                and claim.get('pos') not in self.failed_cells}
        self.mine_claims = live

    def _attack_direction(self):
        if not self.w.base:
            return (0.0, 0.0)
        enemy = next((role for role in self.w.enemy if role.get('roleType') == 'station'), None)
        if self._attack_votes:
            return direction_from_votes(
                self._attack_votes, self.w.base, self.w.width, self.w.height, enemy)
        return default_attack_direction(self.w.base, self.w.width, self.w.height, enemy)

    def _refresh_layout(self):
        if not self.w.base:
            self._rocket_cells = []
            self._wall_plan = []
            self._control_point = None
            self._repair_post = None
            return
        direction = self._attack_direction()
        gun_cells = {pos(gun) for gun in self.w.guns}
        mobile = {pos(role) for role in self.w.people} | {pos(robot) for robot in self.w.robots}
        rocket_blocked = (self.w.blocked | set(self.failed_cells)) - gun_cells - mobile
        if len(self._rocket_cells) != ROCKET_TURRET_COUNT:
            guns, stand = assemble_base_plot(
                self.w.base, self.w.width, self.w.height, rocket_blocked, direction)
            self._rocket_cells = list(guns)
            if stand:
                self._control_point = stand
        our_walls = {pos(role) for role in self.w.ours if role.get('roleType') == 'wall'}
        wall_blocked = (self.w.blocked | set(self.failed_cells)) - our_walls
        if self._wall_slots and len(self._wall_slots) >= 10:
            plan = list(self._wall_slots)
        else:
            plan, gate = wall_build_order(
                self.w.base, self.w.width, self.w.height, direction,
                self._wall_gate, wall_blocked)
            if plan:
                self._wall_slots = list(plan)
                self._wall_gate = gate
        self._wall_plan = list(self._wall_slots or plan)
        guns = self._rocket_cells or [pos(gun) for gun in self.w.guns]
        self._control_point = choose_control_point(
            guns, self.w.base, self.w.width, self.w.height,
            set(self.failed_cells), self._control_point)
        self._repair_post = choose_repair_post(
            self.w.base, self._wall_plan, guns, self._control_point,
            self.w.width, self.w.height, set(self.failed_cells))
        walls_by_pos = {pos(wall): wall for wall in self.w.ours if wall.get('roleType') == 'wall'}
        front = front_wall_cells(self._wall_plan, self._control_point)
        self._wall_registry.observe(
            self._wall_plan, walls_by_pos, front, int(self.w.data.get('roundNo', 1)))

    def _effective_walls(self):
        slots = set(self._wall_slots or self._wall_plan or ())
        if not slots:
            return sum(role.get('roleType') == 'wall' for role in self.w.ours)
        return sum(role.get('roleType') == 'wall' and pos(role) in slots for role in self.w.ours)

    def _pending_walls(self):
        return sum(cmd.get('action') == 'build' and cmd.get('name') == 'wall'
                   for cmd in self.commands.values())

    def _wall_target(self):
        plan = list(self._wall_plan or self._wall_slots or [])
        return max(0, len(plan)) or TARGET_WALL_COUNT

    def _walls_done(self):
        target = self._wall_target()
        return target > 0 and self._effective_walls() + self._pending_walls() >= target

    def _team_stones(self):
        return sum(Counter(role.get('backpack', []))['stone'] for role in self.w.people)

    def _stone_deficit(self):
        return wall_stone_deficit(
            self._effective_walls() + self._pending_walls(), self._team_stones(),
            target=self._wall_target() or TARGET_WALL_COUNT)

    def _stone_keep_for(self, role):
        if role.get('roleType') == 'worker' and self._is_repairer(role):
            return STONE_KEEP
        return 0

    def _is_repairer(self, role):
        workers = sorted(
            (item for item in self.w.people if item.get('roleType') == 'worker'),
            key=lambda item: (int(item['id']) if str(item['id']).isdigit() else 10 ** 9, str(item['id'])))
        return bool(workers) and str(workers[0]['id']) == str(role['id'])

    def _fixer_stock(self, role=None):
        if role is not None:
            return Counter(role.get('backpack', []))[WALL_REPAIR_KIT]
        repairer = next((item for item in self.w.people if self._is_repairer(item)), None)
        if repairer:
            return Counter(repairer.get('backpack', []))[WALL_REPAIR_KIT]
        return sum(Counter(item.get('backpack', []))[WALL_REPAIR_KIT] for item in self.w.people)

    def _medicine_stock(self):
        return sum(Counter(role.get('backpack', []))['Medicine'] for role in self.w.people)

    def _base_health_ratio(self):
        if not self.w.base:
            return 0.0
        level = int(self.w.base.get('level', 1))
        cap = max(1, self._health_caps.get(('station', level), 1500))
        return int(self.w.base.get('health', 0)) / cap

    def _bag_capacity(self, role):
        return max(0, int(role.get('backPackCapability', 100)))

    def _bag_space(self, role):
        return max(0, self._bag_capacity(role) - len(role.get('backpack', [])))

    def _bag_full(self, role):
        return self._bag_space(role) <= 0

    def _bag_ratio(self, role):
        cap = self._bag_capacity(role)
        if cap <= 0:
            return 1.0
        return len(role.get('backpack', [])) / cap

    def _path(self, role, targets):
        return self.w.path(role, self.w.interaction_cells(targets), self.reserved)

    def _path_to_cell(self, role, cell):
        return self.w.path(role, [cell], self.reserved)

    def _go(self, role, targets):
        path = self._path(role, targets)
        if not path:
            return False
        if len(path) > 1:
            dest = path[1]
            if dest in self.reserved:
                return True
            self.commands[str(role['id'])] = command('move', dest)
            self.reserved.add(dest)
        return True

    def _go_cell(self, role, cell):
        if cell is None:
            return False
        if pos(role) == cell:
            return False
        path = self._path_to_cell(role, cell)
        if not path:
            return False
        if len(path) > 1:
            dest = path[1]
            if dest in self.reserved:
                return True
            self.commands[str(role['id'])] = command('move', dest)
            self.reserved.add(dest)
        return True

    def _adjacent(self, role, targets):
        return any(distance(pos(role), target) == 1 for target in targets)

    def _zones(self, kind):
        return [pos(zone) for zone in self.w.zones if zone.get('neutralType') == kind]

    def _max_health(self, role):
        return 200 if role.get('roleType') == 'pioneer' else 220

    def _use_emergency(self, role):
        bag = role.get('backpack', [])
        medicine = next((item for item in bag if item.lower() == 'medicine'), None)
        if not medicine:
            return False
        hp = int(role.get('health', 0))
        if role.get('roleType') == 'pioneer':
            if hp > PIONEER_MEDICINE_HP:
                return False
        elif hp >= self._max_health(role) * BASE_LOW_HEALTH_RATIO:
            return False
        self.commands[str(role['id'])] = command('use', name=medicine)
        return True

    def _facing_key(self, building):
        if not self.w.base:
            return 0.0
        return facing_score(pos(building), base_center(self.w.base), self._attack_direction())

    def _health_ratio(self, building):
        kind = building.get('roleType')
        level = int(building.get('level', 1))
        cap = self._health_caps.get((kind, level), 1500 if kind == 'station' else 1000)
        return int(building.get('health', 0)) / max(1, cap)

    def _weapons_l2(self):
        if len(self.w.guns) < ROCKET_TURRET_COUNT:
            return False
        return all(int(gun.get('level', 1)) >= 2 for gun in self.w.guns)

    def _l3_walls(self):
        return sum(1 for wall in self.w.ours
                   if wall.get('roleType') == 'wall' and int(wall.get('level', 1)) >= 3)

    def _has_l1_walls(self):
        return any(wall.get('roleType') == 'wall' and int(wall.get('level', 1)) < 2
                   for wall in self.w.ours)

    def _walls_hurt(self):
        walls = [wall for wall in self.w.ours if wall.get('roleType') == 'wall']
        damaged = sum(1 for wall in walls if self._health_ratio(wall) < 0.999)
        critical = any(self._health_ratio(wall) < WALL_FIX_RATIO for wall in walls)
        return damaged >= 3 or critical

    def _need_repair_post(self):
        return self.w.day >= 4 or self._walls_hurt()

    def _wave_clear(self):
        return not our_wave(self.w)

    def _base_pressed(self):
        if not self.w.base:
            return False
        cells = footprint(self.w.base)
        return any(min(distance(robot, cell) for cell in cells) <= ROBOT_BASE_DANGER
                   for robot in our_wave(self.w))

    def _robot_near(self, cell, radius):
        if cell is None:
            return False
        return any(distance(robot, cell) <= radius for robot in our_wave(self.w))

    def _nearest_robots(self, role, radius=None):
        robots = [robot for robot in self.w.robots if int(robot.get('health', 0)) > 0]
        if radius is not None:
            robots = [robot for robot in robots if distance(role, robot) <= radius]
        return robots

    def _night_unsafe(self, cell):
        return any(distance(robot, cell) <= WORKER_DANGER
                   for robot in self.w.robots if int(robot.get('health', 0)) > 0)

    def _dusk(self):
        return (not self.w.night) and self.w.remaining_day <= DUSK_LEAD

    def _inner_cells(self):
        if not self.w.base:
            return []
        blue, _ = building_rings(self.w.base, self.w.width, self.w.height)
        cells = set(blue) | set(footprint(self.w.base))
        cells -= set(self._rocket_cells or ())
        if self._control_point and self._robot_near(self._control_point, PIONEER_RETREAT):
            cells.discard(self._control_point)
        return [cell for cell in cells if cell not in self.w.blocked or cell == self._repair_post]

    def _retreat_inner(self, role):
        cells = self._inner_cells()
        if self._repair_post and self._repair_post in cells:
            return self._go_cell(role, self._repair_post) or self._go(role, cells)
        return self._go(role, cells) if cells else False

    def _inside_walls(self, role):
        if not self.w.base:
            return False
        blue, _ = building_rings(self.w.base, self.w.width, self.w.height)
        here = pos(role)
        return here in blue or here in footprint(self.w.base) or here == self._control_point

    def _use_adjacent(self, role, allow_upgrade=True, wall_only=False):
        bag = role.get('backpack', [])
        rid = str(role['id'])
        buildings = []
        wanted = []
        for index, item in enumerate(self._upgrade_agenda()):
            if item.get('cell') and item.get('name'):
                wanted.append((item['cell'], item['name'], index))
        if allow_upgrade:
            for building in self.w.ours:
                kind = building.get('roleType')
                prefix = {'station': 'Station', 'wall': 'Wall',
                          'gatling': 'Weapon', 'rocket': 'Weapon', 'railgun': 'Weapon'}.get(kind)
                level = int(building.get('level', 1))
                voucher = '%sUpgradeVoucher%d' % (prefix, level) if prefix else None
                if wall_only and kind != 'wall':
                    continue
                if kind == 'station' and self.gold < BASE_UPGRADE_GOLD and voucher in bag:
                    continue
                if kind == 'wall' and level >= 3:
                    continue
                if kind == 'wall' and level >= 2 and self._has_l1_walls():
                    continue
                if prefix and level < 3 and voucher in bag and self._adjacent(role, [pos(building)]):
                    buildings.append((building, voucher))
            def _adjacent_rank(item):
                building, voucher = item
                cell = pos(building)
                for target, name, index in wanted:
                    if target == cell and name == voucher:
                        return (0, index)
                return (
                    1,
                    0 if building.get('roleType') in GUNS else 1 if building.get('roleType') == 'wall' else 2,
                    0 if cell in front_wall_cells(self._wall_plan, self._control_point) else 1,
                    self._health_ratio(building) if building.get('roleType') == 'wall' else 0,
                )
            buildings.sort(key=_adjacent_rank)
            if buildings and not self._upgrade_dispatched:
                building, voucher = buildings[0]
                self.commands[rid] = command('use', pos(building), name=voucher)
                self._upgrade_dispatched = True
                return True
        if WALL_REPAIR_KIT in bag:
            threshold = WALL_CRITICAL_RATIO if not self._weapons_l2() else WALL_FIX_RATIO
            walls = [wall for wall in self.w.ours
                     if wall.get('roleType') == 'wall' and self._adjacent(role, [pos(wall)])
                     and (int(wall.get('level', 1)) >= 3 or not allow_upgrade)
                     and self._health_ratio(wall) < threshold]
            if not walls and WALL_REPAIR_KIT in bag:
                walls = [wall for wall in self.w.ours
                         if wall.get('roleType') == 'wall' and self._adjacent(role, [pos(wall)])
                         and int(wall.get('level', 1)) >= 3
                         and self._health_ratio(wall) < threshold]
            if walls:
                target = min(walls, key=lambda wall: (
                    0 if pos(wall) in front_wall_cells(self._wall_plan, self._control_point) else 1,
                    self._health_ratio(wall), str(wall['id'])))
                self.commands[rid] = command('use', pos(target), name=WALL_REPAIR_KIT)
                return True
        return False

    def _night(self, available, pioneer):
        gunner, repairer, miner = self._roles(available)
        if gunner or (pioneer and str(pioneer['id']) in {str(role['id']) for role in available}):
            self._night_gunner(gunner or pioneer)
        elif self._guns_unmanned():
            volunteer = miner or repairer or next(
                (role for role in available if role.get('roleType') == 'worker'), None)
            if volunteer:
                self._night_gunner(volunteer)
        if repairer and str(repairer['id']) not in self.commands:
            if self._need_repair_post():
                self._night_repairer(repairer)
            else:
                self._night_worker(repairer, True)
        if miner and str(miner['id']) not in self.commands:
            self._night_worker(miner, False)
        for extra in available:
            if extra.get('roleType') == 'worker' and str(extra['id']) not in self.commands:
                self._night_worker(extra, False)

    def _night_gunner(self, role):
        if role is None or str(role['id']) in self.commands:
            return False
        if self._robot_near(self._control_point, PIONEER_RETREAT):
            return self._retreat_inner(role)
        if self._wave_clear() and self.w.remaining_night > NIGHT_CLEAR_ROUNDS:
            if self._buy_weapon_vouchers(role) or self._accept_task(role):
                return True
        if self._control_point and pos(role) != self._control_point:
            return self._go_cell(role, self._control_point)
        guns = [gun for gun in self.w.guns if distance(role, gun) == 1]
        guns.sort(key=lambda gun: str(gun['id']))
        ready = [gun for gun in guns if int(gun.get('cooldown', 0)) <= 0]
        if ready:
            start = self._gun_cursor % len(ready)
            ordered = ready[start:] + ready[:start]
            chosen = ordered[0]
            attacks, _ = plan_attacks(self.w, {str(role['id']): chosen})
            if attacks:
                self.commands.update(attacks)
                self._gun_cursor = (self._gun_cursor + 1) % max(1, len(guns))
                return True
        if int(role.get('health', 0)) <= PIONEER_MEDICINE_HP:
            return self._use_emergency(role)
        return False

    def _night_repairer(self, role):
        if self._guns_unmanned():
            return self._night_gunner(role)
        if self._finish_current_mine(role, night=True):
            return True
        if self._apply_upgrade_ticket(role) or self._use_adjacent(role, allow_upgrade=True):
            return True
        if self._repair_post and pos(role) != self._repair_post:
            if self._go_cell(role, self._repair_post):
                return True
        walls = [wall for wall in self.w.ours if wall.get('roleType') == 'wall']
        if walls:
            target = min(walls, key=lambda wall: (
                0 if pos(wall) in front_wall_cells(self._wall_plan, self._control_point) else 1,
                self._health_ratio(wall), str(wall['id'])))
            if self._go(role, [pos(target)]):
                return True
        return self._night_worker(role, True)

    def _night_worker(self, role, is_repairer):
        if str(role['id']) in self.commands:
            return True
        if self._guns_unmanned():
            return self._night_gunner(role)
        if self._finish_current_mine(role, night=True):
            return True
        close = self._nearest_robots(role, WORKER_DANGER)
        rid = str(role['id'])
        if close or self._evading.get(rid):
            far = not self._nearest_robots(role, WORKER_RESUME - 1)
            if far and not close:
                self._evading.pop(rid, None)
            else:
                self._evading[rid] = True
                if self._base_pressed():
                    return self._retreat_inner(role)
                return self._flee(role, close or self._nearest_robots(role))
        if is_repairer and self._use_adjacent(role, allow_upgrade=True):
            return True
        return self._economy(role, is_repairer, night=True)

    def _flee(self, role, robots):
        if not robots:
            return False
        here = pos(role)
        best = None
        for nxt in self.w.neighbors(here):
            if nxt in self.reserved or nxt in (self.w.blocked - {here}):
                continue
            gap = min(distance(nxt, robot) for robot in robots)
            toward_home = 0
            if self.w.base and not self._base_pressed():
                toward_home = -distance(nxt, self.w.base)
            key = (gap, toward_home, nxt)
            if best is None or key > best[0]:
                best = (key, nxt)
        if best is None or best[1] == here:
            return False
        self.commands[str(role['id'])] = command('move', best[1])
        self.reserved.add(best[1])
        return True

    def _day(self, available, pioneer):
        gunner, repairer, miner = self._roles(available)
        if pioneer and str(pioneer['id']) not in self.commands:
            self._pioneer_day(pioneer)
        if repairer and str(repairer['id']) not in self.commands:
            self._repairer_day(repairer)
        if miner and str(miner['id']) not in self.commands:
            self._miner_day(miner)
        for extra in available:
            if extra.get('roleType') == 'worker' and str(extra['id']) not in self.commands:
                self._economy(extra, False)

    def _pioneer_day(self, role):
        if str(role['id']) in self.commands:
            return True
        if self._buy_weapon_vouchers(role):
            return True
        if self._upgrade_weapon(role):
            return True
        if self._pioneer_should_home(role):
            target = self._control_point or (pos(self.w.base) if self.w.base else None)
            if target:
                return self._go_cell(role, target) if self._control_point else self._go(role, [target])
        if self._accept_task(role):
            return True
        if self._treasure(role):
            return True
        if self.w.day == 1 and self.w.base and distance(role, self.w.base) > 6:
            return self._go(role, [pos(self.w.base)])
        return False

    def _pioneer_should_home(self, role):
        if self.w.data.get('phaseTask'):
            return False
        target = self._control_point or (pos(self.w.base) if self.w.base else None)
        if not target:
            return False
        path = self._path_to_cell(role, target) if self._control_point else self._path(role, [target])
        walk = (len(path) - 1) if path else distance(role, target)
        return self.w.remaining_day <= walk + PIONEER_HOME_BUFFER

    def _upgrade_weapon(self, role):
        bag = Counter(role.get('backpack', []))
        guns = sorted(self.w.guns, key=lambda gun: (int(gun.get('level', 1)), str(gun['id'])))
        for gun in guns:
            level = int(gun.get('level', 1))
            name = 'WeaponUpgradeVoucher%d' % level
            if level < 3 and bag[name]:
                if self._adjacent(role, [pos(gun)]):
                    if self._upgrade_dispatched:
                        return False
                    self.commands[str(role['id'])] = command('use', pos(gun), name=name)
                    self._upgrade_dispatched = True
                    return True
                return self._go(role, [pos(gun)])
        return False

    def _repairer_day(self, role):
        if str(role['id']) in self.commands:
            return True
        if self._finish_current_mine(role, night=False):
            return True
        if self._handle_repairer_return(role):
            return True
        if not self._walls_done() or len(self.w.guns) < ROCKET_TURRET_COUNT:
            if self._build_rockets(role):
                return True
            if self._gather_for_walls(role):
                return True
            if self._build_wall(role):
                return True
        if self._shop_for_repairer(role):
            return True
        if self._apply_upgrade_ticket(role):
            return True
        return self._economy(role, True)

    def _handle_repairer_return(self, role):
        if self.w.day < 4:
            self._repairer_return = False
            self._repairer_return_round = None
            return False
        if self._inside_walls(role):
            self._repairer_return = False
            self._repairer_return_round = None
            return False
        path = None
        if self._repair_post:
            path = self._path_to_cell(role, self._repair_post)
        if path is None and self.w.base:
            path = self._path(role, [pos(self.w.base)])
        walk = (len(path) - 1) if path else 8
        if self.w.remaining_day <= walk + 2:
            self._repairer_return = True
            if self._repairer_return_round is None:
                self._repairer_return_round = int(self.w.data.get('roundNo', 1))
        if not self._repairer_return:
            return False
        started = self._repairer_return_round or int(self.w.data.get('roundNo', 1))
        if int(self.w.data.get('roundNo', 1)) - started >= REPAIRER_HOME_GIVEUP:
            self._repairer_return = False
            self._repairer_return_round = None
            return False
        if self._repair_post:
            return self._go_cell(role, self._repair_post) or self._retreat_inner(role)
        return self._retreat_inner(role)

    def _miner_day(self, role):
        if str(role['id']) in self.commands:
            return True
        if self._finish_current_mine(role, night=False):
            return True
        stones = Counter(role.get('backpack', []))['stone']
        leftover = max(0, self._wall_target() - self._effective_walls() - self._pending_walls())
        emergency = (self.w.day == 1 and leftover > 0
                     and (stones > MINER_HELP_STONES or self.w.remaining_day <= leftover + 6))
        if emergency and leftover > 0:
            if stones and self._build_wall(role):
                return True
            if leftover and self._collect_ore(role, 'stone'):
                return True
        return self._economy(role, False)

    def _gather_for_walls(self, role):
        if self._finish_current_mine(role, night=False):
            return True
        leftover = max(0, self._wall_target() - self._effective_walls() - self._pending_walls())
        if leftover <= 0:
            return False
        stones = Counter(role.get('backpack', []))['stone']
        batch = min(leftover, STONE_BATCH_DAY1 if self.w.day == 1 else STONE_BATCH_LATER)
        urgent = self.w.remaining_day <= leftover + 6
        if stones and (stones >= batch or urgent):
            return False
        if self._stone_deficit() <= 0 and stones:
            return False
        return self._collect_ore(role, 'stone')

    def _economy(self, role, is_repairer, night=False):
        if str(role['id']) in self.commands:
            return True
        if self._finish_current_mine(role, night):
            return True
        if night and self._should_sell(role, night=True):
            if self._sell(role, True):
                return True
        elif self._should_sell(role, night=False):
            if self._sell(role, True):
                return True
        if is_repairer and not night and self._shop_for_repairer(role):
            return True
        return self._mine(role, is_repairer, night)

    def _ore_value(self, role):
        counts = self._sellable_ores(role)
        return sum(int(self.prices.get(ore, 0)) * count for ore, count in counts.items())

    def _vendor_dist(self, role):
        vendors = self._zones('vendor')
        if not vendors:
            return None
        path = self._path(role, vendors)
        if path is None:
            return None
        return max(0, len(path) - 1)

    def _need_weapon_gold(self):
        pending = sum(cmd.get('action') == 'build' and cmd.get('name') in GUNS
                      for cmd in self.commands.values())
        return len(self.w.guns) + pending < ROCKET_TURRET_COUNT and self.gold < WEAPON_GOLD

    def _should_sell(self, role, night=False):
        if self._open_mine_claim(role) and not self._bag_full(role):
            if not (night and self._guns_unmanned()):
                return False
        urgent = self._bag_full(role) or self._need_weapon_gold()
        if not self._sellable_ores(role, emergency=urgent):
            return False
        vendors = self._zones('vendor')
        if not vendors:
            return False
        if night and any(self._night_unsafe(vendor) for vendor in vendors):
            return False
        if urgent:
            return True
        sellable = self._sellable_ores(role)
        if sellable and all(self.reasoning.should_stockpile(ore, self.w.day)
                            for ore in sellable):
            return False
        if self._bag_ratio(role) >= BAG_SELL_RATIO:
            return True
        dist = self._vendor_dist(role)
        value = self._ore_value(role)
        if dist is None:
            return False
        if dist <= SELL_NEAR_DIST and value >= SELL_NEAR_VALUE:
            return True
        if value >= SELL_TRIP_VALUE and dist <= SELL_TRIP_DIST:
            return True
        return False

    def _sellable_ores(self, role, keep=None, emergency=False):
        counts = Counter(role.get('backpack', [])) - Counter(keep or {})
        counts['stone'] = max(0, counts['stone'] - self._stone_keep_for(role))
        if not emergency:
            held = self.market.reserves(role, emergency=False)
            for ore, reserved in held.items():
                if not self.market.take_profit(ore):
                    counts[ore] = max(0, counts[ore] - reserved)
        return {ore: counts[ore] for ore in ORES if counts[ore] and self.prices.get(ore, 0) > 0}

    def _predicted(self, ore):
        return self.reasoning.predicted_price(ore, self.prices.get(ore, 0), self.w.day)

    def _sell(self, role, force=False, keep=None):
        vendors = self._zones('vendor')
        urgent = force or self._bag_full(role) or self._need_weapon_gold()
        ore_counts = self._sellable_ores(role, keep, emergency=urgent)
        if not ore_counts or not vendors:
            return False
        if self.w.night and any(self._night_unsafe(vendor) for vendor in vendors):
            return False
        if self._adjacent(role, vendors):
            ore = max(ore_counts, key=lambda name: (
                self.reasoning.sale_boost(name, self.w.day),
                self._predicted(name), ore_counts[name], name))
            self.commands[str(role['id'])] = command('sell', name=ore, num=ore_counts[ore])
            LOG.info("sell %s x%s keep_stone=%s", ore, ore_counts[ore], self._stone_keep_for(role))
            return True
        if not force and not self._bag_full(role) and self._bag_ratio(role) < BAG_SELL_RATIO:
            return False
        return self._go(role, vendors)

    def _collect_ore(self, role, ore):
        return self._mine(role, True, False, force_ores=(ore,))

    def _unlock_mine(self, rid, cell=None):
        claim = self.mine_claims.get(rid)
        if claim and (cell is None or claim.get('pos') == cell):
            self.mine_claims.pop(rid, None)

    def _open_mine_claim(self, role):
        claim = self.mine_claims.get(str(role['id']))
        if not claim or claim.get('pos') in self.failed_cells:
            return None
        if not any(pos(zone) == claim['pos'] and zone.get('neutralType') == claim['ore']
                   for zone in self.w.zones):
            return None
        return claim

    def _guns_unmanned(self):
        if not getattr(self, 'w', None) or not self.w.night or not self.w.guns:
            return False
        for person in self.w.people:
            if int(person.get('health', 0)) <= 0:
                continue
            if self._control_point and pos(person) == self._control_point:
                return False
            if any(distance(person, gun) == 1 for gun in self.w.guns):
                return False
        return True

    def _finish_current_mine(self, role, night=False):
        if self._bag_full(role):
            return False
        if night and self._guns_unmanned():
            return False
        claim = self._open_mine_claim(role)
        if not claim:
            return False
        path = self._path(role, [claim['pos']])
        if not path:
            self._unlock_mine(str(role['id']), claim['pos'])
            return False
        return self._walk_collect(role, claim['pos'], path)

    def _mine(self, role, is_repairer, night=False, force_ores=None):
        if self._bag_full(role):
            return self._sell(role, True)
        if self._finish_current_mine(role, night):
            return True
        if night and self._guns_unmanned():
            return False
        blocked = set(getattr(self.reasoning, 'blocked_ores', set()) or ())
        blocked |= set(getattr(self.market, 'blocked_ores', set()) or ())
        money = self._walls_done() or (not is_repairer and self.w.day >= 1 and force_ores is None)
        if force_ores:
            ores = tuple(force_ores)
            money = False
        else:
            deficit = self._stone_deficit() if is_repairer and not self._walls_done() else 0
            ores = preferred_ores(deficit, self.prices, blocked, money=money and deficit <= 0)
            if not is_repairer and 'stone' in ores and self.w.day == 1 and not self._walls_done():
                ores = tuple(item for item in ores if item != 'stone') or ores
        rid = str(role['id'])
        claim = self.mine_claims.get(rid)
        if claim and not self._open_mine_claim(role):
            self._unlock_mine(rid)
        candidates = self._ore_candidates(role, ores, night)
        if not candidates:
            return self._sell(role, True) if self._sellable_ores(role) else False
        score, _, ore, target, path = max(candidates)
        self.mine_claims[rid] = {'pos': target, 'ore': ore, 'score': score}
        return self._walk_collect(role, target, path)

    def _walk_collect(self, role, target, path):
        landing = path[-1]
        if landing in self.reserved and landing != pos(role):
            return True
        if len(path) == 1:
            self.commands[str(role['id'])] = command('collect', target)
            return True
        return self._go(role, [target])

    def _ore_candidates(self, role, ores, night=False):
        candidates = []
        enemy = next((item for item in self.w.enemy if item.get('roleType') == 'station'), None)
        claimed = {claim['pos'] for other, claim in self.mine_claims.items()
                   if other != str(role['id'])}
        near, far = [], []
        for zone in self.w.zones:
            ore = zone.get('neutralType')
            target = pos(zone)
            if ore not in ores or target in self.failed_cells or target in claimed:
                continue
            if not on_our_side(target, self.w.base, enemy, self.w.width, self.w.height):
                continue
            if night and self._night_unsafe(target):
                continue
            if (night or self._dusk()) and self._spawn_point and distance(target, self._spawn_point) <= SPAWN_AVOID:
                continue
            if (night or self._dusk()) and in_march_corridor(target, self._spawn_point, self.w.base):
                continue
            path = self._path(role, [target])
            if not path:
                continue
            travel = max(0, len(path) - 1)
            price = self._predicted(ore)
            stockpile = 1.35 if self.reasoning.should_stockpile(ore, self.w.day) else 1.0
            weight = self.market.collection_weight(ore, role)
            rate = price * stockpile * weight * 10 / (2 * travel + 10)
            base_dist = min(distance(target, cell) for cell in footprint(self.w.base)) if self.w.base else travel
            item = (rate, -travel, ore, target, path)
            if base_dist <= NEAR_MINE_RADIUS:
                near.append(item)
            else:
                far.append(item)
        return near or far

    def _build_rockets(self, role):
        if self.w.night or role['roleType'] != 'worker':
            return False
        pending = sum(cmd.get('action') == 'build' and cmd.get('name') in GUNS
                      for cmd in self.commands.values())
        if len(self.w.guns) + pending >= ROCKET_TURRET_COUNT:
            return False
        action = self.auto_construction.propose(
            self.w, role, self.reserved, self.gold, getattr(self, '_rocket_cells', None))
        if action is None:
            return False
        self.commands[str(role['id'])] = action
        target = pos(action['targetPos'][0])
        self.reserved.add(target)
        if action['action'] == 'build':
            self.gold -= WEAPON_GOLD
            self.build_jobs.add(target)
        return True

    def _build_wall(self, role):
        if not self.w.base or role.get('roleType') != 'worker':
            return False
        plan = list(self._wall_plan or self._wall_slots or [])
        current = {pos(item) for item in self.w.ours if item.get('roleType') == 'wall'}
        if self._effective_walls() + self._pending_walls() >= len(plan) and plan:
            return False
        blocked = self.w.blocked | self.reserved | set(self.failed_cells)
        missing = [cell for cell in plan if cell not in current]
        if not missing:
            return False
        stones = Counter(role.get('backpack', []))['stone']
        if not stones:
            return False
        for cell in missing:
            if cell in blocked and cell not in current:
                continue
            path = self._path(role, [cell])
            if not path:
                continue
            if len(path) == 1:
                self.commands[str(role['id'])] = command('build', cell, name='wall')
                self.reserved.add(cell)
                self.build_jobs.add(cell)
                return True
            return self._go(role, [cell])
        LOG.info("wall build skipped: no reachable missing cell")
        return False

    def _owned(self, name, role=None):
        people = [role] if role is not None else self.w.people
        bags = sum(Counter(item.get('backpack', []))[name] for item in people if item)
        pending = sum(cmd.get('action') == 'buy' and cmd.get('name') == name
                      for cmd in self.commands.values())
        return bags + pending

    def _can_buy(self, name, number=1, keep_reserve=True):
        price = self.shop.get(name)
        reserve = EMERGENCY_GOLD if keep_reserve else 0
        if not price or self.gold - price * number < reserve:
            return False
        return True

    def _weapons_l3(self):
        return (len(self.w.guns) >= ROCKET_TURRET_COUNT
                and all(int(gun.get('level', 1)) >= 3 for gun in self.w.guns))

    def _key_wall_target(self):
        walls = [wall for wall in self.w.ours if wall.get('roleType') == 'wall']
        return min(FRONT_WALL_L2_CAP, len(walls))

    def _facing_half(self):
        marked = {cell for cell, rec in self._wall_registry.cells.items() if rec.get('front')}
        if marked:
            return marked
        return front_wall_cells(self._wall_plan, self._control_point)

    def _restored_cells(self):
        restored = set()
        for cell, rec in self._wall_registry.cells.items():
            if rec.get('exists') and rec.get('rebuilt_round') and int(rec.get('level') or 0) < 3:
                restored.add(cell)
        return restored

    def _cp_gap(self, cell):
        if not self._control_point:
            return 0
        return distance(cell, self._control_point)

    def _wall_rank(self, wall):
        cell = pos(wall)
        return (
            0 if cell in self._restored_cells() else 1,
            -self._cp_gap(cell),
            cell[0],
            cell[1],
        )

    def _upgrade_agenda(self):
        """Weapon L2 → critical/hurt walls → facing L2 cap → weapon L3 → rest L2 → wall L3 → station."""
        agenda = []
        guns = sorted(self.w.guns, key=lambda gun: (int(gun.get('level', 1)), str(gun.get('id'))))
        walls = [wall for wall in self.w.ours if wall.get('roleType') == 'wall']
        guns_need_l2 = any(int(gun.get('level', 1)) < 2 for gun in guns)
        leftover_l1 = any(int(wall.get('level', 1)) == 1 for wall in walls)
        facing = self._facing_half()
        hurt_line = WALL_CRITICAL_RATIO if guns_need_l2 else WALL_HURT_RATIO

        def push(name, cell, kind, qty=1):
            agenda.append({'name': name, 'cell': cell, 'kind': kind, 'qty': qty})

        for gun in guns:
            if int(gun.get('level', 1)) == 1:
                push('WeaponUpgradeVoucher1', pos(gun), 'weapon')
        hurt = [wall for wall in walls
                if int(wall.get('level', 1)) == 1 and self._health_ratio(wall) < hurt_line]
        hurt.sort(key=lambda wall: (self._health_ratio(wall), -self._cp_gap(pos(wall))))
        for wall in hurt:
            push('WallUpgradeVoucher1', pos(wall), 'wall')
        if not guns_need_l2:
            raised = 0
            facing_l1 = [wall for wall in walls
                         if int(wall.get('level', 1)) == 1 and pos(wall) in facing]
            facing_l1.sort(key=self._wall_rank)
            for wall in facing_l1:
                if raised >= FRONT_WALL_L2_CAP:
                    break
                push('WallUpgradeVoucher1', pos(wall), 'wall')
                raised += 1
            for gun in guns:
                if int(gun.get('level', 1)) == 2:
                    push('WeaponUpgradeVoucher2', pos(gun), 'weapon')
            leftover = [wall for wall in walls if int(wall.get('level', 1)) == 1]
            leftover.sort(key=self._wall_rank)
            for wall in leftover:
                push('WallUpgradeVoucher1', pos(wall), 'wall')
        if not leftover_l1:
            seniors = [wall for wall in walls if int(wall.get('level', 1)) == 2]
            seniors.sort(key=self._wall_rank)
            for wall in seniors:
                push('WallUpgradeVoucher2', pos(wall), 'wall')
        if self.w.base and 1 <= int(self.w.base.get('level', 1)) <= 2 and self.gold >= BASE_UPGRADE_GOLD:
            push('StationUpgradeVoucher%d' % int(self.w.base.get('level', 1)),
                 pos(self.w.base), 'station')
        if self.w.day >= 3:
            target = fixer_target_stock(self.w.day, self._l3_walls())
            gap = fixer_restock_count(self._fixer_stock(), target)
            if gap:
                push(WALL_REPAIR_KIT, None, 'stock', gap)
        seen = set()
        unique = []
        for item in agenda:
            key = (item['kind'], item['name'], item['cell'], item['qty'])
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique

    def _apply_upgrade_ticket(self, role):
        bag = role.get('backpack', [])
        for item in self._upgrade_agenda():
            if not item.get('cell') or item['name'] not in bag:
                continue
            cell = item['cell']
            if self._adjacent(role, [cell]):
                if self._upgrade_dispatched:
                    return False
                self.commands[str(role['id'])] = command('use', cell, name=item['name'])
                self._upgrade_dispatched = True
                return True
            return self._go(role, [cell])
        return False

    def _weapon_voucher_needs(self):
        counts = Counter()
        for item in self._upgrade_agenda():
            if item['kind'] == 'weapon':
                counts[item['name']] += 1
        needs = []
        for name, total in counts.items():
            gap = total - self._owned(name)
            if gap > 0 and name in self.shop:
                needs.append((name, gap))
        return needs

    def _wall_voucher(self):
        for item in self._upgrade_agenda():
            if item['kind'] != 'wall':
                continue
            name = item['name']
            if name in self.shop and not self._owned(name):
                return name
        return None

    def _station_voucher(self):
        if not self.w.base or self.gold < BASE_UPGRADE_GOLD:
            return None
        level = int(self.w.base.get('level', 1))
        name = 'StationUpgradeVoucher%d' % level
        if level < 3 and name in self.shop and not self._owned(name):
            return name
        return None

    def _buy_at_shop(self, role, name, number=1):
        shops = self._zones('weaponShop')
        keep_reserve = name != 'Medicine'
        if (not shops or not self._can_buy(name, number, keep_reserve=keep_reserve)
                or self._bag_space(role) < number):
            return False
        if name in SUMMON_ORDERS:
            return False
        if not self._adjacent(role, shops):
            return self._go(role, shops)
        cost = self.shop[name] * number
        self.commands[str(role['id'])] = command('buy', name=name, num=number)
        self.gold -= cost
        return True

    def _buy_weapon_vouchers(self, role):
        if len(self.w.guns) < ROCKET_TURRET_COUNT:
            return False
        needs = self._weapon_voucher_needs()
        if not needs:
            return False
        name, number = needs[0]
        number = min(number, self._bag_space(role), self.gold // max(1, self.shop.get(name, 10 ** 9)))
        if number < 1:
            return False
        return self._buy_at_shop(role, name, number)

    def _shop_for_repairer(self, role):
        if len(self.w.guns) < ROCKET_TURRET_COUNT:
            return False
        if self._buy_weapon_vouchers(role):
            return True
        wall = self._wall_voucher()
        if wall and self._can_buy(wall):
            return self._buy_at_shop(role, wall, 1)
        if self._station_voucher() and self._can_buy(self._station_voucher()):
            return self._buy_at_shop(role, self._station_voucher(), 1)
        if ('Medicine' in self.shop and self._medicine_stock() < MEDICINE_TARGET_STOCK
                and self._can_buy('Medicine', keep_reserve=False)):
            return self._buy_at_shop(role, 'Medicine', 1)
        if self.w.day >= 3:
            target = fixer_target_stock(self.w.day, self._l3_walls())
            need = fixer_restock_count(self._fixer_stock(role), target)
            if need and self._can_buy(WALL_REPAIR_KIT):
                number = min(need, self._bag_space(role), self.gold // self.shop[WALL_REPAIR_KIT])
                if number >= 1:
                    return self._buy_at_shop(role, WALL_REPAIR_KIT, number)
        return False

    def _purchase_plan(self, role):
        needs = self._weapon_voucher_needs()
        if needs and self._can_buy(needs[0][0]):
            name, number = needs[0]
            number = min(number, self._bag_space(role), self.gold // self.shop[name])
            if number >= 1:
                return name, number
        if self._is_repairer(role):
            wall = self._wall_voucher()
            if wall and self._can_buy(wall):
                return wall, 1
        if self._station_voucher() and self._can_buy(self._station_voucher()):
            return self._station_voucher(), 1
        if self._is_repairer(role) and self.w.day >= 3:
            target = fixer_target_stock(self.w.day, self._l3_walls())
            need = fixer_restock_count(self._fixer_stock(role), target)
            if need and self._can_buy(WALL_REPAIR_KIT):
                return WALL_REPAIR_KIT, min(need, self._bag_space(role),
                                            self.gold // self.shop[WALL_REPAIR_KIT])
        return None

    def _buy_fixer(self, role, shops):
        target = fixer_target_stock(max(self.w.day, 3), self._l3_walls())
        need = fixer_restock_count(self._fixer_stock(role), target)
        if not need or not self._can_buy(WALL_REPAIR_KIT):
            return False
        number = min(need, self._bag_space(role), self.gold // self.shop[WALL_REPAIR_KIT])
        if number < 1:
            return False
        if not self._adjacent(role, shops):
            return self._go(role, shops)
        self.commands[str(role['id'])] = command('buy', name=WALL_REPAIR_KIT, num=number)
        self.gold -= self.shop[WALL_REPAIR_KIT] * number
        return True

    def _next_upgrade(self):
        for item in self._upgrade_agenda():
            if item['kind'] not in ('weapon', 'wall', 'station') or not item.get('cell'):
                continue
            building = next((unit for unit in self.w.ours if pos(unit) == item['cell']), None)
            if building is None and item['kind'] == 'weapon':
                building = next((gun for gun in self.w.guns if pos(gun) == item['cell']), None)
            return item['name'], building
        return None

    def _available_player_tasks(self):
        side = (self.w.data.get('teamOur') or {}).get('type', '')
        ledger = getattr(self.reasoning, 'ledger', None)
        ready = []
        last_family = None
        if ledger:
            for tid in reversed(getattr(ledger, 'order', []) or []):
                item = (getattr(ledger, 'tasks', {}) or {}).get(tid)
                if item and item.get('state') == 'COMPLETED':
                    last_family = item.get('family')
                    break
        for task in (self.w.data.get('teamOur') or {}).get('playerTasks') or []:
            if not isinstance(task, dict):
                continue
            family = str(task.get('taskType', ''))
            if ledger and ledger.evolution_complete():
                continue
            if ledger and ledger.family_exhausted(family, task):
                continue
            cooldown = int(task.get('coldDownRounds', 0) or 0)
            valid = bool(task.get('isValid')) and cooldown <= 0
            pending = bool(ledger and ledger.pending_claim_family == family)
            if not valid and not pending:
                continue
            if valid and ledger and ledger.should_defer_new_claim() and not pending:
                continue
            anchor = task.get('taskPosition') or {}
            if not isinstance(anchor, dict) or 'x' not in anchor or 'y' not in anchor:
                continue
            point_xy = (int(anchor['x']), int(anchor['y']))
            if point_xy in self._banned_tasks:
                continue
            cells = self._task_cells(point_xy, side)
            if cells:
                ready.append((task, cells, pending, valid, family != last_family))
        return ready

    def _accept_task(self, role):
        if self.w.data.get('phaseTask'):
            return False
        ledger = getattr(self.reasoning, 'ledger', None)
        if ledger and ledger.evolution_complete():
            return False
        best = None
        for task, cells, pending, valid, alt in self._available_player_tasks():
            route = self._path(role, cells)
            if not route:
                continue
            timeout = max(1, float(task.get('timeoutRounds') or 10))
            home = self._control_point or (pos(self.w.base) if self.w.base else cells[0])
            home_walk = distance(cells[0], home)
            default = min(timeout, float((self.config.get('tasks') or {}).get('solve_estimate', 6)))
            learner = getattr(self.reasoning, 'learning', None)
            estimate = learner.estimate(str(task.get('taskType', '')), default=default) if learner else default
            if not self.w.night and self.w.remaining_day < len(route) + estimate + home_walk:
                continue
            reward = float(task.get('scoreReward', 0) or 0)
            key = (0 if alt else 1, -reward / timeout, 0 if pending else 1, 0 if valid else 1, len(route))
            if best is None or key < best[0]:
                best = (key, cells, route, task, pending, valid)
        if not best:
            return False
        _, cells, route, task, pending, valid = best
        if len(route) == 1:
            if pending or not valid:
                return True
            if ledger:
                ledger.note_claim_attempt(str(task.get('taskType', '')))
            self._pending_accept = cells[0] if cells else None
            self.commands[str(role['id'])] = command('acceptTask')
            return True
        return self._go(role, cells)

    def _task_cells(self, anchor, side):
        kind = next((zone.get('neutralType') for zone in self.w.zones if pos(zone) == anchor), None)
        if kind:
            if not str(kind).startswith(side + 'TaskPoint'):
                return []
            return self._zones(kind)
        own = [pos(zone) for zone in self.w.zones
               if str(zone.get('neutralType', '')).startswith(side + 'TaskPoint')
               and distance(pos(zone), anchor) <= 1]
        return own or [anchor]

    def _treasure_ready(self):
        treasure = getattr(self.reasoning, 'treasure', None)
        if not self.config.get('enable_treasure', True) or not treasure:
            return None
        if treasure.get('ready') is not True:
            return None
        now = int(self.w.data['roundNo'])
        if now < int(treasure.get('open_round', 0) or 0):
            return None
        if now > treasure.get('close_round', 10 ** 9):
            return None
        return treasure

    def _treasure(self, role):
        if self._available_player_tasks():
            return False
        treasure = self._treasure_ready()
        if not treasure:
            return False
        p = (treasure['pos']['x'], treasure['pos']['y'])
        items = tuple(sorted(treasure['items']))
        identity = (p, items, treasure['open_round'])
        now = int(self.w.data['roundNo'])
        if identity in self.treasure_attempted or now > treasure.get('close_round', 10 ** 9):
            return False
        inventory = Counter(role.get('backpack', []))
        required = Counter(items)
        missing = required - inventory
        route = self._path(role, [p])
        if not route:
            return False
        earliest = max(now + len(route) + len(missing) - 1, treasure['open_round'])
        if earliest > treasure.get('close_round', 10 ** 9):
            return False
        if missing:
            if self.w.night or self.w.remaining_day < 25:
                return False
            if len(role.get('backpack', [])) + sum(missing.values()) > role.get('backPackCapability', 40):
                return False
            if any(name not in self.shop for name in missing):
                return False
            total = sum(self.shop[name] * n for name, n in missing.items())
            if self.gold < total + 20:
                return False
            shops = self._zones('weaponShop')
            shop_route = self._path(role, shops)
            if not shop_route:
                return False
            after_shop = dict(role, pos=point(shop_route[-1]))
            if not self._path(after_shop, [p]):
                return False
            if self._adjacent(role, shops):
                name = sorted(missing)[0]
                self.commands[str(role['id'])] = command('buy', name=name, num=missing[name])
                self.gold -= self.shop[name] * missing[name]
                return True
            return self._go(role, shops)
        if now + len(route) < treasure['open_round'] - 5:
            return False
        if len(route) == 1 and now >= treasure['open_round']:
            self.commands[str(role['id'])] = command('summonTreasure', p, item=list(items))
            self.treasure_attempted.add(identity)
            return True
        return self._go(role, [p])
