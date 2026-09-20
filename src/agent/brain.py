"""Official v1.0 contestant policy; no simulator or local model dependency."""
import copy
import logging
from collections import Counter

from .grid import (AutoConstruction, World, best_area_effect,
                   defense_assignments, distance, generate_rocket_positions,
                   generate_wall_positions, plan_attacks, pos,
                   projected_base_damage, ROCKET_TURRET_COUNT,
                   TARGET_WALL_COUNT, WALL_STONE_COST)
from .protocol import Reasoning

LOG = logging.getLogger(__name__)
ORES = ('stone', 'iron', 'copper')
GUNS = ('gatling', 'railgun', 'rocket')
SUMMON_ORDERS = ('SmallRobotSummonOrder', 'MiddleRobotSummonOrder',
                 'LargeRobotSummonOrder', 'BossRobotSummonOrder')
WALL_REPAIR_KIT = 'WallFixer'
WALL_REPAIR_KIT_TARGET_STOCK = 5
BASE_LOW_HEALTH_RATIO = 0.5
WORKER_DANGER_RANGE = 1
SELL_ACTION_COST = 1
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
    return max(0, needed - max(0, int(available_stone or 0)))


def upgrade_group_order(base_health_ratio):
    if base_health_ratio < BASE_LOW_HEALTH_RATIO:
        return (UPGRADE_WEAPON, UPGRADE_STATION, UPGRADE_WALL)
    return (UPGRADE_WEAPON, UPGRADE_WALL, UPGRADE_STATION)


def fixer_restock_count(stock, target=WALL_REPAIR_KIT_TARGET_STOCK):
    return max(0, int(target) - max(0, int(stock or 0)))


def can_complete_same_day(remaining_rounds, flow_cost):
    if remaining_rounds is None or flow_cost is None:
        return False
    return int(remaining_rounds) >= int(flow_cost)


def preferred_ores(stone_deficit, prices, blocked=()):
    blocked = set(blocked or ())
    if int(stone_deficit or 0) > 0 and 'stone' not in blocked:
        return ('stone',)
    valued = [ore for ore in ORES if ore not in blocked]
    return tuple(sorted(valued, key=lambda ore: (-int((prices or {}).get(ore, 0)), ore)))


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
        self._movement_failures = 0
        self._mine_yield = Counter()
        self._was_night = False
        self._current_night_peak = 0.0
        self._last_night_peak = 0.0
        self._current_wave_mix = Counter()
        self._current_wave_ids = set()
        self._last_wave_mix = Counter()
        self._health_caps = {}
        self._summon_day = None
        self._summons_used = 0
        self._wall_slots = []
        self._seen_walls = set()

    def callback(self, data):
        if not isinstance(data, dict) or not isinstance(data.get('teamOur'), dict):
            return empty_response()
        team = data['teamOur']
        identity = (team.get('teamId'), team.get('type'))
        round_no = int(data.get('roundNo', 1))
        if identity != self._identity or (self._round is not None and round_no < self._round):
            self.reasoning = Reasoning(self.config)
            self.auto_construction = AutoConstruction(self.config)
            self.failed_cells = {}
            self.treasure_attempted = set()
            self._previous = {}
            self._movement_failures = 0
            self._mine_yield = Counter()
            self._was_night = False
            self._current_night_peak = 0.0
            self._last_night_peak = 0.0
            self._current_wave_mix = Counter()
            self._current_wave_ids = set()
            self._last_wave_mix = Counter()
            self._health_caps = {}
            self._summon_day = None
            self._summons_used = 0
            self._wall_slots = []
            self._seen_walls = set()
            self._cached = None
            self._round = None
            self._identity = identity
        if round_no == self._round and self._cached is not None:
            return copy.deepcopy(self._cached)
        self.auto_construction.observe(data, self._previous, self._round)
        # Feedback is one turn late. Do not blame stale results after skipped turns.
        if self._round is not None and round_no == self._round + 1:
            for rid, result in (data.get('lastRoundRoleActionResults') or {}).items():
                old = self._previous.get(str(rid), {})
                if result is False and old.get('action') in ('collect', 'build'):
                    p = old.get('targetPos', [{}])[0]
                    if 'x' in p and 'y' in p:
                        self.failed_cells[(p['x'], p['y'])] = round_no + 4
                if old.get('action') == 'move':
                    if result is False:
                        self._movement_failures = min(8, self._movement_failures + 2)
                    elif result is True:
                        self._movement_failures = max(0, self._movement_failures - 1)
                if result is True and old.get('action') == 'collect':
                    targets = old.get('targetPos') or []
                    if len(targets) == 1:
                        target = pos(targets[0])
                        ore = next((zone.get('neutralType') for zone in
                                    (data.get('mapInfo') or {}).get('zones', [])
                                    if pos(zone) == target), None)
                        if ore in ORES:
                            self._mine_yield[(target, ore)] += 1
        self.failed_cells = {p: expiry for p, expiry in self.failed_cells.items() if expiry > round_no}
        response = self._decide(data)
        self._round, self._cached = round_no, copy.deepcopy(response)
        self._previous = copy.deepcopy(response['roleCommandMap'])
        return response

    def _decide(self, data):
        config = dict(self.config)
        # An observed round 0 establishes a zero-based match.
        if data.get('roundNo') == 0:
            self.config['round_origin'] = config['round_origin'] = 0
            self.reasoning.config['round_origin'] = 0
        w = World(data, config)
        self.w = w
        self._observe_world()
        self.commands, self.reserved, self.build_jobs = {}, set(), set()
        self.area_item_used = False
        self._wall_plan = None
        self.gold = int(data['teamOur'].get('goldNum', 0))
        self.prices = {x['name']: max(0, int(x['price'])) for x in data.get('vendorShopList', []) if isinstance(x, dict) and 'name' in x and 'price' in x}
        self.shop = {x['name']: max(0, int(x['price'])) for x in data.get('weaponShopList', []) if isinstance(x, dict) and 'name' in x and 'price' in x}
        pioneers = [r for r in w.people if r['roleType'] == 'pioneer']
        pioneer = pioneers[0] if pioneers else None
        active = bool(data.get('phaseTask'))
        self.safety_ratio = self._safety_ratio()
        self.defense_mode = ('critical' if self.safety_ratio < 1.2 else
                             'defensive' if self.safety_ratio < 1.8 else 'reward')
        self._refresh_layout()
        thought = self.reasoning.update(data, w.day, str(pioneer['id']) if pioneer else None)
        response = {'roleCommandMap': self.commands, 'prompt': thought.get('prompt', ''), 'executeCmd': thought.get('executeCmd', '')}
        if active and pioneer and isinstance(thought.get('taskAnswer'), str):
            self.commands[str(pioneer['id'])] = command('submitAnswer', taskAnswer=thought['taskAnswer'])
        available = [r for r in w.people if not (active and r['roleType'] == 'pioneer')]
        pioneers_ready = [r for r in available if r['roleType'] == 'pioneer']
        assignments = defense_assignments(w, pioneers_ready)
        LOG.info(
            "round=%s night=%s cycle_left=%s walls=%s/%s stone_gap=%s fixer=%s/%s base_hp=%.3f upgrades=%s",
            data.get('roundNo'), w.night, w.remaining_cycle,
            self._effective_walls(), TARGET_WALL_COUNT, self._stone_deficit(),
            self._fixer_stock(), WALL_REPAIR_KIT_TARGET_STOCK,
            self._base_health_ratio(), upgrade_group_order(self._base_health_ratio()))
        # Medicines and upgrades consume a controller action; remove that controller from firing.
        for role in available:
            self._use_emergency(role)
            if str(role['id']) not in self.commands:
                self._use_area_item(role)
            if str(role['id']) not in self.commands:
                self._use_adjacent(role)
            if str(role['id']) not in self.commands:
                self._use_summon(role)
        if w.night:
            firing = {}
            for role in pioneers_ready:
                rid = str(role['id'])
                if rid in self.commands:
                    continue
                gun = self._rotate_gun(role) or assignments.get(rid)
                if gun:
                    firing[rid] = gun
            attacks, controllers = plan_attacks(w, firing)
            self.commands.update(attacks)
            for role in available:
                rid = str(role['id'])
                if rid in self.commands or rid in controllers:
                    continue
                if role['roleType'] == 'pioneer':
                    gun = firing.get(rid) or self._rotate_gun(role)
                    if gun:
                        self._go(role, [pos(gun)])
                    elif not self._task_or_treasure(role, allow_new_task=False):
                        self._shelter(role)
                else:
                    self._night_worker(role)
            return response
        for role in sorted(available, key=lambda r: (r['roleType'] != 'pioneer', r['id'])):
            rid = str(role['id'])
            if rid in self.commands:
                continue
            if role['roleType'] == 'pioneer':
                gun = assignments.get(rid)
                home_path = self._path(role, [pos(gun)]) if gun else None
                return_steps = len(home_path) - 1 if home_path else 0
                if gun and w.remaining_day <= return_steps + self._return_margin():
                    self._go(role, [pos(gun)])
                    continue
                if not self._task_or_treasure(role, allow_new_task=True):
                    if not self._shopping(role):
                        self._go(role, [pos(gun)]) if gun else self._shelter(role)
            else:
                if self._build(role):
                    continue
                if self._stone_deficit() > 0 and self._gather(role):
                    continue
                if self._mine(role):
                    continue
                if self._sell(role) or self._shopping(role):
                    continue
        return response

    def _observe_world(self):
        if self._was_night and not self.w.night:
            self._last_night_peak = self._current_night_peak
            self._last_wave_mix = Counter(self._current_wave_mix)
            self._current_night_peak = 0.0
            self._current_wave_mix = Counter()
            self._current_wave_ids = set()
        if self.w.night:
            self._current_night_peak = max(
                self._current_night_peak, projected_base_damage(self.w, 15))
            team = self.w.data['teamOur'].get('type')
            for robot in self.w.robots:
                target = robot.get('targetTeam')
                if target not in (team, None, ''):
                    continue
                robot_id = str(robot.get('id'))
                if robot_id not in self._current_wave_ids:
                    self._current_wave_ids.add(robot_id)
                    self._current_wave_mix[robot.get('roleType')] += 1
        self._was_night = self.w.night
        if self._summon_day != self.w.day:
            self._summon_day, self._summons_used = self.w.day, 0
        for building in self.w.ours:
            if building.get('roleType') not in ('station', 'wall') + GUNS:
                continue
            key = (building.get('roleType'), int(building.get('level', 1)))
            baseline = 1500 if building.get('roleType') == 'station' else 1000
            self._health_caps[key] = max(
                baseline, self._health_caps.get(key, 0), int(building.get('health', 0)))
        visible_mines = {(pos(zone), zone.get('neutralType')) for zone in self.w.zones
                         if zone.get('neutralType') in ORES}
        self._mine_yield = Counter({key: value for key, value in self._mine_yield.items()
                                    if key in visible_mines})
        for building in self.w.ours:
            if building.get('roleType') == 'wall':
                self._seen_walls.add(pos(building))

    def _safety_ratio(self):
        if not self.w.base:
            return 0.0
        if self.w.night:
            risk = projected_base_damage(self.w, 15)
        else:
            risk = self._last_night_peak * 1.25
        # Before the first observed wave, keep a modest reserve without
        # pretending to know the future spawn table.
        risk = max(200.0, risk)
        return int(self.w.base.get('health', 0)) / risk

    def _return_margin(self):
        base = max(1, int(self.config.get('return_margin', 4)))
        return base + min(4, self._movement_failures)

    def _remaining_cycle(self):
        return int(getattr(self.w, 'remaining_cycle', self.w.remaining_day))

    def _refresh_layout(self):
        if not self.w.base:
            self._rocket_cells = []
            self._wall_plan = []
            return
        gun_cells = {pos(gun) for gun in self.w.guns}
        rocket_blocked = (self.w.blocked | set(self.failed_cells)) - gun_cells
        self._rocket_cells = generate_rocket_positions(
            self.w.base, self.w.width, self.w.height, rocket_blocked)
        our_walls = {pos(role) for role in self.w.ours if role.get('roleType') == 'wall'}
        wall_blocked = (self.w.blocked | set(self.failed_cells)) - our_walls
        if len(self._wall_slots) == TARGET_WALL_COUNT:
            plan = list(self._wall_slots)
        else:
            plan = generate_wall_positions(
                self.w.base, self.w.width, self.w.height, wall_blocked, self._wall_slots)
            if plan:
                self._wall_slots = list(plan)
        self._wall_plan = list(self._wall_slots or plan)

    def _effective_walls(self):
        slots = set(self._wall_slots or self._wall_plan or ())
        if not slots:
            return sum(role.get('roleType') == 'wall' for role in self.w.ours)
        return sum(role.get('roleType') == 'wall' and pos(role) in slots
                   for role in self.w.ours)

    def _pending_walls(self):
        return sum(c.get('action') == 'build' and c.get('name') == 'wall'
                   for c in self.commands.values())

    def _team_stones(self):
        return sum(Counter(role.get('backpack', []))['stone'] for role in self.w.people)

    def _stone_deficit(self):
        pending = self._pending_walls()
        gap = wall_stone_deficit(
            self._effective_walls() + pending, self._team_stones())
        return gap

    def _stone_keep_for(self, role):
        reserve = max(0, TARGET_WALL_COUNT - self._effective_walls() - self._pending_walls())
        others = self._team_stones() - Counter(role.get('backpack', []))['stone']
        return max(0, reserve - max(0, others))

    def _fixer_stock(self):
        return sum(Counter(role.get('backpack', []))[WALL_REPAIR_KIT]
                   for role in self.w.people)

    def _base_health_ratio(self):
        if not self.w.base:
            return 0.0
        level = int(self.w.base.get('level', 1))
        cap = max(1, self._health_caps.get(('station', level), 1500))
        return int(self.w.base.get('health', 0)) / cap

    def _rotate_gun(self, role):
        guns = [gun for gun in self.w.guns if gun.get('roleType') == 'rocket'] or list(self.w.guns)
        if not guns:
            return None
        ready = [gun for gun in guns if int(gun.get('cooldown', 0)) <= 0]
        adjacent = [gun for gun in ready if distance(role, gun) == 1]
        if adjacent:
            return min(adjacent, key=lambda gun: str(gun['id']))
        if ready:
            return min(ready, key=lambda gun: (distance(role, gun), str(gun['id'])))
        return min(guns, key=lambda gun: (int(gun.get('cooldown', 0)),
                                          distance(role, gun), str(gun['id'])))

    def _worker_direct_danger(self, role):
        if int(role.get('health', 0)) <= 80:
            return True
        return any(distance(role, robot) <= WORKER_DANGER_RANGE for robot in self.w.robots)

    def _night_worker(self, role):
        if self._worker_direct_danger(role):
            LOG.info("worker %s interrupt mining: direct danger", role.get('id'))
            return self._shelter(role)
        if self._stone_deficit() > 0 and self._gather(role):
            return True
        if self._mine(role):
            return True
        if self._sell(role):
            return True
        LOG.info("worker %s night collect skipped: no reachable mine", role.get('id'))
        return False

    def _sell_turns(self, role):
        vendors = self._zones('vendor')
        if not vendors:
            return None
        path = self._path(role, vendors)
        if not path:
            return None
        extra = min(2, self._movement_failures) if len(path) > 1 else 0
        return len(path) + extra

    def _can_complete_sell(self, role):
        cost = self._sell_turns(role)
        ok = can_complete_same_day(self._remaining_cycle(), cost)
        if not ok:
            LOG.info("sell deferred: cycle_left=%s cost=%s role=%s",
                     self._remaining_cycle(), cost, role.get('id'))
        return ok

    def _collect_sell_flow(self, role, mine_path, remaining_collect):
        if not mine_path:
            return None
        arrive = dict(role, pos=point(mine_path[-1]))
        sell_cost = self._sell_turns(arrive)
        if sell_cost is None:
            return None
        travel = max(0, len(mine_path) - 1)
        return travel + max(0, int(remaining_collect)) + sell_cost

    def _sellable_ores(self, role, keep=None):
        counts = Counter(role.get('backpack', [])) - Counter(keep or {})
        keep_stone = self._stone_keep_for(role)
        if keep_stone:
            counts['stone'] = max(0, counts['stone'] - keep_stone)
        return {ore: counts[ore] for ore in ORES if counts[ore] and self.prices.get(ore, 0) > 0}

    def _path(self, role, targets):
        return self.w.path(role, self.w.interaction_cells(targets), self.reserved)

    def _go(self, role, targets):
        path = self._path(role, targets)
        if not path:
            return False
        if len(path) > 1:
            dest = path[1]
            self.commands[str(role['id'])] = command('move', dest)
            self.reserved.add(dest)
        return True

    def _adjacent(self, role, targets):
        return any(distance(pos(role), target) == 1 for target in targets)

    def _zones(self, kind):
        return [pos(z) for z in self.w.zones if z.get('neutralType') == kind]

    def _use_adjacent(self, role):
        bag = role.get('backpack', [])
        rid = str(role['id'])
        for building in sorted(self.w.ours, key=lambda b: (b.get('roleType') != 'station', b['id'])):
            prefix = {'station': 'Station', 'wall': 'Wall', 'gatling': 'Weapon', 'rocket': 'Weapon', 'railgun': 'Weapon'}.get(building.get('roleType'))
            level = int(building.get('level', 1))
            voucher = '%sUpgradeVoucher%d' % (prefix, level)
            cap = self._health_caps.get((building.get('roleType'), level),
                                        1500 if building.get('roleType') == 'station' else 1000)
            damaged = int(building.get('health', 0)) < cap * 0.75
            useful_now = (building.get('roleType') in GUNS or damaged or
                          self.defense_mode != 'reward' or self.w.day >= 8)
            if (prefix and level < 3 and voucher in bag and useful_now
                    and self._adjacent(role, [pos(building)])):
                self.commands[rid] = command('use', pos(building), name=voucher)
                return True
        fixer = next((x for x in bag if x == 'WallFixer'), None)
        if fixer:
            walls = []
            for wall in self.w.ours:
                if wall.get('roleType') != 'wall' or not self._adjacent(role, [pos(wall)]):
                    continue
                level = int(wall.get('level', 1))
                cap = self._health_caps.get(('wall', level), 1000)
                if int(wall.get('health', 0)) < cap * 0.55:
                    walls.append(wall)
            if walls:
                target = min(walls, key=lambda wall: (wall.get('health', 0), str(wall['id'])))
                self.commands[rid] = command('use', pos(target), name=fixer)
                return True
        return False

    def _use_emergency(self, role):
        bag = role.get('backpack', [])
        rid = str(role['id'])
        max_health = 200 if role['roleType'] == 'pioneer' else 220
        medicine = next((item for item in bag if item.lower() == 'medicine'), None)
        medicine_limit = 0.7 if self.defense_mode == 'critical' else 0.5
        if medicine and role.get('health', 0) < max_health * medicine_limit:
            self.commands[rid] = command('use', name=medicine)
            return True
        if self.defense_mode != 'critical' or 'WallFixer' not in bag:
            return False
        walls = []
        for wall in self.w.ours:
            if wall.get('roleType') != 'wall' or not self._adjacent(role, [pos(wall)]):
                continue
            level = int(wall.get('level', 1))
            cap = self._health_caps.get(('wall', level), 1000)
            if int(wall.get('health', 0)) < cap * 0.55:
                walls.append(wall)
        if not walls:
            return False
        target = min(walls, key=lambda wall: (wall.get('health', 0), str(wall['id'])))
        self.commands[rid] = command('use', pos(target), name='WallFixer')
        return True

    def _use_area_item(self, role):
        if not self.w.night or self.area_item_used or self.defense_mode == 'reward':
            return False
        bag = role.get('backpack', [])
        options = []
        for name, threshold in (('Bomb', 180), ('DizzyWeapon', 100)):
            if name not in bag:
                continue
            target, value, affected = best_area_effect(self.w, name)
            if target is not None and affected and value >= threshold:
                options.append((value, name, target))
        if not options:
            return False
        _, name, target = max(options, key=lambda entry: (entry[0], entry[1]))
        self.commands[str(role['id'])] = command('use', target, name=name)
        self.area_item_used = True
        return True

    def _offense_ready(self):
        cfg = self.config.get('offense') or {}
        if not cfg.get('enabled', True) or self.w.night:
            return False
        start_day = max(1, min(10, int(cfg.get('start_day', 8))))
        enemy_base = next((role for role in self.w.enemy
                           if role.get('roleType') == 'station'), None)
        threshold = int(cfg.get('enemy_base_health_threshold', 900))
        return (self.w.day >= start_day and self.safety_ratio >= 1.8
                and len(self.w.guns) == ROCKET_TURRET_COUNT and enemy_base is not None
                and int(enemy_base.get('health', 10 ** 9)) <= threshold)

    def _use_summon(self, role):
        if not self._offense_ready() or self._summons_used >= 10:
            return False
        name = next((item for item in SUMMON_ORDERS if item in role.get('backpack', [])), None)
        if not name:
            return False
        self.commands[str(role['id'])] = command('use', name=name)
        self._summons_used += 1
        return True

    def _selling_open(self):
        return True

    def _stone_quota(self, role):
        capacity = int(role.get('backPackCapability', 100))
        have = Counter(role.get('backpack', []))['stone']
        return max(0, min(capacity, have + self._stone_deficit()))

    def _gather(self, role):
        """Fill only the stone gap required for walls or rebuilds."""
        capacity = int(role.get('backPackCapability', 100))
        bag = role.get('backpack', [])
        if len(bag) >= capacity or self._stone_deficit() <= 0:
            return False
        blocked = getattr(self.reasoning, 'blocked_ores', set())
        if 'stone' in blocked:
            return False
        return self._collect_ore(role, 'stone')

    def _collect_ore(self, role, ore, radius=None):
        routes = []
        for zone in self.w.zones:
            if zone.get('neutralType') != ore:
                continue
            target = pos(zone)
            if target in self.failed_cells:
                continue
            if radius is not None and distance(pos(role), target) > radius:
                continue
            path = self._path(role, [target])
            if path:
                routes.append((len(path), target, path))
        if not routes:
            return False
        _, target, path = min(routes)
        if len(path) == 1:
            self.commands[str(role['id'])] = command('collect', target)
            return True
        return self._go(role, [target])

    def _sell(self, role, force=False, keep=None):
        if role.get('roleType') == 'worker' and not self._selling_open():
            return False
        vendors = self._zones('vendor')
        ore_counts = self._sellable_ores(role, keep)
        if not ore_counts or not vendors:
            return False
        if self._adjacent(role, vendors):
            ore = max(ore_counts, key=lambda x: (ore_counts[x] * self.prices[x], x))
            self.commands[str(role['id'])] = command('sell', name=ore, num=ore_counts[ore])
            LOG.info("sell %s x%s keep_stone=%s", ore, ore_counts[ore], self._stone_keep_for(role))
            return True
        if not self._can_complete_sell(role):
            return False
        bag_size = len(role.get('backpack', []))
        market_value = sum(ore_counts[ore] * self.prices[ore] for ore in ore_counts)
        target_value = int((self.config.get('economy') or {}).get('sell_value_threshold', 20))
        if len(self.w.guns) == 2 and self.w.day == 1:
            target_value = max(target_value, 25)
        if force or market_value >= target_value or bag_size >= min(20, role.get('backPackCapability', 100)):
            LOG.info("go vendor value=%s cycle=%s cost=%s",
                     market_value, self._remaining_cycle(), self._sell_turns(role))
            return self._go(role, vendors)
        return False

    def _mine(self, role, return_steps=0):
        capacity = int(role.get('backPackCapability', 100))
        bag = role.get('backpack', [])
        if len(bag) >= capacity:
            return self._sell(role, True)
        blocked = getattr(self.reasoning, 'blocked_ores', set())
        deficit = self._stone_deficit()
        ores = preferred_ores(deficit, self.prices, blocked)
        candidates = self._ore_candidates(role, ores)
        if not candidates and deficit > 0:
            ores = preferred_ores(0, self.prices, blocked)
            candidates = self._ore_candidates(role, ores)
        if not candidates:
            return self._sell(role, True)
        _, _, ore, target, path = max(candidates)
        remaining_collect = min(
            capacity - len(bag),
            max(1, 10 - self._mine_yield[(target, ore)]),
        )
        if deficit > 0 and ore == 'stone':
            remaining_collect = min(remaining_collect, deficit)
        flow = self._collect_sell_flow(role, path, remaining_collect)
        cycle = self._remaining_cycle()
        LOG.info("mine target=%s ore=%s flow=%s cycle=%s stone_gap=%s",
                 target, ore, flow, cycle, deficit)
        sellable = self._sellable_ores(role)
        value = sum(sellable[item] * self.prices[item] for item in sellable)
        threshold = int((self.config.get('economy') or {}).get('sell_value_threshold', 20))
        if (len(path) > 1 and sellable and value >= threshold
                and not can_complete_same_day(cycle, flow)
                and self._can_complete_sell(role)):
            return self._sell(role)
        if len(path) == 1:
            self.commands[str(role['id'])] = command('collect', target)
            return True
        return self._go(role, [target])

    def _ore_candidates(self, role, ores):
        vendors = self._zones('vendor')
        candidates = []
        for zone in self.w.zones:
            ore = zone.get('neutralType')
            target = pos(zone)
            if ore not in ores or target in self.failed_cells:
                continue
            path = self._path(role, [target])
            if not path:
                continue
            travel = len(path) - 1
            vendor_travel = min((distance(target, v) for v in vendors), default=20)
            value = self.prices.get(ore, 0)
            estimated_left = max(1, 10 - self._mine_yield[(target, ore)])
            utility = estimated_left * value / (travel + estimated_left + vendor_travel + 1)
            if ore == 'stone' and self._stone_deficit() > 0:
                utility += 1000
            candidates.append((utility, -travel, ore, target, path))
        return candidates

    def _construction(self):
        return self.config.get('construction', {})

    def _build(self, role):
        cfg = self._construction()
        mode = cfg.get('mode', 'auto')
        if mode == 'off' or self.w.night or role['roleType'] != 'worker':
            return False
        if mode != 'auto':
            return False
        pending = sum(c.get('action') == 'build' and c.get('name') in GUNS for c in self.commands.values())
        if len(self.w.guns) + pending >= ROCKET_TURRET_COUNT:
            if len(self.w.guns) == ROCKET_TURRET_COUNT and self._reconfigure_weapon(role):
                return True
            return self._build_wall(role)
        reserve = max(0, int((self.config.get('economy') or {}).get('opening_cash_reserve', 0)))
        opening_deadline = max(
            12, distance(role, self.w.base) + self._return_margin() + 6)
        if (self.w.day == 1 and len(self.w.guns) + pending == 2 and
                self.w.remaining_day > opening_deadline and self.gold < 25 + reserve):
            return False
        action = self.auto_construction.propose(
            self.w, role, self.reserved, self.gold, getattr(self, '_rocket_cells', None))
        if action is None:
            return False
        self.commands[str(role['id'])] = action
        target = pos(action['targetPos'][0])
        self.reserved.add(target)
        if action['action'] == 'build':
            # All three weapons cost 25 gold in the supplied building table.
            self.gold -= 25
            self.build_jobs.add(target)
        return True

    def _desired_weapon_plan(self):
        cfg = self._construction()
        fallback = [kind for kind in cfg.get(
            'weapon_plan', ('rocket', 'rocket', 'rocket')) if kind in GUNS][:ROCKET_TURRET_COUNT]
        if len(fallback) != ROCKET_TURRET_COUNT or not cfg.get('adaptive_weapons', False):
            return fallback
        swarm = self._last_wave_mix['smallRobot'] + self._last_wave_mix['middleRobot']
        heavy = self._last_wave_mix['largeRobot'] + self._last_wave_mix['bossRobot']
        if swarm >= max(6, heavy * 3):
            return ['rocket', 'gatling', 'gatling']
        if heavy >= 2 and heavy * 2 >= max(1, swarm):
            return ['rocket', 'railgun', 'railgun']
        return fallback

    def _defense_reserve(self):
        missing = max(0, ROCKET_TURRET_COUNT - len(self.w.guns)) * 25
        buffer = 20 if self.defense_mode != 'reward' else 10
        if self._effective_walls() and fixer_restock_count(self._fixer_stock()):
            buffer += int(self.shop.get(WALL_REPAIR_KIT, 10))
        return missing + buffer

    def _reconfigure_weapon(self, role):
        if self.w.day < 3 or self.w.remaining_day <= self._return_margin() + 5:
            return False
        if self.gold < 25 + self._defense_reserve():
            return False
        action = self.auto_construction.propose_replacement(
            self.w, role, self.reserved, self.gold, self._desired_weapon_plan())
        if action is None:
            return False
        self.commands[str(role['id'])] = action
        target = pos(action['targetPos'][0])
        self.reserved.add(target)
        if action['action'] == 'build':
            self.gold -= 25
            self.build_jobs.add(target)
        return True

    def _build_wall(self, role):
        if not self.w.base:
            return False
        plan = self._wall_priority()[:self._wall_target()]
        current = {pos(item) for item in self.w.ours if item.get('roleType') == 'wall'}
        pending = self._pending_walls()
        if self._effective_walls() + pending >= len(plan) and plan:
            return False
        blocked = self.w.blocked | self.reserved | set(self.failed_cells)
        missing = [cell for cell in plan if cell not in current]
        destroyed = [cell for cell in missing if cell in self._seen_walls]
        ordered = destroyed + [cell for cell in missing if cell not in self._seen_walls]
        routes = []
        for index, p in enumerate(ordered):
            if p in blocked and p not in current:
                continue
            path = self._path(role, [p])
            if path and (self.w.night or len(path) + 2 < self.w.remaining_day):
                routes.append((0 if p in self._seen_walls else 1, index, len(path), p))
        if not routes:
            LOG.info("wall build skipped: no reachable missing cell")
            return False
        stones = Counter(role.get('backpack', []))['stone']
        if stones < 1 and self._gather(role):
            return True
        if not stones:
            return False
        _, _, length, p = min(routes)
        if length == 1:
            self.commands[str(role['id'])] = command('build', p, name='wall')
            self.reserved.add(p)
            self.build_jobs.add(p)
            return True
        return self._go(role, [p])

    def _wall_priority(self):
        if self._wall_plan is None:
            self._refresh_layout()
        return list(self._wall_plan or self._wall_slots or [])

    def _wall_target(self):
        plan = self._wall_priority()
        configured = max(0, int(self._construction().get('wall_target', TARGET_WALL_COUNT)))
        return min(TARGET_WALL_COUNT, configured, max(0, len(plan)))

    def _required_wall_cells(self):
        return set(self._wall_priority()[:self._wall_target()])

    def _perimeter_walls(self):
        return self._effective_walls()

    def _perimeter_ready(self):
        return self._perimeter_walls() >= self._wall_target()

    def _shopping(self, role):
        # Deliver carried upgrades before buying more. Any role can use a voucher.
        bag = Counter(role.get('backpack', []))
        deliver_order = (self.w.guns +
                         [wall for wall in self.w.ours if wall.get('roleType') == 'wall'] +
                         ([self.w.base] if self.w.base else []))
        for building in deliver_order:
            prefix = {'station': 'Station', 'wall': 'Wall'}.get(
                building['roleType'], 'Weapon')
            level = int(building.get('level', 1))
            name = '%sUpgradeVoucher%d' % (prefix, level)
            if level < 3 and bag[name]:
                cap = self._health_caps.get((building.get('roleType'), level),
                                            1500 if prefix == 'Station' else 1000)
                damaged = int(building.get('health', 0)) < cap * 0.75
                if (prefix == 'Station' and not damaged and self.defense_mode == 'reward'
                        and self.w.day < 8 and self.w.remaining_day > 10):
                    continue
                return self._go(role, [pos(building)])
        # Fund the initial three weapons before buying upgrades.
        if len(self.w.guns) < ROCKET_TURRET_COUNT or self._remaining_cycle() < 25:
            return False
        shops = self._zones('weaponShop')
        if not shops or len(role.get('backpack', [])) >= role.get('backPackCapability', 0):
            return False
        # Building upgrades always outrank consumables.
        if self._buy_upgrade(role, shops):
            return True
        if self._buy_fixer(role, shops):
            return True
        economy = self.config.get('economy') or {}
        opening = max(0, int(economy.get('consumable_start_round', 391)))
        if int(self.w.data.get('roundNo', 1)) < opening:
            return False
        purchase = self._desired_purchase(role)
        if not purchase:
            return False
        name, number = purchase
        cost = self.shop[name] * number
        # Keep saving while any upgrade voucher is still unbought.
        if self.gold < cost + self._defense_reserve() + self._upgrade_backlog():
            return False
        route = self._path(role, shops)
        if not route:
            return False
        if len(route) == 1:
            self.commands[str(role['id'])] = command('buy', name=name, num=number)
            self.gold -= cost
            return True
        return self._go(role, shops)

    def _upgrade_groups(self):
        walls = [building for building in self.w.ours if building.get('roleType') == 'wall']
        mapping = {
            UPGRADE_WEAPON: self.w.guns,
            UPGRADE_WALL: walls,
            UPGRADE_STATION: [self.w.base] if self.w.base else [],
        }
        return [(name, mapping[name]) for name in upgrade_group_order(self._base_health_ratio())]

    def _first_upgrade_in(self, group):
        for building in sorted(group, key=lambda item: (
                int(item.get('level', 1)), int(item.get('health', 0)),
                str(item['id']))):
            if not building:
                continue
            level = int(building.get('level', 1))
            prefix = {'station': 'Station', 'wall': 'Wall'}.get(
                building['roleType'], 'Weapon')
            name = '%sUpgradeVoucher%d' % (prefix, level)
            if level >= 3 or name not in self.shop:
                continue
            existing = sum(Counter(r.get('backpack', []))[name] for r in self.w.people)
            pending = sum(c.get('action') == 'buy' and c.get('name') == name
                          for c in self.commands.values())
            if existing + pending:
                continue
            return name, building
        return None

    def _next_upgrade(self):
        for _, group in self._upgrade_groups():
            found = self._first_upgrade_in(group)
            if found:
                return found
        return None

    def _upgrade_backlog(self):
        target = self._next_upgrade()
        return self.shop.get(target[0], 0) if target else 0

    def _buy_upgrade(self, role, shops):
        for name_group, group in self._upgrade_groups():
            target = self._first_upgrade_in(group)
            if not target:
                continue
            name, building = target
            if self.gold < self.shop[name] + self._defense_reserve():
                LOG.info("upgrade %s unaffordable gold=%s reserve=%s, try next",
                         name, self.gold, self._defense_reserve())
                continue
            route = self._path(role, shops)
            if not route or len(route) + min(distance(s, pos(building)) for s in shops) + 7 >= self._remaining_cycle():
                continue
            LOG.info("buy upgrade %s group=%s base_hp=%.3f",
                     name, name_group, self._base_health_ratio())
            if len(route) == 1:
                self.commands[str(role['id'])] = command('buy', name=name, num=1)
                self.gold -= self.shop[name]
                return True
            return self._go(role, shops)
        return False

    def _buy_fixer(self, role, shops):
        need = fixer_restock_count(self._fixer_stock())
        price = self.shop.get(WALL_REPAIR_KIT)
        if not need or not price:
            return False
        hold = max(0, ROCKET_TURRET_COUNT - len(self.w.guns)) * 25
        if self.defense_mode == 'critical':
            hold += int(self.shop.get(WALL_REPAIR_KIT, 10))
        affordable = max(0, (self.gold - hold) // price)
        space = max(0, int(role.get('backPackCapability', 0)) - len(role.get('backpack', [])))
        number = min(need, affordable, space)
        if number < 1:
            LOG.info("fixer restock deferred stock=%s need=%s gold=%s hold=%s",
                     self._fixer_stock(), need, self.gold, hold)
            return False
        route = self._path(role, shops)
        if not route or not can_complete_same_day(self._remaining_cycle(), len(route)):
            return False
        LOG.info("buy WallFixer x%s stock=%s", number, self._fixer_stock())
        if len(route) == 1:
            self.commands[str(role['id'])] = command('buy', name=WALL_REPAIR_KIT, num=number)
            self.gold -= price * number
            return True
        return self._go(role, shops)

    def _desired_purchase(self, role):
        inventories = Counter(item for person in self.w.people
                              for item in person.get('backpack', []))
        if (role.get('health', 0) < (120 if role['roleType'] == 'pioneer' else 130)
                and not inventories['Medicine'] and 'Medicine' in self.shop):
            return 'Medicine', 1
        damaged_wall = any(
            wall.get('roleType') == 'wall' and int(wall.get('health', 0)) <
            self._health_caps.get(('wall', int(wall.get('level', 1))), 1000) * 0.55
            for wall in self.w.ours)
        if damaged_wall and not inventories[WALL_REPAIR_KIT] and WALL_REPAIR_KIT in self.shop:
            return WALL_REPAIR_KIT, 1
        if (self._last_night_peak >= 180 and not inventories['DizzyWeapon']
                and 'DizzyWeapon' in self.shop):
            return 'DizzyWeapon', 1
        swarm = self._last_wave_mix['smallRobot'] + self._last_wave_mix['middleRobot']
        if (self.defense_mode != 'reward' and swarm >= 6 and not inventories['Bomb']
                and 'Bomb' in self.shop):
            return 'Bomb', 1
        if self._offense_ready() and not any(inventories[name] for name in SUMMON_ORDERS):
            name = 'MiddleRobotSummonOrder'
            if name in self.shop:
                return name, min(3, 10 - self._summons_used)
        return None

    def _task_or_treasure(self, role, allow_new_task):
        if self.defense_mode == 'critical':
            return False
        if self._treasure(role):
            return True
        if not allow_new_task:
            return False
        tasks = []
        side = self.w.data['teamOur'].get('type', '')
        for task in self.w.data['teamOur'].get('playerTasks', []):
            if not task.get('isValid') or task.get('coldDownRounds', 0) > 0:
                continue
            anchor = (task['taskPosition']['x'], task['taskPosition']['y'])
            cells = self._task_cells(anchor, side)
            if not cells:
                continue
            route = self._path(role, cells)
            if not route:
                continue
            # Reserve task time and travel back to defenses before nightfall.
            home = min((distance(anchor, pos(g)) for g in self.w.guns), default=0)
            budget = max(1, int(task.get('timeoutRounds', 15) or 15))
            family = str(task.get('taskType', 'unknown'))
            familiarity = getattr(self.reasoning, 'task_familiarity', lambda _: 0)(family)
            # Walking away only forfeits the task, so budget the rounds the
            # solver actually needs instead of the whole timeout.
            estimate = max(2, int((self.config.get('tasks') or {}).get('solve_estimate', 6)))
            task_rounds = min(budget, 3 if familiarity else estimate)
            margin = self._return_margin() + (4 if self.defense_mode == 'defensive' else 0)
            travel = len(route) - 1
            if travel + task_rounds + home + margin >= self.w.remaining_day:
                continue
            probability = 0.9 if familiarity else 0.65
            reward = max(0, float(task.get('scoreReward', 0)))
            speed_bonus = 5.0 * budget / max(1, task_rounds)
            expected = probability * (reward + speed_bonus)
            expected += (1.0 - probability) * reward * 0.35
            expected += probability * 0.3 * max(0, float(task.get('goldReward', 0)))
            utility = expected / max(1, travel + task_rounds)
            tasks.append((utility, cells, route))
        if not tasks:
            return False
        _, cells, route = max(tasks, key=lambda t: t[0])
        if len(route) == 1:
            self.commands[str(role['id'])] = command('acceptTask')
            return True
        return self._go(role, cells)

    def _task_cells(self, anchor, side):
        """Own task-point cells; task point 2 spans two of them."""
        kind = next((z.get('neutralType') for z in self.w.zones if pos(z) == anchor), None)
        if kind:
            if not str(kind).startswith(side + 'TaskPoint'):
                return []  # Accepting at an enemy task point is void.
            return self._zones(kind)
        own = [pos(z) for z in self.w.zones
               if str(z.get('neutralType', '')).startswith(side + 'TaskPoint')
               and distance(pos(z), anchor) <= 1]
        return own or [anchor]

    def _treasure(self, role):
        treasure = getattr(self.reasoning, 'treasure', None)
        if not self.config.get('enable_treasure', True) or not treasure:
            return False
        p = (treasure['pos']['x'], treasure['pos']['y'])
        items = tuple(sorted(treasure['items']))
        identity = (p, items, treasure['open_round'])
        now = int(self.w.data['roundNo'])
        if identity in self.treasure_attempted or now > treasure.get('close_round', 10 ** 9):
            return False
        # The news module validates quoted evidence; never send a blind probe.
        inventory = Counter(role.get('backpack', []))
        required = Counter(items)
        missing = required - inventory
        route = self._path(role, [p])
        if not route:
            return False
        # A direct route is an optimistic lower bound even when a shop detour
        # is still necessary. Reject impossible deadlines before spending gold.
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
            # Verify the selected shop approach has a reachable altar route.
            after_shop = dict(role, pos=point(shop_route[-1]))
            altar_route = self._path(after_shop, [p])
            if not altar_route:
                return False
            earliest = max(now + len(shop_route) - 1 + len(missing) + len(altar_route) - 1, treasure['open_round'])
            if earliest > treasure.get('close_round', 10 ** 9):
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
            # Mark the attempt immediately: legal failures consume every item.
            self.treasure_attempted.add(identity)
            return True
        return self._go(role, [p])

    def _shelter(self, role):
        if self.w.base:
            return self._go(role, [pos(self.w.base)])
        return False
