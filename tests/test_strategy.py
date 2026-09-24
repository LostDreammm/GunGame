#!/usr/bin/env python3
"""Acceptance tests for the scheme-B layout, night roles, and economy."""
import unittest

from agent.brain import (BASE_LOW_HEALTH_RATIO, MEDICINE_TARGET_STOCK, STONE_KEEP,
                         Strategy, WALL_REPAIR_KIT, WALL_REPAIR_KIT_TARGET_STOCK,
                         can_complete_same_day, fixer_restock_count, fixer_target_stock,
                         preferred_ores, upgrade_group_order, wall_stone_deficit)
from agent.grid import (ROCKET_TURRET_COUNT, TARGET_WALL_COUNT, World,
                        base_center, building_rings, choose_control_point,
                        default_attack_direction, generate_rocket_positions,
                        generate_wall_positions, layout_front, pair_controllers,
                        specified_control_point, specified_rocket_offsets,
                        template_wall_cells, wall_build_order)


MAP_W, MAP_H = 41, 32
TOP_LEFT_BASE = (10, 24)
BOTTOM_RIGHT_BASE = (30, 10)
CONFIG = {
    "round_origin": 1,
    "enable_llm": False,
    "enable_treasure": False,
    "return_margin": 4,
    "economy": {
        "sell_value_threshold": 25,
        "consumable_start_round": 1,
        "opening_cash_reserve": 0,
    },
    "construction": {
        "mode": "auto",
        "wall_target": 14,
        "adaptive_weapons": False,
        "weapon_plan": ["rocket", "rocket", "rocket"],
    },
}


def station(x, y, health=1500, level=1, ident=1):
    return {
        "id": ident, "pos": {"x": x, "y": y}, "roleType": "station",
        "health": health, "level": level, "attackPower": 0, "attackRange": 0,
        "backPackCapability": 0, "backpack": [],
    }


def role(ident, x, y, kind, backpack=None, health=None, **extra):
    return {
        "id": ident, "pos": {"x": x, "y": y}, "roleType": kind,
        "health": health if health is not None else (200 if kind == "pioneer" else 220),
        "level": extra.get("level", 1),
        "backPackCapability": extra.get("backPackCapability", 40 if kind == "pioneer" else 100),
        "backpack": list(backpack or []),
        "attackPower": extra.get("attackPower", 0),
        "attackRange": extra.get("attackRange", 0),
        "cooldown": extra.get("cooldown", 0),
    }


def snapshot(base_xy=TOP_LEFT_BASE, round_no=10, roles=None, zones=None,
             robots=None, gold=75, night=False, shops=None, prices=None):
    if night:
        round_no = 80
    sx, sy = base_xy
    people = roles or [
        role(2, sx + 3, sy, "pioneer"),
        role(3, sx + 4, sy - 2, "worker"),
        role(4, sx + 5, sy - 3, "worker"),
    ]
    return {
        "roundNo": round_no,
        "mapInfo": {
            "width": MAP_W,
            "height": MAP_H,
            "zones": zones or [
                {"neutralType": "vendor", "pos": {"x": 20, "y": 16}},
                {"neutralType": "weaponShop", "pos": {"x": 25, "y": 20}},
                {"neutralType": "stone", "pos": {"x": 4, "y": 24}},
                {"neutralType": "iron", "pos": {"x": 25, "y": 8}},
                {"neutralType": "copper", "pos": {"x": 18, "y": 12}},
            ],
        },
        "teamOur": {
            "type": "challenger" if base_xy == TOP_LEFT_BASE else "defender",
            "teamId": "t1",
            "goldNum": gold,
            "roles": [station(*base_xy)] + people,
            "playerTasks": [],
        },
        "teamEnemy": {"roles": []},
        "robot": {"roles": robots or []},
        "vendorShopList": prices or [
            {"name": "stone", "price": 2},
            {"name": "iron", "price": 6},
            {"name": "copper", "price": 8},
        ],
        "weaponShopList": shops or [
            {"name": "WeaponUpgradeVoucher1", "price": 100},
            {"name": "WeaponUpgradeVoucher2", "price": 150},
            {"name": "WallUpgradeVoucher1", "price": 20},
            {"name": "WallUpgradeVoucher2", "price": 30},
            {"name": "StationUpgradeVoucher1", "price": 100},
            {"name": "StationUpgradeVoucher2", "price": 150},
            {"name": WALL_REPAIR_KIT, "price": 10},
            {"name": "Medicine", "price": 10},
        ],
        "lastRoundRoleActionResults": {},
        "errors": [],
    }


def bind(strategy, data, **attrs):
    strategy.w = World(data, CONFIG)
    strategy.commands = {}
    strategy.reserved = set()
    strategy.build_jobs = set()
    strategy.gold = int(data["teamOur"].get("goldNum", 0))
    strategy.prices = {x["name"]: x["price"] for x in data["vendorShopList"]}
    strategy.shop = {x["name"]: x["price"] for x in data["weaponShopList"]}
    strategy._refresh_layout()
    for key, value in attrs.items():
        setattr(strategy, key, value)
    return strategy


def add_self_task(data, x=8, y=20):
    team = data["teamOur"]["type"]
    data["mapInfo"]["zones"].append({
        "neutralType": team + "TaskPoint1", "pos": {"x": x, "y": y},
    })
    data["teamOur"]["playerTasks"] = [{
        "taskType": "自进化类1",
        "taskPosition": {"x": x, "y": y},
        "isValid": True,
        "coldDownRounds": 0,
        "timeoutRounds": 40,
        "scoreReward": 50,
        "goldReward": 10,
    }]
    return data


class LayoutTests(unittest.TestCase):
    def test_specified_rockets_share_one_control_point(self):
        base = station(*TOP_LEFT_BASE)
        cells = generate_rocket_positions(base, MAP_W, MAP_H)
        self.assertEqual(len(cells), ROCKET_TURRET_COUNT)
        blue, _ = building_rings(base, MAP_W, MAP_H)
        self.assertTrue(set(cells) <= blue)
        sx, sy = TOP_LEFT_BASE
        expected = [(sx + dx, sy + dy) for dx, dy in specified_rocket_offsets("top_left")]
        self.assertEqual(cells, expected)
        cp = specified_control_point(base, "top_left")
        self.assertTrue(all(max(abs(cp[0] - x), abs(cp[1] - y)) == 1 for x, y in cells))

    def test_bottom_right_rockets_mirror_inward(self):
        top = generate_rocket_positions(station(*TOP_LEFT_BASE), MAP_W, MAP_H)
        bottom = generate_rocket_positions(station(*BOTTOM_RIGHT_BASE), MAP_W, MAP_H)
        self.assertEqual(len(bottom), ROCKET_TURRET_COUNT)
        self.assertNotEqual(set(top), set(bottom))
        blue, _ = building_rings(station(*BOTTOM_RIGHT_BASE), MAP_W, MAP_H)
        self.assertTrue(set(bottom) <= blue)
        cp = choose_control_point(bottom, station(*BOTTOM_RIGHT_BASE), MAP_W, MAP_H)
        self.assertIsNotNone(cp)
        self.assertTrue(all(max(abs(cp[0] - x), abs(cp[1] - y)) == 1 for x, y in bottom))

    def test_layout_supports_north_and_south_fronts(self):
        base = station(20, 16)
        for direction, expected in (((0, -1), "N"), ((0, 1), "S")):
            self.assertEqual(layout_front(base, MAP_W, MAP_H, direction), expected)
            rockets = generate_rocket_positions(
                base, MAP_W, MAP_H, direction=direction)
            walls, _ = wall_build_order(
                base, MAP_W, MAP_H, direction=direction)
            cp = choose_control_point(rockets, base, MAP_W, MAP_H)
            self.assertEqual(len(set(rockets)), 3)
            self.assertEqual(len(set(walls)), 14)
            self.assertIsNotNone(cp)
            self.assertTrue(all(max(abs(cp[0] - x), abs(cp[1] - y)) == 1
                                for x, y in rockets))

    def test_walls_are_fourteen_cell_c_ring(self):
        for xy in (TOP_LEFT_BASE, BOTTOM_RIGHT_BASE):
            base = station(*xy)
            walls, gate = wall_build_order(base, MAP_W, MAP_H)
            template = template_wall_cells(base, MAP_W, MAP_H)
            _, yellow = building_rings(base, MAP_W, MAP_H)
            self.assertEqual(len(set(walls)), TARGET_WALL_COUNT, xy)
            self.assertEqual(len(set(template)), TARGET_WALL_COUNT, xy)
            self.assertTrue(set(walls) <= yellow)
            self.assertNotIn(gate, walls)

    def test_walls_do_not_overlap_base_rockets_or_obstacles(self):
        base = station(*TOP_LEFT_BASE)
        rockets = generate_rocket_positions(base, MAP_W, MAP_H)
        blocked = set(rockets) | {(13, 24)}
        walls = generate_wall_positions(base, MAP_W, MAP_H, blocked=blocked)
        green = {(10, 24), (11, 24), (10, 23), (11, 23)}
        self.assertTrue(set(walls).isdisjoint(green))
        self.assertTrue(set(walls).isdisjoint(set(rockets)))
        self.assertNotIn((13, 24), walls)
        _, yellow = building_rings(base, MAP_W, MAP_H)
        self.assertTrue(set(walls) <= yellow)

    def test_occupied_or_oob_cells_use_legal_substitutes(self):
        base = station(*TOP_LEFT_BASE)
        walls, gate = wall_build_order(base, MAP_W, MAP_H)
        blocked = set(walls[:3])
        replaced = generate_wall_positions(base, MAP_W, MAP_H, blocked=blocked)
        self.assertTrue(set(walls[:3]).isdisjoint(set(replaced)))
        self.assertNotIn(gate, replaced)

    def test_no_legal_cells_skips_safely(self):
        base = station(*TOP_LEFT_BASE)
        blue, yellow = building_rings(base, MAP_W, MAP_H)
        self.assertEqual(generate_rocket_positions(base, MAP_W, MAP_H, blocked=blue), [])
        self.assertEqual(generate_wall_positions(base, MAP_W, MAP_H, blocked=yellow), [])
        self.assertEqual(generate_rocket_positions(None, MAP_W, MAP_H), [])
        self.assertEqual(generate_wall_positions(None, MAP_W, MAP_H), [])

    def test_opening_stays_on_gun_side(self):
        base = station(*TOP_LEFT_BASE)
        walls, gate = wall_build_order(base, MAP_W, MAP_H)
        again, kept = wall_build_order(base, MAP_W, MAP_H, old_gate=gate)
        self.assertEqual(kept, gate)
        self.assertEqual(set(walls), set(again))
        self.assertEqual(gate[0], TOP_LEFT_BASE[0] - 2)


class NightAndEconomyTests(unittest.TestCase):
    def test_night_only_pioneer_fires(self):
        sx, sy = TOP_LEFT_BASE
        rockets = generate_rocket_positions(station(sx, sy), MAP_W, MAP_H)
        cp = specified_control_point(station(sx, sy), "top_left")
        roles = [
            role(2, cp[0], cp[1], "pioneer"),
            role(3, 4, 20, "worker"),
            role(4, 6, 18, "worker"),
        ]
        guns = [
            role(20 + i, x, y, "rocket", attackRange=10, attackPower=20, cooldown=0)
            for i, (x, y) in enumerate(rockets)
        ]
        data = snapshot(roles=roles, night=True, robots=[
            role(90, rockets[0][0] + 3, rockets[0][1], "smallRobot", health=30,
                 **{"targetTeam": "challenger"}),
            role(91, rockets[1][0] + 2, rockets[1][1], "smallRobot", health=30,
                 **{"targetTeam": "challenger"}),
        ])
        data["teamOur"]["roles"] = [station(sx, sy)] + roles + guns
        result = Strategy(CONFIG).callback(data)
        attacks = [cmd for cmd in result["roleCommandMap"].values() if cmd.get("action") == "attack"]
        self.assertTrue(attacks)
        self.assertEqual({str(cmd["controllerId"]) for cmd in attacks}, {"2"})
        worker_actions = {result["roleCommandMap"].get("3", {}).get("action"),
                          result["roleCommandMap"].get("4", {}).get("action")}
        self.assertNotIn("attack", worker_actions)

    def test_night_task_pioneer_does_not_man_guns(self):
        sx, sy = TOP_LEFT_BASE
        rockets = generate_rocket_positions(station(sx, sy), MAP_W, MAP_H)
        roles = [
            role(2, 4, 4, "pioneer"),
            role(3, 6, 20, "worker"),
            role(4, 6, 18, "worker"),
        ]
        guns = [
            role(20 + i, x, y, "rocket", attackRange=10, attackPower=20, cooldown=0)
            for i, (x, y) in enumerate(rockets)
        ]
        data = snapshot(roles=roles, night=True, robots=[
            role(90, rockets[0][0] + 3, rockets[0][1], "smallRobot", health=30,
                 **{"targetTeam": "challenger"}),
        ])
        data["phaseTask"] = "solve a puzzle"
        data["teamOur"]["roles"] = [station(sx, sy)] + roles + guns
        result = Strategy(CONFIG).callback(data)
        attacks = [cmd for cmd in result["roleCommandMap"].values()
                   if cmd.get("action") == "attack"]
        self.assertFalse(attacks)
        self.assertNotEqual(str((attacks[0] if attacks else {}).get("controllerId")), "2")
        self.assertNotIn(result["roleCommandMap"].get("2", {}).get("action"), ("attack",))

    def test_fixer_restock_stops_at_two_on_day3(self):
        self.assertEqual(fixer_restock_count(0), 2)
        self.assertEqual(fixer_restock_count(1), 1)
        self.assertEqual(fixer_restock_count(2), 0)
        self.assertEqual(fixer_target_stock(1), 0)
        self.assertEqual(fixer_target_stock(3), 2)
        self.assertEqual(fixer_target_stock(4, 3), 3)
        strategy = bind(Strategy(CONFIG), snapshot(gold=80, round_no=270))
        shops = strategy._zones("weaponShop")
        worker = next(item for item in strategy.w.people if strategy._is_repairer(item))
        worker["pos"] = {"x": 24, "y": 20}
        strategy.w.guns = [role(20, 12, 24, "rocket"), role(21, 12, 25, "rocket"),
                           role(22, 11, 25, "rocket")]
        bought = strategy._buy_fixer(worker, shops)
        self.assertTrue(bought)
        cmd = strategy.commands[str(worker["id"])]
        self.assertEqual(cmd["action"], "buy")
        self.assertEqual(cmd["name"], WALL_REPAIR_KIT)
        self.assertEqual(cmd["num"], WALL_REPAIR_KIT_TARGET_STOCK)
        strategy.commands = {}
        worker["backpack"] = [WALL_REPAIR_KIT] * 2
        self.assertFalse(strategy._buy_fixer(worker, shops))

    def test_upgrade_order_weapons_before_walls(self):
        self.assertEqual(upgrade_group_order(0.49), ("weapon", "wall", "station"))
        self.assertEqual(upgrade_group_order(BASE_LOW_HEALTH_RATIO), ("weapon", "wall", "station"))
        data = snapshot(gold=200)
        data["teamOur"]["roles"][0]["health"] = 700
        strategy = bind(Strategy(CONFIG), data)
        strategy._health_caps[("station", 1)] = 1500
        strategy.w.guns = [role(20, 12, 24, "rocket")]
        strategy.w.ours.append(role(30, 13, 24, "wall"))
        name, building = strategy._next_upgrade()
        self.assertEqual(name, "WeaponUpgradeVoucher1")

    def test_repairer_keeps_five_stones_when_selling(self):
        data = snapshot()
        worker = data["teamOur"]["roles"][2]
        worker["backpack"] = ["stone"] * STONE_KEEP
        worker["pos"] = {"x": 19, "y": 16}
        strategy = bind(Strategy(CONFIG), data)
        self.assertTrue(strategy._is_repairer(worker))
        self.assertEqual(strategy._stone_keep_for(worker), STONE_KEEP)
        self.assertFalse(strategy._sell(worker, True))
        worker["backpack"] = ["stone"] * (STONE_KEEP + 3)
        self.assertTrue(strategy._sell(worker, True))
        self.assertEqual(strategy.commands[str(worker["id"])]["num"], 3)

    def test_rebuild_collects_capped_stone_gap(self):
        self.assertEqual(wall_stone_deficit(14, 0), 0)
        self.assertEqual(wall_stone_deficit(10, 0), 4)
        self.assertEqual(wall_stone_deficit(0, 0, target=14), 14)
        data = snapshot()
        strategy = bind(Strategy(CONFIG), data)
        walls, _ = wall_build_order(strategy.w.base, MAP_W, MAP_H)
        strategy._wall_slots = walls
        strategy._wall_plan = walls
        strategy._seen_walls = set(walls)
        built = [role(40 + i, x, y, "wall") for i, (x, y) in enumerate(walls[:-1])]
        strategy.w.ours.extend(built)
        self.assertEqual(strategy._effective_walls(), len(walls) - 1)
        self.assertEqual(strategy._stone_deficit(), 1)
        self.assertEqual(preferred_ores(strategy._stone_deficit(), strategy.prices), ("stone",))

    def test_complete_walls_prefer_high_value_ore(self):
        prices = {"stone": 2, "iron": 6, "copper": 8}
        self.assertEqual(preferred_ores(0, prices, money=True), ("copper", "iron"))
        self.assertEqual(preferred_ores(3, prices), ("stone",))

    def test_shop_run_at_sixty_percent_bag(self):
        data = snapshot()
        data["mapInfo"]["zones"] = [
            {"neutralType": "vendor", "pos": {"x": 38, "y": 2}},
            {"neutralType": "weaponShop", "pos": {"x": 25, "y": 20}},
            {"neutralType": "stone", "pos": {"x": 4, "y": 24}},
        ]
        worker = data["teamOur"]["roles"][3]
        worker["pos"] = {"x": 4, "y": 24}
        worker["backpack"] = ["copper"] * 59
        worker["backPackCapability"] = 100
        strategy = bind(Strategy(CONFIG), data)
        self.assertGreater(strategy._vendor_dist(worker), 20)
        self.assertFalse(strategy._should_sell(worker))
        worker["backpack"] = ["copper"] * 60
        self.assertTrue(strategy._should_sell(worker))

    def test_nearby_vendor_sells_small_value(self):
        data = snapshot()
        worker = data["teamOur"]["roles"][3]
        worker["backpack"] = ["copper"]
        worker["pos"] = {"x": 19, "y": 16}
        strategy = bind(Strategy(CONFIG), data)
        self.assertTrue(strategy._should_sell(worker))

    def test_same_day_helper_still_gates_impossible_trips(self):
        self.assertTrue(can_complete_same_day(12, 12))
        self.assertFalse(can_complete_same_day(11, 12))
        self.assertFalse(can_complete_same_day(None, 3))

    def test_half_health_uses_medicine(self):
        data = snapshot()
        worker = data["teamOur"]["roles"][2]
        worker["health"] = 100
        worker["backpack"] = ["Medicine"]
        strategy = bind(Strategy(CONFIG), data)
        self.assertTrue(strategy._use_emergency(worker))
        self.assertEqual(strategy.commands[str(worker["id"])]["action"], "use")

    def test_early_days_workers_do_not_evening_return(self):
        data = snapshot(round_no=56)
        strategy = bind(Strategy(CONFIG), data)
        worker = strategy.w.people[-1]
        worker["pos"] = {"x": 4, "y": 4}
        self.assertFalse(strategy._handle_repairer_return(worker))
        self.assertNotIn(str(worker["id"]), strategy.commands)

    def test_day4_repairer_returns_before_night(self):
        data = snapshot(round_no=56 + 130 * 3)
        strategy = bind(Strategy(CONFIG), data)
        self.assertGreaterEqual(strategy.w.day, 4)
        repairer = next(item for item in strategy.w.people if strategy._is_repairer(item))
        repairer["pos"] = {"x": 4, "y": 4}
        self.assertTrue(strategy._handle_repairer_return(repairer))
        self.assertEqual(strategy.commands[str(repairer["id"])]["action"], "move")

    def test_do_not_buy_summon_orders(self):
        data = snapshot(gold=400)
        data["weaponShopList"].append({"name": "MiddleRobotSummonOrder", "price": 30})
        strategy = bind(Strategy(CONFIG), data)
        strategy.w.guns = [role(20, 12, 24, "rocket"), role(21, 12, 25, "rocket"),
                           role(22, 11, 25, "rocket")]
        worker = strategy.w.people[-1]
        plan = strategy._purchase_plan(worker)
        if plan:
            self.assertNotIn("Summon", plan[0])

    def test_medicine_target_is_two(self):
        self.assertEqual(MEDICINE_TARGET_STOCK, 2)

    def test_pioneer_does_not_double_accept_before_phase_arrives(self):
        data = add_self_task(snapshot())
        strategy = Strategy(CONFIG)
        accepted = False
        for _ in range(24):
            result = strategy.callback(data)
            cmd = result["roleCommandMap"].get("2", {})
            if cmd.get("action") == "acceptTask":
                accepted = True
                break
            if cmd.get("action") == "move" and cmd.get("targetPos"):
                data["teamOur"]["roles"][1]["pos"] = dict(cmd["targetPos"][0])
            data["roundNo"] = int(data["roundNo"]) + 1
        self.assertTrue(accepted)
        data["roundNo"] = int(data["roundNo"]) + 1
        later = strategy.callback(data)
        self.assertNotEqual(later["roleCommandMap"].get("2", {}).get("action"), "acceptTask")

    def test_pioneer_claims_again_after_previous_task_finishes(self):
        data = add_self_task(snapshot())
        strategy = Strategy(CONFIG)
        accepted = 0
        for _ in range(40):
            result = strategy.callback(data)
            cmd = result["roleCommandMap"].get("2", {})
            if cmd.get("action") == "acceptTask":
                accepted += 1
                if accepted == 1:
                    strategy.reasoning.ledger.pending_claim_family = None
                    if strategy.reasoning.ledger.current():
                        strategy.reasoning.ledger.mark_current_completed()
                    data["roundNo"] = int(data["roundNo"]) + 1
                    continue
                break
            if cmd.get("action") == "move" and cmd.get("targetPos"):
                data["teamOur"]["roles"][1]["pos"] = dict(cmd["targetPos"][0])
            data["roundNo"] = int(data["roundNo"]) + 1
        self.assertGreaterEqual(accepted, 2)

    def test_pioneer_stops_new_tasks_when_cannot_finish_and_return(self):
        data = add_self_task(snapshot(round_no=66))
        result = Strategy(CONFIG).callback(data)
        cmd = result["roleCommandMap"].get("2", {})
        self.assertNotEqual(cmd.get("action"), "acceptTask")

    def test_pioneer_prefers_task_over_treasure(self):
        data = add_self_task(snapshot())
        strategy = Strategy(CONFIG)
        strategy.reasoning.treasure = {
            "pos": {"x": 25, "y": 20}, "items": ["StarSand"],
            "open_round": 1, "close_round": 2000, "ready": True,
        }
        result = strategy.callback(data)
        cmd = result["roleCommandMap"].get("2", {})
        self.assertIn(cmd.get("action"), ("move", "acceptTask"))
        if cmd.get("action") == "move":
            dest = cmd["targetPos"][0]
            start = (13, 24)
            task = (8, 20)
            after = (dest["x"], dest["y"])
            self.assertLess(
                max(abs(after[0] - task[0]), abs(after[1] - task[1])),
                max(abs(start[0] - task[0]), abs(start[1] - task[1])),
            )

    def test_pair_controllers_is_greedy_by_distance(self):
        people = [role(2, 10, 10, "pioneer"), role(3, 20, 20, "worker")]
        guns = [role(20, 11, 10, "rocket"), role(21, 21, 20, "rocket")]
        assigned = pair_controllers(people, guns)
        self.assertEqual(str(assigned["2"]["id"]), "20")
        self.assertEqual(str(assigned["3"]["id"]), "21")

    def test_price_forecast_from_negative_news(self):
        from agent.protocol import Reasoning
        brain = Reasoning({"enable_llm": False, "enable_treasure": False})
        data = snapshot()
        data["worldNews"] = {"officialNews": "铜矿塌方，预计停产2天", "folkLegends": ""}
        brain.update(data, 1, "2")
        self.assertEqual(brain.predicted_price("copper", 8, 1), 8)
        self.assertGreater(brain.predicted_price("copper", 8, 2), 8)
        self.assertTrue(brain.is_spiking("copper", 8, 2))
        self.assertTrue(brain.should_stockpile("copper", 1))

    def test_news_stockpiles_before_shutdown_and_sells_during_it(self):
        from agent.protocol import Reasoning
        brain = Reasoning({"enable_llm": False, "enable_treasure": False})
        data = snapshot()
        data["worldNews"] = {
            "officialNews": "铁矿明日停工，检修两天", "folkLegends": ""}
        brain.update(data, 1, "2")
        self.assertTrue(brain.should_stockpile("iron", 1))
        self.assertEqual(brain.sale_boost("iron", 1), 0)
        self.assertGreater(brain.sale_boost("iron", 2), 0)
        data["worldNews"]["officialNews"] = "铁矿恢复开采"
        brain.update(data, 2, "2")
        self.assertFalse(brain.should_stockpile("iron", 2))
        self.assertEqual(brain.sale_boost("iron", 2), 0)

    def test_treasure_without_ready_never_acts(self):
        strategy = bind(Strategy(CONFIG), snapshot())
        strategy.reasoning.treasure = {
            "pos": {"x": 25, "y": 20}, "items": ["StarSand"],
            "open_round": 1, "close_round": 2000,
        }
        self.assertIsNone(strategy._treasure_ready())

    def test_critical_wall_can_queue_before_all_guns_l2(self):
        data = snapshot(gold=200)
        strategy = bind(Strategy(CONFIG), data)
        strategy._health_caps[("wall", 1)] = 1000
        strategy.w.guns = [role(20, 12, 24, "rocket", level=1, health=1000)]
        cracked = role(30, 13, 24, "wall", health=200)
        solid = role(31, 14, 23, "wall", health=900)
        strategy.w.ours.extend([cracked, solid])
        kinds = [(item["kind"], item["cell"]) for item in strategy._upgrade_agenda()
                 if item["kind"] in ("weapon", "wall")]
        self.assertEqual(kinds[0][0], "weapon")
        self.assertIn(("wall", (13, 24)), kinds)
        self.assertNotIn(("wall", (14, 23)), kinds)

    def test_facing_walls_cap_before_weapon_l3(self):
        data = snapshot(gold=800)
        strategy = bind(Strategy(CONFIG), data)
        strategy._health_caps[("wall", 1)] = 1000
        strategy.w.guns = [
            role(20, 12, 24, "rocket", level=2, health=1000),
            role(21, 13, 24, "rocket", level=2, health=1000),
            role(22, 12, 23, "rocket", level=2, health=1000),
        ]
        walls = [role(40 + i, 10 + i, 20, "wall", health=900) for i in range(8)]
        strategy.w.ours.extend(walls)
        cells = [(10 + i, 20) for i in range(8)]
        strategy._wall_plan = cells
        strategy._wall_registry.cells = {
            cell: {"exists": True, "level": 1, "front": True, "rebuilt_round": None}
            for cell in cells
        }
        agenda = strategy._upgrade_agenda()
        before_l3 = []
        for item in agenda:
            if item["kind"] == "weapon" and item["name"].endswith("2"):
                break
            if item["kind"] == "wall":
                before_l3.append(item["cell"])
        self.assertLessEqual(len(before_l3), 6)
        self.assertTrue(any(item["name"] == "WeaponUpgradeVoucher2" for item in agenda))

    def test_restored_wall_jumps_queue(self):
        data = snapshot(gold=400)
        strategy = bind(Strategy(CONFIG), data)
        strategy._health_caps[("wall", 1)] = 1000
        strategy.w.guns = [
            role(20, 12, 24, "rocket", level=2, health=1000),
            role(21, 13, 24, "rocket", level=2, health=1000),
            role(22, 12, 23, "rocket", level=2, health=1000),
        ]
        rebuilt = role(30, 15, 22, "wall", health=900)
        ordinary = role(31, 16, 22, "wall", health=900)
        strategy.w.ours.extend([rebuilt, ordinary])
        strategy._wall_plan = [(15, 22), (16, 22)]
        strategy._control_point = (12, 24)
        strategy._wall_registry.cells = {
            (15, 22): {"exists": True, "level": 1, "front": True, "rebuilt_round": 80},
            (16, 22): {"exists": True, "level": 1, "front": True, "rebuilt_round": None},
        }
        walls = [item for item in strategy._upgrade_agenda() if item["kind"] == "wall"]
        self.assertEqual(walls[0]["cell"], (15, 22))

    def test_no_wall_l3_while_l1_remains(self):
        data = snapshot(gold=400)
        strategy = bind(Strategy(CONFIG), data)
        strategy._health_caps[("wall", 1)] = 1000
        strategy._health_caps[("wall", 2)] = 1000
        strategy.w.guns = [
            role(20, 12, 24, "rocket", level=3, health=1000),
            role(21, 13, 24, "rocket", level=3, health=1000),
            role(22, 12, 23, "rocket", level=3, health=1000),
        ]
        low = role(30, 15, 22, "wall", health=900)
        high = role(31, 16, 22, "wall", health=900, level=2)
        strategy.w.ours.extend([low, high])
        names = [item["name"] for item in strategy._upgrade_agenda() if item["kind"] == "wall"]
        self.assertNotIn("WallUpgradeVoucher2", names)
        self.assertIn("WallUpgradeVoucher1", names)

    def test_station_ticket_waits_for_rich_gold(self):
        data = snapshot(gold=180)
        strategy = bind(Strategy(CONFIG), data)
        strategy.w.guns = [
            role(20, 12, 24, "rocket", level=3, health=1000),
            role(21, 13, 24, "rocket", level=3, health=1000),
            role(22, 12, 23, "rocket", level=3, health=1000),
        ]
        self.assertFalse(any(item["kind"] == "station" for item in strategy._upgrade_agenda()))
        strategy.gold = 280
        self.assertTrue(any(item["kind"] == "station" for item in strategy._upgrade_agenda()))

    def test_news_llm_only_after_regex_miss(self):
        from agent.protocol import Reasoning
        parsed = Reasoning({"enable_llm": True, "enable_treasure": False})
        data = snapshot()
        data["worldNews"] = {"officialNews": "铁矿明日停工，检修两天", "folkLegends": ""}
        first = parsed.update(data, 1, "2")
        self.assertFalse(first.get("prompt"))
        self.assertTrue(parsed._official_resolved)
        vague = Reasoning({"enable_llm": True, "enable_treasure": False})
        data["worldNews"] = {"officialNews": "矿区传闻价格可能波动", "folkLegends": ""}
        second = vague.update(data, 1, "2")
        self.assertTrue(second.get("prompt"))
        self.assertIn("halt", second["prompt"])

    def test_regex_treasure_stays_unready(self):
        from agent.protocol import Reasoning
        brain = Reasoning({"enable_llm": False, "enable_treasure": True})
        data = snapshot()
        data["worldNews"] = {
            "officialNews": "",
            "folkLegends": "祭坛在（10，12），需献上星辰之沙，第3天开启",
        }
        brain.update(data, 1, "2")
        self.assertTrue(brain.treasure)
        self.assertFalse(brain.treasure.get("ready"))

    def test_task_prompt_keeps_family_recipe_and_workspace(self):
        from agent.protocol import Reasoning
        brain = Reasoning({"enable_llm": True, "enable_treasure": False})
        brain._task = "修复 check 并提交 Token"
        brain._task_family = "自进化类1"
        brain._sop_notes["自进化类1"] = "先读 README 再跑 check"
        brain._task_workspace = "/tmp/ws_demo"
        prompt = brain._task_prompt(snapshot())
        self.assertIn("先读 README 再跑 check", prompt)
        self.assertIn("/tmp/ws_demo", prompt)
        self.assertEqual(brain._line_token("TOKEN: xxx"), "")
        self.assertEqual(brain._line_token('{"token": "should-not-use"}'), "")
        self.assertEqual(brain._line_token("TOKEN: file-ok-123"), "file-ok-123")
        self.assertTrue(brain._bind_workspace("./check").startswith("cd "))

    def test_market_holds_ore_before_closure_and_releases_on_rise(self):
        from agent.protocol import Reasoning
        data = snapshot(gold=80)
        worker = data["teamOur"]["roles"][2]
        worker["backpack"] = ["copper"] * 8
        worker["pos"] = {"x": 19, "y": 16}
        strategy = bind(Strategy(CONFIG), data)
        strategy.reasoning._closures = [{"ore": "copper", "start_day": 2, "end_day": 3}]
        strategy.market.observe(1, strategy.prices, strategy.reasoning)
        self.assertTrue(strategy.market.has_plan("copper"))
        self.assertGreater(strategy.market.collection_weight("copper", worker), 1.0)
        self.assertFalse(strategy._should_sell(worker))
        strategy.prices["copper"] = 20
        strategy.market.observe(2, strategy.prices, strategy.reasoning)
        self.assertTrue(strategy.market.take_profit("copper"))
        self.assertTrue(strategy._should_sell(worker))

    def test_worker_finishes_detected_mine_before_selling(self):
        data = snapshot()
        worker = data["teamOur"]["roles"][3]
        worker["pos"] = {"x": 5, "y": 24}
        worker["backpack"] = ["copper"] * 40
        worker["backPackCapability"] = 100
        strategy = bind(Strategy(CONFIG), data)
        strategy.mine_claims[str(worker["id"])] = {"pos": (4, 24), "ore": "stone"}
        self.assertFalse(strategy._should_sell(worker))
        self.assertTrue(strategy._finish_current_mine(worker, night=False))
        self.assertEqual(strategy.commands[str(worker["id"])]["action"], "collect")
        worker["backpack"] = ["copper"] * 100
        strategy.commands = {}
        self.assertFalse(strategy._finish_current_mine(worker, night=False))
        self.assertTrue(strategy._should_sell(worker))

    def test_night_unmanned_guns_can_abort_mine(self):
        data = snapshot(night=True)
        data["teamOur"]["roles"][1]["pos"] = {"x": 2, "y": 2}
        worker = data["teamOur"]["roles"][3]
        worker["pos"] = {"x": 5, "y": 24}
        worker["backpack"] = ["copper"] * 10
        strategy = bind(Strategy(CONFIG), data)
        strategy.w.guns = [role(20, 12, 24, "rocket"), role(21, 13, 24, "rocket"),
                           role(22, 12, 23, "rocket")]
        strategy._control_point = (11, 24)
        strategy.mine_claims[str(worker["id"])] = {"pos": (4, 24), "ore": "stone"}
        self.assertTrue(strategy._guns_unmanned())
        self.assertFalse(strategy._finish_current_mine(worker, night=True))
        data["teamOur"]["roles"][1]["pos"] = {"x": 11, "y": 24}
        strategy = bind(Strategy(CONFIG), data)
        strategy.w.guns = [role(20, 12, 24, "rocket"), role(21, 13, 24, "rocket"),
                           role(22, 12, 23, "rocket")]
        strategy._control_point = (11, 24)
        strategy.mine_claims[str(worker["id"])] = {"pos": (4, 24), "ore": "stone"}
        worker = next(item for item in strategy.w.people if str(item["id"]) == "4")
        worker["pos"] = {"x": 5, "y": 24}
        self.assertFalse(strategy._guns_unmanned())
        self.assertTrue(strategy._finish_current_mine(worker, night=True))
        self.assertEqual(strategy.commands[str(worker["id"])]["action"], "collect")


if __name__ == "__main__":
    unittest.main()
