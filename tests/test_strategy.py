#!/usr/bin/env python3
"""Acceptance tests for CoreGeek layout, night roles, and economy policy."""
import unittest

from agent.brain import (BASE_LOW_HEALTH_RATIO, Strategy, WALL_REPAIR_KIT,
                         WALL_REPAIR_KIT_TARGET_STOCK, can_complete_same_day,
                         fixer_restock_count, preferred_ores, upgrade_group_order,
                         wall_stone_deficit)
from agent.grid import (ROCKET_TURRET_COUNT, TARGET_WALL_COUNT, World,
                        base_region, building_rings,
                        cluster_is_operable, generate_rocket_positions,
                        generate_wall_positions, template_wall_cells)


MAP_W, MAP_H = 41, 32
TOP_LEFT_BASE = (10, 24)
BOTTOM_RIGHT_BASE = (30, 10)
CONFIG = {
    "round_origin": 1,
    "enable_llm": False,
    "enable_treasure": False,
    "return_margin": 4,
    "economy": {
        "sell_value_threshold": 40,
        "consumable_start_round": 1,
        "opening_cash_reserve": 0,
    },
    "construction": {
        "mode": "auto",
        "wall_target": 12,
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
    payload = {
        "id": ident, "pos": {"x": x, "y": y}, "roleType": kind,
        "health": health if health is not None else (200 if kind == "pioneer" else 220),
        "level": extra.get("level", 1),
        "backPackCapability": extra.get("backPackCapability", 40 if kind == "pioneer" else 100),
        "backpack": list(backpack or []),
        "attackPower": extra.get("attackPower", 0),
        "attackRange": extra.get("attackRange", 0),
        "cooldown": extra.get("cooldown", 0),
    }
    return payload


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
    data = {
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
            {"name": "WallUpgradeVoucher1", "price": 20},
            {"name": "StationUpgradeVoucher1", "price": 100},
            {"name": WALL_REPAIR_KIT, "price": 10},
            {"name": "Medicine", "price": 10},
        ],
        "lastRoundRoleActionResults": {},
        "errors": [],
    }
    return data


def bind(strategy, data, **attrs):
    strategy.w = World(data, CONFIG)
    strategy.commands = {}
    strategy.reserved = set()
    strategy.build_jobs = set()
    strategy.gold = int(data["teamOur"].get("goldNum", 0))
    strategy.prices = {x["name"]: x["price"] for x in data["vendorShopList"]}
    strategy.shop = {x["name"]: x["price"] for x in data["weaponShopList"]}
    strategy.defense_mode = "reward"
    strategy.safety_ratio = 3.0
    strategy._refresh_layout()
    for key, value in attrs.items():
        setattr(strategy, key, value)
    return strategy


class LayoutTests(unittest.TestCase):
    def test_top_left_rockets_match_specified_offsets(self):
        base = station(*TOP_LEFT_BASE)
        x, y = TOP_LEFT_BASE
        cells = generate_rocket_positions(base, MAP_W, MAP_H)
        self.assertEqual(base_region(base, MAP_W, MAP_H), "top_left")
        self.assertEqual(cells, [(x - 1, y), (x - 1, y - 2), (x, y - 2)])
        blue, _ = building_rings(base, MAP_W, MAP_H)
        self.assertTrue(set(cells) <= blue)
        self.assertTrue(cluster_is_operable(base, cells, MAP_W, MAP_H))

    def test_bottom_right_rockets_match_specified_offsets(self):
        base = station(*BOTTOM_RIGHT_BASE)
        x, y = BOTTOM_RIGHT_BASE
        cells = generate_rocket_positions(base, MAP_W, MAP_H)
        self.assertEqual(base_region(base, MAP_W, MAP_H), "bottom_right")
        self.assertEqual(cells, [(x + 2, y - 1), (x + 2, y + 1), (x + 1, y + 1)])
        blue, _ = building_rings(base, MAP_W, MAP_H)
        self.assertTrue(set(cells) <= blue)
        top = generate_rocket_positions(station(*TOP_LEFT_BASE), MAP_W, MAP_H)
        self.assertNotEqual(set(cells), set(top))
        self.assertTrue(cluster_is_operable(base, cells, MAP_W, MAP_H))

    def test_both_bases_generate_twelve_unique_walls(self):
        for xy in (TOP_LEFT_BASE, BOTTOM_RIGHT_BASE):
            base = station(*xy)
            walls = generate_wall_positions(base, MAP_W, MAP_H)
            self.assertEqual(len(walls), TARGET_WALL_COUNT, xy)
            self.assertEqual(len(set(walls)), TARGET_WALL_COUNT)
            template = template_wall_cells(base, MAP_W, MAP_H)
            self.assertEqual(len(set(template)), TARGET_WALL_COUNT)

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
        template = template_wall_cells(base, MAP_W, MAP_H)
        walls = generate_wall_positions(base, MAP_W, MAP_H, blocked=set(template[:3]))
        self.assertEqual(len(set(walls)), TARGET_WALL_COUNT)
        self.assertTrue(set(template[:3]).isdisjoint(set(walls)))

    def test_no_legal_cells_skips_safely(self):
        base = station(*TOP_LEFT_BASE)
        blue, yellow = building_rings(base, MAP_W, MAP_H)
        self.assertEqual(generate_rocket_positions(base, MAP_W, MAP_H, blocked=blue), [])
        self.assertEqual(generate_wall_positions(base, MAP_W, MAP_H, blocked=yellow), [])
        self.assertEqual(generate_rocket_positions(None, MAP_W, MAP_H), [])
        self.assertEqual(generate_wall_positions(None, MAP_W, MAP_H), [])


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


class NightAndEconomyTests(unittest.TestCase):
    def test_night_only_pioneer_controls_guns_workers_keep_mining(self):
        sx, sy = TOP_LEFT_BASE
        rockets = generate_rocket_positions(station(sx, sy), MAP_W, MAP_H)
        roles = [
            role(2, rockets[0][0] - 1, rockets[0][1], "pioneer"),
            role(3, 5, 24, "worker"),
            role(4, 6, 23, "worker"),
        ]
        guns = [
            role(20 + i, x, y, "rocket", attackRange=10, attackPower=20, cooldown=0)
            for i, (x, y) in enumerate(rockets)
        ]
        data = snapshot(roles=roles, night=True, robots=[
            role(90, rockets[0][0] + 3, rockets[0][1], "smallRobot", health=30),
        ])
        data["teamOur"]["roles"] = [station(sx, sy)] + roles + guns
        result = Strategy(CONFIG).callback(data)
        commands = result["roleCommandMap"]
        self.assertIn("20", commands)
        self.assertEqual(commands["20"]["action"], "attack")
        self.assertEqual(str(commands["20"]["controllerId"]), "2")
        for worker_id in ("3", "4"):
            action = commands.get(worker_id, {}).get("action")
            self.assertIn(action, ("collect", "move", "sell", None))
            self.assertNotEqual(action, "attack")

    def test_night_worker_covers_guns_when_pioneer_is_busy(self):
        sx, sy = TOP_LEFT_BASE
        rockets = generate_rocket_positions(station(sx, sy), MAP_W, MAP_H)
        pad_x, pad_y = rockets[0][0] - 1, rockets[0][1]
        roles = [
            role(2, 4, 4, "pioneer"),
            role(3, pad_x, pad_y, "worker"),
            role(4, 6, 23, "worker"),
        ]
        guns = [
            role(20 + i, x, y, "rocket", attackRange=10, attackPower=20, cooldown=0)
            for i, (x, y) in enumerate(rockets)
        ]
        data = snapshot(roles=roles, night=True, robots=[
            role(90, rockets[0][0] + 3, rockets[0][1], "smallRobot", health=30),
        ])
        data["phaseTask"] = "solve a puzzle"
        data["teamOur"]["roles"] = [station(sx, sy)] + roles + guns
        result = Strategy(CONFIG).callback(data)
        attacks = [cmd for cmd in result["roleCommandMap"].values()
                   if cmd.get("action") == "attack"]
        self.assertTrue(attacks)
        self.assertEqual(str(attacks[0]["controllerId"]), "3")

    def test_fixer_restock_stops_at_five(self):
        self.assertEqual(fixer_restock_count(0), 5)
        self.assertEqual(fixer_restock_count(4), 1)
        self.assertEqual(fixer_restock_count(5), 0)
        self.assertEqual(fixer_restock_count(8), 0)
        strategy = bind(Strategy(CONFIG), snapshot(gold=80))
        shops = strategy._zones("weaponShop")
        worker = strategy.w.people[-1]
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
        worker["backpack"] = [WALL_REPAIR_KIT] * 5
        self.assertFalse(strategy._buy_fixer(worker, shops))

    def test_upgrade_order_flips_below_half_base_health(self):
        self.assertEqual(upgrade_group_order(0.49), ("weapon", "station", "wall"))
        self.assertEqual(upgrade_group_order(BASE_LOW_HEALTH_RATIO), ("weapon", "wall", "station"))
        self.assertEqual(upgrade_group_order(0.8), ("weapon", "wall", "station"))
        data = snapshot(gold=200)
        data["teamOur"]["roles"][0]["health"] = 700
        strategy = bind(Strategy(CONFIG), data)
        strategy._health_caps[("station", 1)] = 1500
        strategy.w.guns = [role(20, 12, 24, "rocket")]
        strategy.w.ours.append(role(30, 13, 24, "wall"))
        name, building = strategy._next_upgrade()
        self.assertEqual(name, "WeaponUpgradeVoucher1")
        strategy.shop.pop("WeaponUpgradeVoucher1")
        name, building = strategy._next_upgrade()
        self.assertEqual(name, "StationUpgradeVoucher1")
        self.assertEqual(building["roleType"], "station")
        data["teamOur"]["roles"][0]["health"] = 750
        strategy = bind(Strategy(CONFIG), data)
        strategy._health_caps[("station", 1)] = 1500
        strategy.w.guns = [role(20, 12, 24, "rocket")]
        strategy.w.ours.append(role(30, 13, 24, "wall"))
        strategy.shop.pop("WeaponUpgradeVoucher1")
        name, building = strategy._next_upgrade()
        self.assertEqual(name, "WallUpgradeVoucher1")

    def test_do_not_sell_wall_stones_before_twelve_walls(self):
        data = snapshot()
        worker = data["teamOur"]["roles"][2]
        worker["backpack"] = ["stone"] * 8
        worker["pos"] = {"x": 19, "y": 16}
        strategy = bind(Strategy(CONFIG), data)
        strategy._wall_slots = generate_wall_positions(strategy.w.base, MAP_W, MAP_H)
        self.assertGreater(strategy._stone_keep_for(worker), 0)
        self.assertFalse(strategy._sell(worker, True))

    def test_rebuild_collects_only_the_stone_gap(self):
        self.assertEqual(wall_stone_deficit(12, 0), 0)
        self.assertEqual(wall_stone_deficit(10, 0), 2)
        self.assertEqual(wall_stone_deficit(10, 2), 0)
        self.assertEqual(wall_stone_deficit(10, 1), 1)
        data = snapshot()
        strategy = bind(Strategy(CONFIG), data)
        slots = generate_wall_positions(strategy.w.base, MAP_W, MAP_H)
        strategy._wall_slots = slots
        strategy._wall_plan = slots
        strategy._seen_walls = set(slots)
        walls = [role(40 + i, x, y, "wall") for i, (x, y) in enumerate(slots[:11])]
        strategy.w.ours.extend(walls)
        self.assertEqual(strategy._effective_walls(), 11)
        self.assertEqual(strategy._stone_deficit(), 1)
        self.assertEqual(preferred_ores(strategy._stone_deficit(), strategy.prices), ("stone",))

    def test_complete_walls_prefer_high_value_ore(self):
        prices = {"stone": 2, "iron": 6, "copper": 8}
        self.assertEqual(preferred_ores(0, prices), ("copper", "iron", "stone"))
        self.assertEqual(preferred_ores(3, prices), ("stone",))

    def test_workers_fill_backpack_before_selling(self):
        data = snapshot()
        worker = data["teamOur"]["roles"][2]
        worker["backpack"] = ["copper"] * 20
        worker["pos"] = {"x": 4, "y": 24}
        strategy = bind(Strategy(CONFIG), data)
        strategy._wall_slots = generate_wall_positions(strategy.w.base, MAP_W, MAP_H)
        strategy._wall_plan = list(strategy._wall_slots)
        fake_walls = [role(40 + i, x, y, "wall") for i, (x, y) in enumerate(strategy._wall_slots)]
        strategy.w.ours.extend(fake_walls)
        self.assertEqual(strategy._bag_space(worker), 80)
        self.assertFalse(strategy._sell(worker))
        self.assertTrue(strategy._mine(worker))
        action = strategy.commands[str(worker["id"])]["action"]
        self.assertIn(action, ("collect", "move"))
        strategy.commands = {}
        worker["backpack"] = ["copper"] * 100
        self.assertTrue(strategy._bag_full(worker))
        self.assertTrue(strategy._sell(worker))
        self.assertEqual(strategy.commands[str(worker["id"])]["action"], "move")

    def test_same_day_sell_only_when_cycle_allows(self):
        self.assertTrue(can_complete_same_day(12, 12))
        self.assertFalse(can_complete_same_day(11, 12))
        self.assertFalse(can_complete_same_day(None, 3))
        data = snapshot(round_no=69)
        worker = data["teamOur"]["roles"][2]
        worker["backpack"] = ["copper"] * 8
        worker["pos"] = {"x": 4, "y": 24}
        strategy = bind(Strategy(CONFIG), data)
        strategy._wall_slots = generate_wall_positions(strategy.w.base, MAP_W, MAP_H)
        strategy._wall_plan = list(strategy._wall_slots)
        fake_walls = [role(40 + i, x, y, "wall") for i, (x, y) in enumerate(strategy._wall_slots)]
        strategy.w.ours.extend(fake_walls)
        self.assertGreaterEqual(strategy._effective_walls(), TARGET_WALL_COUNT)
        self.assertGreaterEqual(strategy.w.remaining_cycle, 60)
        self.assertTrue(strategy._can_complete_sell(worker))
        self.assertTrue(strategy._sell(worker, True))
        data_late = snapshot(round_no=129)
        worker = data_late["teamOur"]["roles"][2]
        worker["backpack"] = ["copper"] * 8
        worker["pos"] = {"x": 4, "y": 24}
        late = bind(Strategy(CONFIG), data_late)
        late._wall_slots = list(strategy._wall_slots)
        late._wall_plan = list(strategy._wall_slots)
        late.w.ours.extend(fake_walls)
        self.assertFalse(late._can_complete_sell(worker))
        self.assertFalse(late._sell(worker, True))

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

    def test_pioneer_accepts_task_even_near_nightfall(self):
        data = add_self_task(snapshot(round_no=66))
        result = Strategy(CONFIG).callback(data)
        cmd = result["roleCommandMap"].get("2", {})
        self.assertIn(cmd.get("action"), ("move", "acceptTask"))

    def test_pioneer_prefers_task_over_treasure(self):
        data = add_self_task(snapshot())
        strategy = Strategy(CONFIG)
        strategy.reasoning.treasure = {
            "pos": {"x": 25, "y": 20}, "items": ["StarSand"],
            "open_round": 1, "close_round": 2000,
        }
        result = strategy.callback(data)
        cmd = result["roleCommandMap"].get("2", {})
        self.assertIn(cmd.get("action"), ("move", "acceptTask"))
        if cmd.get("action") == "move":
            dest = cmd["targetPos"][0]
            start = (13, 24)
            task = (8, 20)
            altar = (25, 20)
            after = (dest["x"], dest["y"])
            self.assertLess(
                max(abs(after[0] - task[0]), abs(after[1] - task[1])),
                max(abs(start[0] - task[0]), abs(start[1] - task[1])),
            )
            self.assertGreater(
                max(abs(after[0] - altar[0]), abs(after[1] - altar[1])),
                max(abs(after[0] - task[0]), abs(after[1] - task[1])),
            )

    def test_pioneer_takes_tasks_at_night_when_worker_covers(self):
        sx, sy = TOP_LEFT_BASE
        rockets = generate_rocket_positions(station(sx, sy), MAP_W, MAP_H)
        roles = [
            role(2, rockets[0][0] - 1, rockets[0][1], "pioneer"),
            role(3, 5, 24, "worker"),
            role(4, 6, 23, "worker"),
        ]
        guns = [
            role(20 + i, x, y, "rocket", attackRange=10, attackPower=20, cooldown=0)
            for i, (x, y) in enumerate(rockets)
        ]
        data = snapshot(roles=roles, night=True, robots=[
            role(90, rockets[0][0] + 3, rockets[0][1], "smallRobot", health=30),
        ])
        add_self_task(data)
        data["teamOur"]["roles"] = [station(sx, sy)] + roles + guns
        result = Strategy(CONFIG).callback(data)
        commands = result["roleCommandMap"]
        if commands.get("20", {}).get("action") == "attack":
            self.assertNotEqual(str(commands["20"]["controllerId"]), "2")
        self.assertIn(commands.get("2", {}).get("action"), ("move", "acceptTask"))

    def test_must_sell_next_day_if_night_had_no_sell(self):
        data = snapshot()
        worker = data["teamOur"]["roles"][2]
        worker["backpack"] = ["copper"] * 20
        worker["pos"] = {"x": 4, "y": 24}
        strategy = bind(Strategy(CONFIG), data)
        slots = generate_wall_positions(strategy.w.base, MAP_W, MAP_H)
        strategy._wall_slots = slots
        strategy._wall_plan = list(slots)
        strategy.w.ours.extend(role(40 + i, x, y, "wall") for i, (x, y) in enumerate(slots))
        strategy._was_night = True
        strategy._sold_this_night = False
        strategy._observe_world()
        self.assertTrue(strategy._must_sell_today)
        self.assertTrue(strategy._sell(worker))
        self.assertEqual(strategy.commands[str(worker["id"])]["action"], "move")

    def test_no_forced_sell_if_night_already_sold(self):
        data = snapshot()
        worker = data["teamOur"]["roles"][2]
        worker["backpack"] = ["copper"] * 20
        worker["pos"] = {"x": 4, "y": 24}
        strategy = bind(Strategy(CONFIG), data)
        slots = generate_wall_positions(strategy.w.base, MAP_W, MAP_H)
        strategy._wall_slots = slots
        strategy._wall_plan = list(slots)
        strategy.w.ours.extend(role(40 + i, x, y, "wall") for i, (x, y) in enumerate(slots))
        strategy._was_night = True
        strategy._sold_this_night = True
        strategy._observe_world()
        self.assertFalse(strategy._must_sell_today)
        self.assertFalse(strategy._sell(worker))


if __name__ == "__main__":
    unittest.main()
