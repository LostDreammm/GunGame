#!/usr/bin/env python3
"""Market, task learning, engineering contract, and treasure-memory checks."""
import unittest

from agent.market import MarketPlanner, grounded_closure
from agent.news_proof import _parse, _sentences
from agent.task_engineering import engineering_workspace
from agent.task_learning import TaskLearning
from agent.treasure_memory import TreasureMemory


class MarketTests(unittest.TestCase):
    def test_grounded_closure_needs_exact_quote_and_dates(self):
        history = [{"day": 1, "officialNews": "铁矿明日停工，检修需要2天。"}]
        ok = {"ore": "iron", "start_day": 2, "end_day": 3,
              "evidence": "铁矿明日停工，检修需要2天。"}
        self.assertTrue(grounded_closure(ok, history))
        bad = dict(ok, start_day=1, end_day=2)
        self.assertFalse(grounded_closure(bad, history))

    def test_planner_blocks_during_window(self):
        planner = MarketPlanner({"market": {"enabled": True}})
        reasoning = type("R", (), {"_closures": [{"ore": "iron", "start_day": 2, "end_day": 3}]})()
        planner.observe(2, {"iron": 6, "copper": 8, "stone": 2}, reasoning)
        self.assertIn("iron", planner.blocked_ores)


class LearningTests(unittest.TestCase):
    def test_learn_strips_answer_and_estimates(self):
        memory = TaskLearning()
        self.assertEqual(memory.estimate("自进化类1"), 6)
        sop = memory.learn("自进化类1", "答案是 42", "42",
                           [{"command": "curl http://127.0.0.1/api"}], 4)
        self.assertNotIn("42", sop)
        self.assertLessEqual(memory.estimate("自进化类1"), 20)
        memory.record_failure("自进化类1")
        self.assertGreaterEqual(memory.estimate("自进化类1"), memory.estimate("missing"))


class EngineeringTests(unittest.TestCase):
    def test_incomplete_context_is_rejected(self):
        self.assertIsNone(engineering_workspace({"status": "ok", "workspace": "/tmp/ws"}))


class TreasureProofTests(unittest.TestCase):
    def test_parse_coord_and_item_sentence(self):
        self.assertEqual(_sentences("祭坛在（10，12）。献上星辰之沙。"),
                         ["祭坛在（10，12）。", "献上星辰之沙。"])
        self.assertEqual(_parse("祭坛在（10，12）。", 1, 1), ("pos", "in", {(10, 12)}))
        field, op, items = _parse("只需献上星辰之沙", 1, 1)
        self.assertEqual(field, "items")
        self.assertEqual(op, "exact")
        self.assertEqual(items, {"StarSand"})

    def test_memory_rejects_invalid_history(self):
        memory = TreasureMemory()
        self.assertFalse(memory.sync([{"day": 99, "officialNews": "x"}], {"roundNo": 1}, 1))
        self.assertIsNone(memory.candidate)


if __name__ == "__main__":
    unittest.main()
