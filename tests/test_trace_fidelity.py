import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trace_core.canonicalizer import canonicalize_action, extract_gsm8k_answer, extract_gold_gsm8k, normalize_number
from trace_core.controller import GreedyController, SelfConsistencyController, TrACEController
from trace_core.minihouse import MiniHouseEnv, generate_minihouse_tasks

class TestCanonicalizer(unittest.TestCase):
    def test_canonicalize_action(self):
        self.assertEqual(canonicalize_action("  Go to kitchen.  "), "go to kitchen")
        self.assertEqual(canonicalize_action("TAKE APPLE FROM FRIDGE!"), "take apple from fridge")
        self.assertEqual(canonicalize_action("Open   Microwave.."), "open microwave")

    def test_gsm8k_extraction_markers(self):
        # 1. Answer: marker
        ans, method = extract_gsm8k_answer("The final calculation gives Answer: 42")
        self.assertEqual(ans, "42")
        self.assertEqual(method, "answer_marker")

        # 2. GSM8K #### marker with currency and comma
        ans, method = extract_gsm8k_answer("Therefore, total revenue is #### $1,250.0")
        self.assertEqual(ans, "1250")
        self.assertEqual(method, "gsm8k_marker")

        # 3. LaTeX \boxed{} marker
        ans, method = extract_gsm8k_answer("Thus \\boxed{ 99 } is the count")
        self.assertEqual(ans, "99")
        self.assertEqual(method, "boxed_marker")

        # 4. Fallback to last number
        ans, method = extract_gsm8k_answer("He sold 10 cows and remaining 15")
        self.assertEqual(ans, "15")
        self.assertEqual(method, "last_number_fallback")

    def test_normalize_number(self):
        self.assertEqual(normalize_number("$1,000.00"), "1000")
        self.assertEqual(normalize_number("12.5"), "12.5")
        self.assertEqual(normalize_number("0"), "0")

    def test_extract_gold(self):
        self.assertEqual(extract_gold_gsm8k("Some text\n#### 42"), "42")
        self.assertEqual(extract_gold_gsm8k("#### $3,500"), "3500")

class TestControllers(unittest.TestCase):
    def test_greedy_controller(self):
        ctrl = GreedyController()
        res = ctrl.execute(lambda t, s: "Go to kitchen.")
        self.assertEqual(res["calls"], 1)
        self.assertEqual(res["action"], "go to kitchen")

    def test_sc4_controller(self):
        ctrl = SelfConsistencyController(k=4)
        samples = ["a", "b", "a", "a"]
        res = ctrl.execute(lambda t, s: samples[s])
        self.assertEqual(res["calls"], 4)
        self.assertEqual(res["action"], "a")
        self.assertAlmostEqual(res["agreement"], 0.75)

    def test_trace_early_exit(self):
        """When initial 2 samples agree (alpha=1.0 >= 0.75), TrACE exits immediately with 2 calls."""
        ctrl = TrACEController(k_init=2, k_max=4, tau_high=0.75)
        samples = ["a", "a", "b", "b"]
        res = ctrl.execute(lambda t, s: samples[s])
        self.assertEqual(res["calls"], 2)
        self.assertEqual(res["action"], "a")
        self.assertEqual(res["agreement"], 1.0)
        self.assertTrue(res["early_exit"])

    def test_trace_expansion_to_cap(self):
        """When initial 2 disagree and 3rd doesn't meet 0.75 threshold, TrACE expands to k_max=4."""
        ctrl = TrACEController(k_init=2, k_max=4, tau_high=0.75)
        # sample 0: "a", sample 1: "b" (agreement 0.5 < 0.75)
        # sample 2: "a" (agreement 2/3 = 0.667 < 0.75)
        # sample 3: "a" (agreement 3/4 = 0.75 >= 0.75)
        samples = ["a", "b", "a", "a"]
        res = ctrl.execute(lambda t, s: samples[s])
        self.assertEqual(res["calls"], 4)
        self.assertEqual(res["action"], "a")
        self.assertAlmostEqual(res["agreement"], 0.75)

    def test_trace8_expansion(self):
        """TrACE-8 continues sampling when agreement is low."""
        ctrl = TrACEController(k_init=2, k_max=8, tau_high=0.75)
        # 2 samples: ["a", "a"] -> exits at 2
        res = ctrl.execute(lambda t, s: "a")
        self.assertEqual(res["calls"], 2)

class TestMiniHouse(unittest.TestCase):
    def test_task_generation(self):
        tasks = generate_minihouse_tasks(n_tasks=10, seed=0)
        self.assertEqual(len(tasks), 10)
        for t in tasks:
            self.assertIn(t["template_id"], [0, 1])
            self.assertNotEqual(t["start_receptacle"], t["target_receptacle"])

    def test_env_simulation(self):
        env = MiniHouseEnv(template_id=0, target_object="apple", target_receptacle="countertop")
        self.assertEqual(env.current_room, "kitchen")
        val_acts = env.get_valid_actions()
        self.assertIn("open fridge", val_acts)
        self.assertIn("go to living_room", val_acts)

        # Open fridge
        obs, succ, done = env.step("open fridge")
        self.assertFalse(succ)
        self.assertFalse(done)

        # Now take apple from fridge should be valid
        val_acts = env.get_valid_actions()
        self.assertIn("take apple from fridge", val_acts)
        obs, succ, done = env.step("take apple from fridge")
        self.assertEqual(env.inventory, "apple")

        # Put apple in countertop
        obs, succ, done = env.step("put apple in countertop")
        self.assertTrue(succ)
        self.assertTrue(done)

if __name__ == "__main__":
    unittest.main()
