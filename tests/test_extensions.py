import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trace_core.extensions import ConfidenceWeightedController, StreamingPrefixController
from trace_core.canonicalizer import extract_gsm8k_answer


class TestConfidenceWeightedController(unittest.TestCase):
    def test_high_confidence_override(self):
        """
        Verify that a single high-confidence candidate can override
        two low-confidence candidates with weak agreement.
        """
        controller = ConfidenceWeightedController(k_init=2, k_max=4, tau_high=0.75)
        
        # Rollouts:
        # 0: "wrong_action", confidence 0.15
        # 1: "wrong_action", confidence 0.15 -> total weight = 0.30
        # 2: "correct_action", confidence 0.95 -> total weight = 0.95
        # Weighted agreement for "correct_action": 0.95 / (0.30 + 0.95) = 76% >= 75%!
        responses = [
            {"action": "wrong_action", "confidence": 0.15},
            {"action": "wrong_action", "confidence": 0.15},
            {"action": "correct_action", "confidence": 0.95},
        ]
        
        def sample_fn(temp, s):
            return responses[s]
            
        res = controller.execute(sample_fn, base_seed=0)
        self.assertEqual(res["action"], "correct_action")
        self.assertEqual(res["calls"], 3)
        self.assertTrue(res["early_exit"])
        self.assertGreaterEqual(res["agreement"], 0.75)

    def test_confident_early_exit(self):
        """Verify early exit at k=2 when both initial candidates agree with high confidence."""
        controller = ConfidenceWeightedController(k_init=2, k_max=4, tau_high=0.75)
        
        responses = [
            {"action": "go to kitchen", "confidence": 0.90},
            {"action": "go to kitchen", "confidence": 0.92},
        ]
        
        def sample_fn(temp, s):
            return responses[s]
            
        res = controller.execute(sample_fn, base_seed=0)
        self.assertEqual(res["action"], "go to kitchen")
        self.assertEqual(res["calls"], 2)
        self.assertEqual(res["agreement"], 1.0)
        self.assertTrue(res["early_exit"])

    def test_expansion_to_kmax_when_ambiguous(self):
        """Verify expansion up to k_max when candidates disagree and no consensus reaches tau_high."""
        controller = ConfidenceWeightedController(k_init=2, k_max=4, tau_high=0.85)

        responses = [
            {"action": "action_A", "confidence": 0.5},
            {"action": "action_B", "confidence": 0.5},
            {"action": "action_A", "confidence": 0.5},
            {"action": "action_C", "confidence": 0.5},
        ]

        def sample_fn(temp, s):
            return responses[s]

        res = controller.execute(sample_fn, base_seed=0)
        self.assertEqual(res["calls"], 4)
        self.assertEqual(res["action"], "action_a")
        self.assertFalse(res["early_exit"])


class TestStreamingPrefixController(unittest.TestCase):
    def test_streaming_early_exit_and_token_savings(self):
        """
        Verify that streaming prunes generation once target is reached,
        and early-exits at k=2 when rollouts agree.
        """
        controller = StreamingPrefixController(
            k_init=2,
            k_max=4,
            tau_high=0.75,
            max_tokens_budget=256
        )

        def stop_check_fn(text):
            if "####" in text:
                parts = text.split("####")
                if parts[-1].strip() and any(c.isdigit() for c in parts[-1]):
                    ans, _ = extract_gsm8k_answer(text)
                    return True, ans
            return False, None

        # Simulated streams that emit answer early
        def stream_sample_fn(temp, s, checker):
            chunks = ["Let's calculate: ", "10 + 5 = ", "15. ", "Final Answer: #### 15", " trailing tokens that should be skipped..."]
            acc = ""
            toks = 0
            act = None
            for c in chunks:
                acc += c
                toks += 5
                is_done, act = checker(acc)
                if is_done:
                    break
            return {"text": acc, "action": act, "tokens": toks}

        res = controller.execute(stream_sample_fn, stop_check_fn, base_seed=0)
        self.assertEqual(res["action"], "15")
        self.assertEqual(res["calls"], 2)
        self.assertTrue(res["early_exit"])
        self.assertLess(res["total_tokens"], 100)
        self.assertGreater(res["token_savings_pct"], 80.0)

    def test_streaming_expansion_on_divergence(self):
        """Verify that streaming expands when initial streams disagree until reaching consensus."""
        controller = StreamingPrefixController(
            k_init=2,
            k_max=4,
            tau_high=0.75,
            max_tokens_budget=100
        )

        stream_answers = ["10", "20", "10", "10"]

        def stop_check_fn(text):
            for a in ["10", "20"]:
                if f"Answer: {a}" in text:
                    return True, a
            return False, None

        def stream_sample_fn(temp, s, checker):
            ans = stream_answers[s]
            text = f"Calculated Answer: {ans}"
            return {"text": text, "action": ans, "tokens": 8}

        res = controller.execute(stream_sample_fn, stop_check_fn, base_seed=0)
        self.assertEqual(res["action"], "10")
        self.assertEqual(res["calls"], 4)
        self.assertGreaterEqual(res["agreement"], 0.75)

    def test_streaming_fallback_when_target_not_found(self):
        """Verify fallback behavior when stream finishes without hitting early stop criteria."""
        controller = StreamingPrefixController(
            k_init=2,
            k_max=4,
            tau_high=0.75,
            max_tokens_budget=100
        )

        def stop_check_fn(text):
            return False, None  # Never hits early stop

        def stream_sample_fn(temp, s, checker):
            # Text contains answer at the end
            return {"text": "Step 1: calculate. The answer is 42.", "tokens": 50}

        res = controller.execute(stream_sample_fn, stop_check_fn, base_seed=0)
        self.assertEqual(res["calls"], 2)
        self.assertEqual(res["action"], "step 1: calculate. the answer is 42")
        self.assertTrue(res["early_exit"])


class TestConfidenceWeightedEdgeCases(unittest.TestCase):
    def test_low_confidence_agreement_blocks_early_exit(self):
        """
        Critical safety test:
        If two rollouts agree on a hallucinated answer but both have very low confidence
        (e.g. 0.20 < min_exit_confidence 0.50), standard TrACE would early exit (100% agreement).
        ConfidenceWeightedController must REFUSE early exit and expand compute to k_max.
        """
        controller = ConfidenceWeightedController(
            k_init=2,
            k_max=4,
            tau_high=0.75,
            min_exit_confidence=0.50
        )

        # Rollouts:
        # 0: "hallucination", conf 0.20
        # 1: "hallucination", conf 0.20 -> agreement is 1.0, but conf 0.20 < 0.50
        # 2: "correct_answer", conf 0.90 -> total weights: hallucination: 0.40, correct: 0.90
        # 3: "correct_answer", conf 0.95 -> total weights: hallucination: 0.40, correct: 1.85
        responses = [
            {"action": "hallucination", "confidence": 0.20},
            {"action": "hallucination", "confidence": 0.20},
            {"action": "correct_answer", "confidence": 0.90},
            {"action": "correct_answer", "confidence": 0.95},
        ]

        def sample_fn(temp, s):
            return responses[s]

        res = controller.execute(sample_fn, base_seed=0)
        # Should NOT have exited at call 2!
        self.assertGreater(res["calls"], 2)
        # Should correctly converge to correct_answer
        self.assertEqual(res["action"], "correct_answer")
        self.assertGreaterEqual(res["agreement"], 0.75)

    def test_deterministic_tie_breaking(self):
        """Verify that exact tie in weighted agreement breaks deterministically by first occurrence."""
        controller = ConfidenceWeightedController(k_init=2, k_max=2, tau_high=0.99)
        responses = [
            {"action": "option_first", "confidence": 0.80},
            {"action": "option_second", "confidence": 0.80},
        ]
        def sample_fn(temp, s):
            return responses[s]

        res = controller.execute(sample_fn, base_seed=0)
        self.assertEqual(res["action"], "option_first")


if __name__ == "__main__":
    unittest.main()
