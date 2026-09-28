import unittest
from trace_core.answers import extract_answer, gold_from_gsm8k, is_correct, select_majority
from trace_core.features import response_features, agreement_features
from trace_core.evaluation import split_rows, choose_threshold, metric_report, policy_report

class AnswerTests(unittest.TestCase):
    def test_gsm8k_marker_preferred(self):
        self.assertEqual(extract_answer("I considered 9, but final: #### 1,200"), ("1200", "gsm8k_marker"))
    def test_fallback_is_labeled(self):
        self.assertEqual(extract_answer("Answer is 42"), ("42", "last_number_fallback"))
    def test_no_number(self):
        self.assertEqual(extract_answer("No numeric answer"), (None, "no_number"))
    def test_gold(self):
        self.assertEqual(gold_from_gsm8k("work\\n#### -12.5"), "-12.5")
    def test_vote_tie_breaks_by_first_occurrence(self):
        self.assertEqual(select_majority(["2", "1", "2", "1"]), "2")
    def test_missing_votes_ignored(self):
        self.assertEqual(select_majority([None, "7", None]), "7")
    def test_correctness(self):
        self.assertEqual(is_correct("5", "5"), 1)
        self.assertEqual(is_correct(None, "5"), 0)

class FeatureTests(unittest.TestCase):
    def test_features_and_agreement(self):
        f = response_features("How many?", "Maybe 2. Actually 3. #### 3")
        self.assertGreaterEqual(f["numeric_token_count"], 2)
        self.assertEqual(f["reversal_count"], 1)
        a = agreement_features(["3", "3", None])
        self.assertAlmostEqual(a["agreement_fraction"], 2/3)
        self.assertAlmostEqual(a["valid_answer_fraction"], 2/3)
    def test_no_valid_answers(self):
        self.assertEqual(agreement_features([None, None])["valid_answer_fraction"], 0)

class EvaluationTests(unittest.TestCase):
    def test_split_is_disjoint_and_complete(self):
        rows = [{"id": i} for i in range(100)]
        a,b,c = split_rows(rows, 11)
        ids = [{r["id"] for r in part} for part in (a,b,c)]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        self.assertEqual(set.union(*ids), set(range(100)))
        self.assertEqual([len(a),len(b),len(c)], [60,20,20])
    def test_split_invalid_fractions(self):
        with self.assertRaises(ValueError):
            split_rows([], 1, .8, .3)
    def test_threshold_is_bounded(self):
        self.assertGreaterEqual(choose_threshold([0,1,1], [.1,.8,.7]), 0)
        self.assertLessEqual(choose_threshold([0,1,1], [.1,.8,.7]), 1)
    def test_metrics_handle_single_class(self):
        m = metric_report([0,0], [.2,.3])
        self.assertIsNone(m["auroc"])
        self.assertIsNotNone(m["brier"])
    def test_policy_accounting(self):
        rows = [
            {"baseline_correct":1,"post_action_correct":0,"initial_output_tokens":30,
             "total_output_tokens":50,"initial_latency_seconds":1.0,"total_latency_seconds":2.0},
            {"baseline_correct":0,"post_action_correct":1,"initial_output_tokens":30,
             "total_output_tokens":50,"initial_latency_seconds":1.0,"total_latency_seconds":2.0},
        ]
        p=policy_report(rows,[False,True])
        self.assertEqual(p["accuracy"],1.0)
        self.assertEqual(p["mean_output_tokens"],40.0)
        self.assertEqual(p["mean_samples"],4.0)

if __name__ == "__main__":
    unittest.main()
