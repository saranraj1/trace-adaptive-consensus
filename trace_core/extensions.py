"""
TrACE Novel Research Extensions:
1. StreamingPrefixController: Token-prefix early pruning and streaming consensus.
2. ConfidenceWeightedController: Confidence/logprob-calibrated adaptive compute.
"""

import time
import math
from typing import Callable, List, Dict, Any, Optional, Tuple
from collections import defaultdict
from trace_core.canonicalizer import canonicalize_action


class StreamingPrefixController:
    """
    Extension 1: Sub-Trajectory TrACE / Token-Prefix Early Pruning.
    
    Instead of waiting for candidates to finish full sequence generation (e.g. 256 tokens),
    this controller:
    1. Streams candidate rollouts in real time.
    2. Uses an early target completion checker (e.g. detecting '#### N' or valid action).
    3. Halts generation immediately once the target is identified (saving intra-call tokens).
    4. Evaluates prefix consensus at k_init = 2:
       - If candidates 1 & 2 agree on target action, immediately exits without querying candidates 3+.
       - If they diverge, expands iteratively up to k_max.
    """
    def __init__(
        self,
        k_init: int = 2,
        k_max: int = 4,
        tau_high: float = 0.75,
        temperature: float = 0.7,
        max_tokens_budget: int = 256,
        name: Optional[str] = None
    ):
        self.k_init = k_init
        self.k_max = k_max
        self.tau_high = tau_high
        self.temperature = temperature
        self.max_tokens_budget = max_tokens_budget
        self.name = name or f"stream_prefix_{k_max}"

    def execute(
        self,
        stream_sample_fn: Callable[[float, int, Callable[[str], Tuple[bool, Optional[str]]]], Dict[str, Any]],
        is_target_complete_fn: Callable[[str], Tuple[bool, Optional[str]]],
        base_seed: int = 0
    ) -> Dict[str, Any]:
        """
        Execute streaming prefix early-exit controller.
        
        Args:
            stream_sample_fn: Function (temp, seed, stop_check_fn) -> {"text": str, "tokens": int, "action": str}
            is_target_complete_fn: Function (accumulated_text) -> (is_complete, extracted_action)
            base_seed: Deterministic seed base.
        """
        raw_actions = []
        canonical_actions = []
        tokens_per_call = []
        full_texts = []
        start_time = time.perf_counter()

        # Step 1: Stream initial k_init samples with early target pruning
        for i in range(self.k_init):
            res = stream_sample_fn(self.temperature, base_seed + i, is_target_complete_fn)
            raw_act = res.get("action")
            if not raw_act:
                text = res.get("text", "")
                done, extracted = is_target_complete_fn(text)
                raw_act = extracted if done else text
            canon = canonicalize_action(raw_act or "")
            raw_actions.append(raw_act or "")
            canonical_actions.append(canon)
            tokens_per_call.append(res.get("tokens", 0))
            full_texts.append(res.get("text", ""))

        # Calculate initial agreement
        counts = defaultdict(int)
        for a in canonical_actions:
            counts[a] += 1
        mode_action = max(counts, key=lambda a: (counts[a], -canonical_actions.index(a)))
        mode_count = counts[mode_action]
        agreement = mode_count / len(canonical_actions)

        # Check early exit threshold
        early_exit = agreement >= self.tau_high
        call_idx = self.k_init

        # Step 2: Expand if consensus is not reached
        while agreement < self.tau_high and len(canonical_actions) < self.k_max:
            res = stream_sample_fn(self.temperature, base_seed + call_idx, is_target_complete_fn)
            raw_act = res.get("action")
            if not raw_act:
                text = res.get("text", "")
                done, extracted = is_target_complete_fn(text)
                raw_act = extracted if done else text
            canon = canonicalize_action(raw_act or "")
            raw_actions.append(raw_act or "")
            canonical_actions.append(canon)
            tokens_per_call.append(res.get("tokens", 0))
            full_texts.append(res.get("text", ""))
            call_idx += 1

            counts = defaultdict(int)
            for a in canonical_actions:
                counts[a] += 1
            mode_action = max(counts, key=lambda a: (counts[a], -canonical_actions.index(a)))
            mode_count = counts[mode_action]
            agreement = mode_count / len(canonical_actions)

            if agreement >= self.tau_high:
                early_exit = True
                break

        elapsed = time.perf_counter() - start_time
        total_tokens_used = sum(tokens_per_call)
        unpruned_baseline_tokens = self.k_max * self.max_tokens_budget
        token_savings_pct = max(0.0, ((unpruned_baseline_tokens - total_tokens_used) / unpruned_baseline_tokens) * 100)

        return {
            "condition": self.name,
            "action": mode_action,
            "raw_actions": raw_actions,
            "canonical_actions": canonical_actions,
            "calls": len(canonical_actions),
            "agreement": agreement,
            "tokens_per_call": tokens_per_call,
            "total_tokens": total_tokens_used,
            "token_savings_pct": round(token_savings_pct, 2),
            "early_exit": early_exit and (len(canonical_actions) < self.k_max),
            "latency_seconds": elapsed,
        }


class ConfidenceWeightedController:
    """
    Extension 2: Confidence/Logprob-Weighted Adaptive Compute.
    
    Standard TrACE gives each rollout an equal unweighted vote of 1.0.
    In this extension:
    1. Each rollout i yields an action a_i and confidence weight w_i in (0, 1] 
       (derived from length-normalized token log-likelihood or calibrated confidence).
    2. Weighted consensus is computed as:
           W(a) = sum_{i: a_i = a} w_i
           alpha_w = W(a*) / sum_{j=1}^k w_j
    3. Plurality action is a* = argmax_a W(a)
    4. Early exit occurs when alpha_w >= tau_high.
    """
    def __init__(
        self,
        k_init: int = 2,
        k_max: int = 4,
        tau_high: float = 0.75,
        min_exit_confidence: float = 0.50,
        temperature: float = 0.7,
        name: Optional[str] = None
    ):
        self.k_init = k_init
        self.k_max = k_max
        self.tau_high = tau_high
        self.min_exit_confidence = min_exit_confidence
        self.temperature = temperature
        self.name = name or f"conf_weighted_{k_max}"

    def compute_weighted_agreement(
        self,
        canonical_actions: List[str],
        confidences: List[float]
    ) -> Tuple[str, float, float, Dict[str, float]]:
        """
        Compute weighted action plurality and normalized agreement.
        Returns: (best_action, weighted_agreement, best_action_confidence, action_weights_dict)
        """
        if not canonical_actions:
            return "", 0.0, 0.0, {}

        action_weights = defaultdict(float)
        action_conf_lists = defaultdict(list)
        for act, conf in zip(canonical_actions, confidences):
            weight = max(float(conf), 1e-4)
            action_weights[act] += weight
            action_conf_lists[act].append(weight)

        total_weight = sum(action_weights.values())
        if total_weight <= 0:
            return canonical_actions[0], 0.0, 0.0, {}

        # Plurality with deterministic tie-breaking by first occurrence
        best_action = max(
            action_weights.keys(),
            key=lambda a: (action_weights[a], -canonical_actions.index(a))
        )
        weighted_agreement = action_weights[best_action] / total_weight
        best_avg_conf = sum(action_conf_lists[best_action]) / len(action_conf_lists[best_action])

        return best_action, weighted_agreement, best_avg_conf, dict(action_weights)

    def execute(
        self,
        sample_fn: Callable[[float, int], Dict[str, Any]],
        base_seed: int = 0
    ) -> Dict[str, Any]:
        """
        Execute confidence-weighted controller.
        
        Args:
            sample_fn: Function (temperature, seed) -> {"action": str, "confidence": float}
            base_seed: Deterministic seed base.
        """
        raw_actions = []
        canonical_actions = []
        confidences = []
        start_time = time.perf_counter()

        # Step 1: Draw k_init samples with confidence scores
        for i in range(self.k_init):
            res = sample_fn(self.temperature, base_seed + i)
            raw = res.get("action", "")
            canon = canonicalize_action(raw)
            conf = res.get("confidence", 1.0)
            raw_actions.append(raw)
            canonical_actions.append(canon)
            confidences.append(float(conf))

        # Step 2: Compute initial weighted agreement
        mode_action, agreement, best_conf, weights = self.compute_weighted_agreement(canonical_actions, confidences)

        # Step 3: Check early exit and expand if needed
        # Early exit requires both high agreement AND confidence above min_exit_confidence
        early_exit = (agreement >= self.tau_high) and (best_conf >= self.min_exit_confidence)
        call_idx = self.k_init

        while not early_exit and len(canonical_actions) < self.k_max:
            res = sample_fn(self.temperature, base_seed + call_idx)
            raw = res.get("action", "")
            canon = canonicalize_action(raw)
            conf = res.get("confidence", 1.0)
            raw_actions.append(raw)
            canonical_actions.append(canon)
            confidences.append(float(conf))
            call_idx += 1

            mode_action, agreement, best_conf, weights = self.compute_weighted_agreement(canonical_actions, confidences)
            if (agreement >= self.tau_high) and (best_conf >= self.min_exit_confidence):
                early_exit = True
                break

        elapsed = time.perf_counter() - start_time

        return {
            "condition": self.name,
            "action": mode_action,
            "raw_actions": raw_actions,
            "canonical_actions": canonical_actions,
            "confidences": confidences,
            "action_weights": weights,
            "calls": len(canonical_actions),
            "agreement": agreement,
            "early_exit": early_exit and (len(canonical_actions) < self.k_max),
            "latency_seconds": elapsed,
        }
