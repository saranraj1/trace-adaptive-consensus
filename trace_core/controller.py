from collections import Counter
from typing import Callable, List, Dict, Any, Optional, Tuple
from trace_core.canonicalizer import canonicalize_action

def select_mode(items: List[str]) -> Tuple[str, int]:
    """Return (mode_item, count) with deterministic tie-breaking (first occurrence)."""
    if not items:
        return "", 0
    counts = Counter(items)
    highest_count = max(counts.values())
    tied = {item for item, c in counts.items() if c == highest_count}
    # Deterministic tie-break by first appearance in items
    mode_item = next(item for item in items if item in tied)
    return mode_item, highest_count

class GreedyController:
    """Greedy decoding baseline (temperature=0, k=1)."""
    def __init__(self):
        self.name = "greedy"

    def execute(self, sample_fn: Callable[[float, int], str], base_seed: int = 0) -> Dict[str, Any]:
        """Calls sample_fn(temperature, seed) once."""
        raw = sample_fn(0.0, base_seed)
        canonical = canonicalize_action(raw)
        return {
            "condition": "greedy",
            "action": canonical,
            "raw_actions": [raw],
            "canonical_actions": [canonical],
            "calls": 1,
            "agreement": 1.0,
            "early_exit": False,
        }

class SelfConsistencyController:
    """Fixed-budget Self-Consistency baseline (SC-k)."""
    def __init__(self, k: int, temperature: float = 0.7):
        self.k = k
        self.temperature = temperature
        self.name = f"sc{k}"

    def execute(self, sample_fn: Callable[[float, int], str], base_seed: int = 0) -> Dict[str, Any]:
        raw_actions = []
        canonical_actions = []
        for i in range(self.k):
            raw = sample_fn(self.temperature, base_seed + i)
            raw_actions.append(raw)
            canonical_actions.append(canonicalize_action(raw))

        mode_action, mode_count = select_mode(canonical_actions)
        agreement = mode_count / len(canonical_actions)
        return {
            "condition": self.name,
            "action": mode_action,
            "raw_actions": raw_actions,
            "canonical_actions": canonical_actions,
            "calls": self.k,
            "agreement": agreement,
            "early_exit": False,
        }

class TrACEController:
    """
    TrACE (Trajectorical Adaptive Compute via agrEement) Controller:
    Described in Section 3.2:
    Step 1: Draw k_init candidate actions at temperature tau.
    Step 2: Compute plurality a* and agreement alpha = count(a*) / |A_t|.
    Step 3: If alpha >= tau_high: commit immediately.
            Else: sample one additional candidate at a time, recompute alpha,
            and repeat until either alpha >= tau_high or |A_t| == k_max.
    """
    def __init__(self, k_init: int = 2, k_max: int = 4, tau_high: float = 0.75, temperature: float = 0.7):
        self.k_init = k_init
        self.k_max = k_max
        self.tau_high = tau_high
        self.temperature = temperature
        self.name = f"trace{k_max}"

    def execute(self, sample_fn: Callable[[float, int], str], base_seed: int = 0) -> Dict[str, Any]:
        raw_actions = []
        canonical_actions = []

        # Step 1: Draw initial k_init samples
        for i in range(self.k_init):
            raw = sample_fn(self.temperature, base_seed + i)
            raw_actions.append(raw)
            canonical_actions.append(canonicalize_action(raw))

        # Step 2: Compute initial agreement
        mode_action, mode_count = select_mode(canonical_actions)
        agreement = mode_count / len(canonical_actions)

        # Step 3: Check threshold & iteratively expand if needed
        early_exit = agreement >= self.tau_high
        call_idx = self.k_init

        while agreement < self.tau_high and len(canonical_actions) < self.k_max:
            raw = sample_fn(self.temperature, base_seed + call_idx)
            raw_actions.append(raw)
            canonical_actions.append(canonicalize_action(raw))
            call_idx += 1
            mode_action, mode_count = select_mode(canonical_actions)
            agreement = mode_count / len(canonical_actions)
            if agreement >= self.tau_high:
                early_exit = True
                break

        return {
            "condition": self.name,
            "action": mode_action,
            "raw_actions": raw_actions,
            "canonical_actions": canonical_actions,
            "calls": len(canonical_actions),
            "agreement": agreement,
            "early_exit": early_exit and (len(canonical_actions) < self.k_max),
        }
