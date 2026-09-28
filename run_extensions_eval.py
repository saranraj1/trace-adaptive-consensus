"""
TrACE Advanced Research Extensions Evaluation Suite
Evaluates:
- Baseline TrACE
- Extension 1: Token-Prefix Early Pruning (StreamingPrefixController)
- Extension 2: Confidence-Weighted Agreement (ConfidenceWeightedController)
"""

import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import json
import time
import re
from typing import Dict, Any, List

from trace_core.canonicalizer import extract_gsm8k_answer, extract_gold_gsm8k, canonicalize_action
from trace_core.controller import TrACEController, SelfConsistencyController
from trace_core.extensions import StreamingPrefixController, ConfidenceWeightedController
from trace_core.gsm8k_runner import load_gsm8k_subset, render_gsm8k_prompt
from trace_core.generation import GroqGenerator, LocalGenerator


def gsm8k_early_target_checker(text: str):
    """
    Detect if final math answer marker has been completely emitted during streaming.
    Requires a non-digit delimiter (whitespace, newline, punctuation) following the
    number to ensure multi-token digits (e.g. 240, 80) are not truncated mid-stream.
    """
    from trace_core.canonicalizer import NUMBER_PATTERN, normalize_number

    # Pattern 1: #### <number> followed by delimiter
    match1 = re.search(r"####\s*(" + NUMBER_PATTERN + r")(?:[\s.\n\r\)]+)", text)
    if match1:
        return True, normalize_number(match1.group(1))

    # Pattern 2: (Answer:|Final Answer:) <number> followed by delimiter
    match2 = re.search(r"(?:answer|the answer is|final answer)\s*[:=]?\s*(" + NUMBER_PATTERN + r")(?:[\s.\n\r\)]+)", text, re.IGNORECASE)
    if match2:
        return True, normalize_number(match2.group(1))

    return False, None


def run_extensions_benchmark(
    generator,
    n_tasks: int = 5,
    seed: int = 0,
    k_max: int = 4
) -> Dict[str, Any]:
    """
    Run comparative evaluation across:
    1. Self-Consistency (sc4)
    2. Baseline TrACE (trace4)
    3. Extension 1: StreamingPrefixController (sub-trajectory pruning)
    4. Extension 2: ConfidenceWeightedController (confidence-calibrated)
    """
    problems = load_gsm8k_subset(n_tasks=n_tasks, seed=seed)
    
    controllers = {
        "sc4": SelfConsistencyController(k=k_max),
        "trace4": TrACEController(k_init=2, k_max=k_max, tau_high=0.75),
        "ext1_streaming": StreamingPrefixController(k_init=2, k_max=k_max, tau_high=0.75, max_tokens_budget=256),
        "ext2_weighted": ConfidenceWeightedController(k_init=2, k_max=k_max, tau_high=0.75, min_exit_confidence=0.50),
    }

    results = {
        name: {
            "correct": 0,
            "total_calls": 0,
            "total_tokens": 0,
            "total_latency": 0.0,
            "early_exits": 0,
        }
        for name in controllers
    }

    print("\n" + "="*80, flush=True)
    print(f"RUNNING EXTENSIONS BENCHMARK (n={n_tasks} GSM8K Problems, k_max={k_max})", flush=True)
    print("="*80, flush=True)

    # Generation cache to ensure fair comparison and avoid redundant API calls
    gen_cache: Dict[Tuple[str, int, float], Dict[str, Any]] = {}

    for p_idx, prob in enumerate(problems):
        gold = prob["gold_answer"]
        prompt = render_gsm8k_prompt(getattr(generator, "tokenizer", None), prob["question"])
        print(f"Problem {p_idx+1}/{n_tasks} (ID {prob['id']}, Gold={gold})...", flush=True)

        # 1. Base sampling function with caching and token tracking
        sc_tokens_used = []
        trace_tokens_used = []
        conf_tokens_used = []

        def get_or_generate(temp: float, s: int) -> Dict[str, Any]:
            run_seed = seed * 10000 + p_idx * 100 + s
            cache_key = (prompt, run_seed, temp)
            if cache_key not in gen_cache:
                out = generator.generate(prompt, temperature=temp, seed=run_seed)
                ans, _ = extract_gsm8k_answer(out["text"])
                gen_cache[cache_key] = {
                    "text": out["text"],
                    "action": ans or "",
                    "tokens": out.get("output_tokens", 0),
                    "confidence": out.get("confidence", 0.85),
                }
            return gen_cache[cache_key]

        def sample_sc(temp: float, s: int) -> str:
            data = get_or_generate(temp, s)
            sc_tokens_used.append(data["tokens"])
            return data["action"]

        def sample_trace(temp: float, s: int) -> str:
            data = get_or_generate(temp, s)
            trace_tokens_used.append(data["tokens"])
            return data["action"]

        # 2. Confidence sampling function
        def sample_with_conf(temp: float, s: int) -> Dict[str, Any]:
            data = get_or_generate(temp, s)
            conf_tokens_used.append(data["tokens"])
            return {
                "action": data["action"],
                "confidence": data["confidence"],
                "raw": data["text"],
                "tokens": data["tokens"],
            }

        # 3. Streaming sample function with early target detection (generates live to measure prefix pruning)
        def sample_stream(temp: float, s: int, checker) -> Dict[str, Any]:
            run_seed = seed * 10000 + p_idx * 100 + s
            if hasattr(generator, "stream_generate"):
                out = generator.stream_generate(
                    prompt,
                    is_target_complete_fn=checker,
                    temperature=temp,
                    seed=run_seed
                )
                raw_text = out["text"]
                ans, _ = extract_gsm8k_answer(raw_text)
                return {
                    "text": raw_text,
                    "action": out.get("action") or ans or "",
                    "tokens": out.get("tokens", 0)
                }
            else:
                out = generator.generate(prompt, temperature=temp, seed=run_seed)
                raw_text = out["text"]
                ans, _ = extract_gsm8k_answer(raw_text)
                return {
                    "text": raw_text,
                    "action": ans or "",
                    "tokens": out.get("output_tokens", 0)
                }

        # Execute each controller
        # SC-4
        sc_tokens_used.clear()
        t0 = time.perf_counter()
        sc_res = controllers["sc4"].execute(sample_sc, base_seed=0)
        results["sc4"]["correct"] += int(sc_res["action"] == gold)
        results["sc4"]["total_calls"] += sc_res["calls"]
        results["sc4"]["total_tokens"] += sum(sc_tokens_used)
        results["sc4"]["total_latency"] += (time.perf_counter() - t0)

        # Baseline TrACE-4
        trace_tokens_used.clear()
        t0 = time.perf_counter()
        tr_res = controllers["trace4"].execute(sample_trace, base_seed=0)
        results["trace4"]["correct"] += int(tr_res["action"] == gold)
        results["trace4"]["total_calls"] += tr_res["calls"]
        results["trace4"]["total_tokens"] += sum(trace_tokens_used)
        results["trace4"]["early_exits"] += int(tr_res["early_exit"])
        results["trace4"]["total_latency"] += (time.perf_counter() - t0)

        # Extension 1: Streaming Prefix (Intra-call pruning + Inter-rollout early exit)
        t0 = time.perf_counter()
        ext1_res = controllers["ext1_streaming"].execute(sample_stream, gsm8k_early_target_checker, base_seed=0)
        results["ext1_streaming"]["correct"] += int(ext1_res["action"] == gold)
        results["ext1_streaming"]["total_calls"] += ext1_res["calls"]
        results["ext1_streaming"]["total_tokens"] += ext1_res["total_tokens"]
        results["ext1_streaming"]["early_exits"] += int(ext1_res["early_exit"])
        results["ext1_streaming"]["total_latency"] += (time.perf_counter() - t0)

        # Extension 2: Confidence-Weighted
        conf_tokens_used.clear()
        t0 = time.perf_counter()
        ext2_res = controllers["ext2_weighted"].execute(sample_with_conf, base_seed=0)
        results["ext2_weighted"]["correct"] += int(ext2_res["action"] == gold)
        results["ext2_weighted"]["total_calls"] += ext2_res["calls"]
        results["ext2_weighted"]["total_tokens"] += sum(conf_tokens_used)
        results["ext2_weighted"]["early_exits"] += int(ext2_res["early_exit"])
        results["ext2_weighted"]["total_latency"] += (time.perf_counter() - t0)

        print(
            f"  -> Problem {p_idx+1} Done: Gold={gold} | "
            f"SC-4={sc_res['action']} ({sc_res['calls']}c) | "
            f"TrACE-4={tr_res['action']} ({tr_res['calls']}c) | "
            f"Ext1_Stream={ext1_res['action']} ({ext1_res['calls']}c, {ext1_res['total_tokens']}toks) | "
            f"Ext2_Conf={ext2_res['action']} ({ext2_res['calls']}c)",
            flush=True
        )

    # Compile Summary
    summary = {}
    for c_name in controllers:
        acc = results[c_name]["correct"] / max(n_tasks, 1)
        mean_calls = results[c_name]["total_calls"] / max(n_tasks, 1)
        mean_tokens = results[c_name]["total_tokens"] / max(n_tasks, 1)
        summary[c_name] = {
            "accuracy": round(acc, 3),
            "mean_calls": round(mean_calls, 2),
            "mean_tokens": round(mean_tokens, 1),
            "early_exits": results[c_name]["early_exits"],
            "total_latency_seconds": round(results[c_name]["total_latency"], 2),
        }

    # Reductions vs SC-4
    sc_tokens = summary["sc4"]["mean_tokens"]
    sc_calls = summary["sc4"]["mean_calls"]
    for c_name in summary:
        if c_name != "sc4":
            call_red = ((sc_calls - summary[c_name]["mean_calls"]) / sc_calls) * 100
            token_red = ((sc_tokens - summary[c_name]["mean_tokens"]) / sc_tokens) * 100
            summary[c_name]["call_reduction_pct"] = round(call_red, 1)
            summary[c_name]["token_reduction_pct"] = round(token_red, 1)

    print("\n" + "="*85, flush=True)
    print("EXTENSIONS COMPARATIVE EVALUATION RESULTS", flush=True)
    print("="*85, flush=True)
    print(f"{'Condition':<18} | {'Accuracy':<10} | {'Mean Calls':<12} | {'Call Red.':<10} | {'Mean Tokens':<12} | {'Token Red.':<10}", flush=True)
    print("-" * 85, flush=True)
    for c_name in summary:
        d = summary[c_name]
        c_red = f"-{d['call_reduction_pct']}%" if "call_reduction_pct" in d else "baseline"
        t_red = f"-{d['token_reduction_pct']}%" if "token_reduction_pct" in d else "baseline"
        print(f"{c_name:<18} | {d['accuracy']:<10.3f} | {d['mean_calls']:<12.2f} | {c_red:<10} | {d['mean_tokens']:<12.1f} | {t_red:<10}", flush=True)
    print("="*85 + "\n", flush=True)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate TrACE Research Extensions")
    parser.add_argument("--backend", choices=["local", "groq"], default="groq")
    parser.add_argument("--model", default="qwen/qwen3.8-27b")
    parser.add_argument("--tasks", type=int, default=3, help="Number of evaluation problems")
    parser.add_argument("--output", default="results/extensions_results.json")
    args = parser.parse_args()

    if args.backend == "groq":
        generator = GroqGenerator(model_name=args.model)
    else:
        generator = LocalGenerator(model_name=args.model)

    res = run_extensions_benchmark(generator, n_tasks=args.tasks, seed=0)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"Detailed extension metrics saved to {args.output}")


if __name__ == "__main__":
    main()
