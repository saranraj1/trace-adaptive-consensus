import random
import time
from typing import Dict, Any, List, Optional
from datasets import load_dataset
from trace_core.canonicalizer import extract_gsm8k_answer, extract_gold_gsm8k
from trace_core.controller import GreedyController, SelfConsistencyController, TrACEController

def load_gsm8k_subset(n_tasks: int = 50, seed: int = 0) -> List[Dict[str, Any]]:
    """Load GSM8K test split subset matching paper Section 4.1 (seed 0, n=50)."""
    ds = load_dataset("gsm8k", "main", split="test")
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    selected_indices = indices[:n_tasks]
    subset = []
    for idx in selected_indices:
        item = ds[idx]
        subset.append({
            "id": idx,
            "question": item["question"],
            "gold_raw": item["answer"],
            "gold_answer": extract_gold_gsm8k(item["answer"]),
        })
    return subset

def render_gsm8k_prompt(tokenizer, question: str) -> str:
    """Format prompt with model chat template if available, else plain text."""
    instruction = (
        "Solve this math word problem step by step. Show concise calculation and "
        "end your response with the final numerical answer as #### number.\n\n"
        f"Question: {question}"
    )
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": instruction}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return instruction + "\nSolution:"

def run_gsm8k_evaluation(
    generator,
    n_tasks: int = 50,
    seed: int = 0,
    conditions: Optional[List[str]] = None,
    progress_callback = None
) -> Dict[str, Any]:
    """
    Evaluate all 5 conditions on GSM8K subset:
    greedy, sc4, sc8, trace4, trace8.
    """
    if conditions is None:
        conditions = ["greedy", "sc4", "sc8", "trace4", "trace8"]

    controllers = {}
    if "greedy" in conditions:
        controllers["greedy"] = GreedyController()
    if "sc4" in conditions:
        controllers["sc4"] = SelfConsistencyController(k=4)
    if "sc8" in conditions:
        controllers["sc8"] = SelfConsistencyController(k=8)
    if "trace4" in conditions:
        controllers["trace4"] = TrACEController(k_init=2, k_max=4, tau_high=0.75)
    if "trace8" in conditions:
        controllers["trace8"] = TrACEController(k_init=2, k_max=8, tau_high=0.75)

    problems = load_gsm8k_subset(n_tasks=n_tasks, seed=seed)
    results = {cond: {"correct": 0, "total_calls": 0, "latencies": [], "details": []} for cond in controllers}

    for p_idx, prob in enumerate(problems):
        gold = prob["gold_answer"]
        prompt = render_gsm8k_prompt(getattr(generator, "tokenizer", None), prob["question"])

        # Cache generations across conditions for efficiency and identical stochastic samples
        # Sample pool for this problem: sample_fn(temperature, seed)
        gen_cache = {}
        def make_sample_fn(prob_id: int):
            def sample(temp: float, s: int) -> str:
                cache_key = (round(temp, 2), s)
                if cache_key not in gen_cache:
                    # Deterministic seed per problem and sample index
                    run_seed = seed * 10000 + prob_id * 100 + s
                    out = generator.generate(prompt, temperature=temp, seed=run_seed)
                    parsed_ans, method = extract_gsm8k_answer(out["text"])
                    gen_cache[cache_key] = (parsed_ans or "", out)
                return gen_cache[cache_key][0]
            return sample

        sample_fn = make_sample_fn(p_idx)

        for cond_name, ctrl in controllers.items():
            t0 = time.perf_counter()
            exec_res = ctrl.execute(sample_fn, base_seed=0)
            elapsed = time.perf_counter() - t0

            predicted_answer = exec_res["action"]
            is_correct = int(predicted_answer is not None and gold is not None and predicted_answer == gold)

            results[cond_name]["correct"] += is_correct
            results[cond_name]["total_calls"] += exec_res["calls"]
            results[cond_name]["latencies"].append(elapsed)
            results[cond_name]["details"].append({
                "problem_id": prob["id"],
                "gold": gold,
                "predicted": predicted_answer,
                "correct": is_correct,
                "calls": exec_res["calls"],
                "agreement": exec_res["agreement"],
                "early_exit": exec_res.get("early_exit", False),
            })

        if progress_callback:
            progress_callback(p_idx + 1, len(problems))

    # Compute summary metrics
    summary = {}
    n = len(problems)
    for cond_name in controllers:
        acc = results[cond_name]["correct"] / max(n, 1)
        mean_calls = results[cond_name]["total_calls"] / max(n, 1)
        mean_lat = sum(results[cond_name]["latencies"]) / max(len(results[cond_name]["latencies"]), 1)
        summary[cond_name] = {
            "accuracy": round(acc, 4),
            "mean_calls": round(mean_calls, 3),
            "total_calls": results[cond_name]["total_calls"],
            "mean_latency_seconds": round(mean_lat, 3),
        }

    # Compute reductions relative to SC counterparts
    if "sc4" in summary and "trace4" in summary:
        sc4_calls = summary["sc4"]["mean_calls"]
        tr4_calls = summary["trace4"]["mean_calls"]
        reduction_4 = ((sc4_calls - tr4_calls) / sc4_calls) * 100 if sc4_calls else 0.0
        summary["trace4"]["call_reduction_pct"] = round(reduction_4, 1)

    if "sc8" in summary and "trace8" in summary:
        sc8_calls = summary["sc8"]["mean_calls"]
        tr8_calls = summary["trace8"]["mean_calls"]
        reduction_8 = ((sc8_calls - tr8_calls) / sc8_calls) * 100 if sc8_calls else 0.0
        summary["trace8"]["call_reduction_pct"] = round(reduction_8, 1)

    return {"summary": summary, "per_problem_results": results, "n_tasks": n}
