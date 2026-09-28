import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import json
import sys
import time
from datetime import datetime, timezone

from trace_core.generation import LocalGenerator
from trace_core.gsm8k_runner import run_gsm8k_evaluation
from trace_core.minihouse_runner import run_minihouse_evaluation

def print_table_1_comparison(gsm8k_summary, minihouse_summary):
    """Print comparative table matching Table 1 from Section 5.2 of the paper."""
    print("\n" + "="*80)
    print("REPRODUCTION OF TABLE 1: MAIN RESULTS (ACCURACY & CALL EFFICIENCY)")
    print("="*80)
    print(f"{'Condition':<12} | {'GSM8K Acc':<10} | {'GSM8K Calls':<12} | {'Reduction':<10} | {'MiniHouse Acc':<14} | {'MiniHouse Calls':<15} | {'Reduction':<10}")
    print("-" * 88)

    # Paper reference numbers from Section 5.2:
    paper_refs = {
        "greedy": {"gsm_acc": "-", "gsm_calls": "1.00", "gsm_red": "-", "mh_acc": "~0.367", "mh_calls": "-", "mh_red": "-"},
        "sc4":    {"gsm_acc": "0.82", "gsm_calls": "4.00", "gsm_red": "baseline", "mh_acc": "0.367", "mh_calls": "24.67", "mh_red": "baseline"},
        "trace4": {"gsm_acc": "0.82", "gsm_calls": "2.68", "gsm_red": "-33.0%", "mh_acc": "0.367", "mh_calls": "15.07", "mh_red": "-39.0%"},
        "sc8":    {"gsm_acc": "0.84", "gsm_calls": "8.00", "gsm_red": "baseline", "mh_acc": "0.367", "mh_calls": "46.93", "mh_red": "baseline"},
        "trace8": {"gsm_acc": "0.84", "gsm_calls": "3.56", "gsm_red": "-55.5%", "mh_acc": "0.367", "mh_calls": "16.27", "mh_red": "-65.3%"},
    }

    all_conds = ["greedy", "sc4", "trace4", "sc8", "trace8"]
    for cond in all_conds:
        # GSM8K empirical
        gsm_data = gsm8k_summary.get(cond, {}) if gsm8k_summary else {}
        g_acc = f"{gsm_data.get('accuracy', '-'):.3f}" if 'accuracy' in gsm_data else "-"
        g_calls = f"{gsm_data.get('mean_calls', '-'):.2f}" if 'mean_calls' in gsm_data else "-"
        g_red = f"-{gsm_data.get('call_reduction_pct', 0):.1f}%" if 'call_reduction_pct' in gsm_data else ("baseline" if "sc" in cond else "-")

        # MiniHouse empirical
        mh_data = minihouse_summary.get(cond, {}) if minihouse_summary else {}
        m_acc = f"{mh_data.get('success_rate', '-'):.3f}" if 'success_rate' in mh_data else "-"
        m_calls = f"{mh_data.get('mean_calls', '-'):.2f}" if 'mean_calls' in mh_data else "-"
        m_red = f"-{mh_data.get('call_reduction_pct', 0):.1f}%" if 'call_reduction_pct' in mh_data else ("baseline" if "sc" in cond else "-")

        print(f"{cond:<12} | {g_acc:<10} | {g_calls:<12} | {g_red:<10} | {m_acc:<14} | {m_calls:<15} | {m_red:<10}")

    print("-" * 88)
    print("\nPAPER REFERENCE TARGETS (Qwen 2.5 3B from Section 5.2):")
    print("  • SC-4:    GSM8K Acc: 0.82 | Calls: 4.00    || MiniHouse Acc: 0.367 | Calls: 24.67")
    print("  • TrACE-4: GSM8K Acc: 0.82 | Calls: 2.68 (-33.0%) || MiniHouse Acc: 0.367 | Calls: 15.07 (-39.0%)")
    print("  • SC-8:    GSM8K Acc: 0.84 | Calls: 8.00    || MiniHouse Acc: 0.367 | Calls: 46.93")
    print("  • TrACE-8: GSM8K Acc: 0.84 | Calls: 3.56 (-55.5%) || MiniHouse Acc: 0.367 | Calls: 16.27 (-65.3%)")
    print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Full Reproduction Suite for TrACE (arXiv:2604.08369)")
    parser.add_argument("--backend", choices=["local", "groq"], default="local", help="Inference backend: 'local' (HuggingFace CPU/GPU) or 'groq' (Groq Cloud API)")
    parser.add_argument("--groq-api-key", default=None, help="Groq API key (defaults to GROQ_API_KEY environment variable)")
    parser.add_argument("--benchmark", choices=["gsm8k", "minihouse", "all"], default="all")
    parser.add_argument("--model", default=None, help="Model name. Default: Qwen/Qwen2.5-0.5B-Instruct for local, llama-3.1-8b-instant for groq")
    parser.add_argument("--gsm8k-tasks", type=int, default=50, help="Number of GSM8K tasks (paper uses 50)")
    parser.add_argument("--minihouse-tasks", type=int, default=30, help="Number of MiniHouse tasks (paper uses 30)")
    parser.add_argument("--seeds", default="0", help="Comma-separated random seeds (paper uses 0 for GSM8K, 0,1,2 for MiniHouse)")
    parser.add_argument("--conditions", default="greedy,sc4,sc8,trace4,trace8")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--num-threads", type=int, default=8, help="CPU threads for PyTorch inference (paper uses 8)")
    args = parser.parse_args()

    # Default model selection based on backend
    if args.model is None:
        args.model = "llama-3.1-8b-instant" if args.backend == "groq" else "Qwen/Qwen2.5-0.5B-Instruct"

    os.makedirs(args.output_dir, exist_ok=True)
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    print("\n" + "#"*70)
    print("STARTING FULL INDEPENDENT REPRODUCTION OF TrACE")
    print(f"Paper: 'Don't Overthink It: Inter-Rollout Action Agreement as a Free Adaptive-Compute Signal'")
    print(f"Backend: {args.backend}")
    print(f"Model: {args.model}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Conditions: {conditions}")
    print(f"Seeds: {seeds}")
    print("#"*70 + "\n")

    if args.backend == "groq":
        from trace_core.generation import GroqGenerator
        print(f"Initializing GroqGenerator for {args.model} via Groq API...")
        generator = GroqGenerator(model_name=args.model, api_key=args.groq_api_key)
        print("GroqGenerator initialized successfully.")
    else:
        print(f"Initializing LocalGenerator for {args.model} with {args.num_threads} CPU threads...")
        generator = LocalGenerator(model_name=args.model, num_threads=args.num_threads)
        print("Generator initialized successfully.")

    gsm8k_final_summary = None
    minihouse_final_summary = None

    # 1. Run GSM8K Benchmark
    if args.benchmark in ["gsm8k", "all"]:
        print(f"\n[1/2] Running GSM8K Benchmark (n={args.gsm8k_tasks}, seed={seeds[0]})...")
        def gsm_progress(cur, total):
            print(f"  GSM8K Progress: {cur}/{total} problems completed...", end="\r", flush=True)

        gsm_res = run_gsm8k_evaluation(
            generator=generator,
            n_tasks=args.gsm8k_tasks,
            seed=seeds[0],
            conditions=conditions,
            progress_callback=gsm_progress
        )
        print(f"\n  GSM8K completed in full.")
        gsm8k_final_summary = gsm_res["summary"]

        gsm8k_output_path = os.path.join(args.output_dir, "gsm8k_results.json")
        with open(gsm8k_output_path, "w", encoding="utf-8") as f:
            json.dump(gsm_res, f, indent=2)
        print(f"  Saved GSM8K detailed results to {gsm8k_output_path}")
    elif os.path.exists(os.path.join(args.output_dir, "gsm8k_results.json")):
        with open(os.path.join(args.output_dir, "gsm8k_results.json"), "r", encoding="utf-8") as f:
            gsm8k_final_summary = json.load(f).get("summary")
        print(f"  Loaded completed GSM8K results from {os.path.join(args.output_dir, 'gsm8k_results.json')}")

    # 2. Run MiniHouse Benchmark
    if args.benchmark in ["minihouse", "all"]:
        print(f"\n[2/2] Running MiniHouse Benchmark (n={args.minihouse_tasks}, seeds={seeds})...")
        # For multiple seeds, aggregate results
        all_seed_results = []
        for s in seeds:
            print(f"  Running MiniHouse seed {s}...")
            def mh_progress(cur, total):
                print(f"    MiniHouse Task: {cur}/{total} completed...", flush=True)

            mh_res = run_minihouse_evaluation(
                generator=generator,
                n_tasks=args.minihouse_tasks,
                seed=s,
                conditions=conditions,
                progress_callback=mh_progress
            )
            print()
            all_seed_results.append(mh_res)

        # Average across seeds
        minihouse_final_summary = {}
        for cond in conditions:
            succ_avg = sum(sr["summary"][cond]["success_rate"] for sr in all_seed_results) / len(all_seed_results)
            calls_avg = sum(sr["summary"][cond]["mean_calls"] for sr in all_seed_results) / len(all_seed_results)
            steps_avg = sum(sr["summary"][cond]["mean_steps"] for sr in all_seed_results) / len(all_seed_results)
            minihouse_final_summary[cond] = {
                "success_rate": round(succ_avg, 4),
                "mean_calls": round(calls_avg, 3),
                "mean_steps": round(steps_avg, 2),
            }

        if "sc4" in minihouse_final_summary and "trace4" in minihouse_final_summary:
            sc4_calls = minihouse_final_summary["sc4"]["mean_calls"]
            tr4_calls = minihouse_final_summary["trace4"]["mean_calls"]
            red_4 = ((sc4_calls - tr4_calls) / sc4_calls) * 100 if sc4_calls else 0.0
            minihouse_final_summary["trace4"]["call_reduction_pct"] = round(red_4, 1)

        if "sc8" in minihouse_final_summary and "trace8" in minihouse_final_summary:
            sc8_calls = minihouse_final_summary["sc8"]["mean_calls"]
            tr8_calls = minihouse_final_summary["trace8"]["mean_calls"]
            red_8 = ((sc8_calls - tr8_calls) / sc8_calls) * 100 if sc8_calls else 0.0
            minihouse_final_summary["trace8"]["call_reduction_pct"] = round(red_8, 1)

        minihouse_output_path = os.path.join(args.output_dir, "minihouse_results.json")
        with open(minihouse_output_path, "w", encoding="utf-8") as f:
            json.dump({"summary": minihouse_final_summary, "seed_runs": all_seed_results}, f, indent=2)
        print(f"  Saved MiniHouse detailed results to {minihouse_output_path}")

    # Print Table 1 Comparison
    print_table_1_comparison(gsm8k_final_summary, minihouse_final_summary)

    # Save overall reproduction summary
    summary_path = os.path.join(args.output_dir, "table1_reproduction_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "gsm8k": gsm8k_final_summary,
            "minihouse": minihouse_final_summary,
            "paper_reference_gsm8k": {
                "sc4": {"accuracy": 0.82, "mean_calls": 4.00},
                "trace4": {"accuracy": 0.82, "mean_calls": 2.68, "call_reduction_pct": 33.0},
                "sc8": {"accuracy": 0.84, "mean_calls": 8.00},
                "trace8": {"accuracy": 0.84, "mean_calls": 3.56, "call_reduction_pct": 55.5},
            },
            "paper_reference_minihouse": {
                "sc4": {"success_rate": 0.367, "mean_calls": 24.67},
                "trace4": {"success_rate": 0.367, "mean_calls": 15.07, "call_reduction_pct": 39.0},
                "sc8": {"success_rate": 0.367, "mean_calls": 46.93},
                "trace8": {"success_rate": 0.367, "mean_calls": 16.27, "call_reduction_pct": 65.3},
            }
        }, f, indent=2)
    print(f"All reproduction artifacts successfully saved to {summary_path}")

if __name__ == "__main__":
    main()
