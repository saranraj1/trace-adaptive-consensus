import argparse, json, os, random, platform, sys
from datetime import datetime, timezone
from trace_core.answers import extract_answer, gold_from_gsm8k, is_correct, select_majority
from trace_core.features import response_features, agreement_features

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--output", default="runs/pilot.jsonl")
    ap.add_argument("--device", default=None)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--split", default="train")
    ap.add_argument("--train-fraction", type=float, default=0.6)
    ap.add_argument("--validation-fraction", type=float, default=0.2)
    args = ap.parse_args()

    if args.limit < 1:
        raise ValueError("--limit must be at least 1.")
    if args.temperature < 0:
        raise ValueError("--temperature cannot be negative.")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1.")
    if args.train_fraction <= 0 or args.validation_fraction <= 0 or args.train_fraction + args.validation_fraction >= 1:
        raise ValueError("Fractions must be positive and sum to less than 1.")
    # Load heavyweight dependencies only after argparse, so --help works without ML packages installed.
    try:
        import numpy as np
        import sklearn, transformers, torch
        from datasets import load_dataset
        from trace_core.generation import LocalGenerator
        from trace_core.evaluation import (
            FEATURE_SETS, fit_model, probabilities, metric_report,
            choose_threshold, policy_report, split_rows
        )
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency: {exc}. Install project dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    random.seed(args.seed)
    np.random.seed(args.seed)

    ds = load_dataset("gsm8k", "main", split=args.split)
    indices = list(range(len(ds)))
    random.Random(args.seed).shuffle(indices)
    indices = indices[:min(args.limit, len(indices))]
    ds = ds.select(indices)

    gen = LocalGenerator(args.model, args.device, args.revision)
    rows = []
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as out:
        for j, item in enumerate(ds):
            prompt = gen.render_prompt(item["question"])
            generations = []
            parsed, methods = [], []
            # Initial state: 3 independent trajectories.
            for k in range(3):
                g = gen.generate(prompt, args.temperature, args.max_new_tokens, args.seed + j*11 + k)
                a, method = extract_answer(g["text"])
                generations.append(g); parsed.append(a); methods.append(method)

            gold = gold_from_gsm8k(item["answer"])
            baseline_answer = select_majority(parsed)
            baseline_correct = is_correct(baseline_answer, gold)

            # Signals available before action: first 3 generations only.
            per_response = [response_features(item["question"], g["text"]) for g in generations]
            feats = {
                key: float(np.mean([f[key] for f in per_response]))
                for key in per_response[0]
            }
            feats.update(agreement_features(parsed))

            # Action: generate 2 more independent samples.
            extra_generations, extra_answers, extra_methods = [], [], []
            for k in range(3, 5):
                g = gen.generate(prompt, args.temperature, args.max_new_tokens, args.seed + j*11 + k)
                a, method = extract_answer(g["text"])
                extra_generations.append(g); extra_answers.append(a); extra_methods.append(method)

            post_answer = select_majority(parsed + extra_answers)
            post_correct = is_correct(post_answer, gold)
            all_gens = generations + extra_generations
            row = dict(feats)
            row.update({
                "problem_id": int(indices[j]),
                "source_split": args.split,
                "question": item["question"],
                "gold": gold,
                "answers_initial": parsed,
                "answers_additional": extra_answers,
                "extraction_methods_initial": methods,
                "extraction_methods_additional": extra_methods,
                "baseline_answer": baseline_answer,
                "post_action_answer": post_answer,
                "baseline_correct": baseline_correct,
                "post_action_correct": post_correct,
                "utility_delta": post_correct - baseline_correct,
                "utility_positive": int(post_correct > baseline_correct),
                "initial_output_tokens": sum(g["output_tokens"] for g in generations),
                "additional_output_tokens": sum(g["output_tokens"] for g in extra_generations),
                "total_output_tokens": sum(g["output_tokens"] for g in all_gens),
                "initial_latency_seconds": sum(g["latency_seconds"] for g in generations),
                "total_latency_seconds": sum(g["latency_seconds"] for g in all_gens),
                "responses_initial": [g["text"] for g in generations],
                "responses_additional": [g["text"] for g in extra_generations],
                "model": args.model,
                "model_revision": args.revision,
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
                "seed_base": args.seed + j*11,
            })
            rows.append(row)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(f"{j+1}/{len(ds)} baseline={baseline_correct} post={post_correct} delta={post_correct-baseline_correct}")

    train, val, test = split_rows(rows, args.seed, args.train_fraction, args.validation_fraction)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "gsm8k",
        "dataset_config": "main",
        "source_split": args.split,
        "selected_problem_ids": indices,
        "model": args.model,
        "model_revision": args.revision,
        "hardware_device": str(next(gen.model.parameters()).device),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "seed": args.seed,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "action": "add_two_independent_samples_after_three_initial_samples",
        "split_sizes": {"train": len(train), "validation": len(val), "test": len(test)},
        "initial_accuracy_all": float(np.mean([r["baseline_correct"] for r in rows])) if rows else None,
        "post_action_accuracy_all": float(np.mean([r["post_action_correct"] for r in rows])) if rows else None,
        "utility_positive_rate_all": float(np.mean([r["utility_positive"] for r in rows])) if rows else None,
        "predictors": {},
        "policy_comparison_test": {},
        "limitations": [
            "Pilot subset and stochastic outcomes; repeat seeds for robust inference.",
            "Text features are proxies, not token entropy or latent-state dynamics.",
            "No learned verifier is included.",
            "Predictor runtime is not separately profiled.",
            "No claim of novelty or state-of-the-art performance is made."
        ]
    }

    y_val = [r["utility_positive"] for r in val]
    y_test = [r["utility_positive"] for r in test]
    for name, columns in FEATURE_SETS.items():
        model = fit_model(train, columns)
        if model is None or not val or not test:
            summary["predictors"][name] = {"status": "insufficient data or one target class in training"}
            continue
        p_val = probabilities(model, val, columns)
        threshold = choose_threshold(y_val, p_val)
        p_test = probabilities(model, test, columns)
        summary["predictors"][name] = {
            "threshold_selected_on_validation": threshold,
            "validation_metrics": metric_report(y_val, p_val, threshold),
            "test_metrics": metric_report(y_test, p_test, threshold),
        }
        cont = p_test >= threshold
        summary["policy_comparison_test"][name] = policy_report(test, cont)

    if test:
        summary["policy_comparison_test"]["always_stop_after_three"] = policy_report(test, [False]*len(test))
        summary["policy_comparison_test"]["always_add_two"] = policy_report(test, [True]*len(test))
        rate = float(np.mean([r["utility_positive"] for r in train])) if train else 0.0
        summary["policy_comparison_test"]["random_allocation_at_train_positive_rate"] = policy_report(
            test, [((i * 1103515245 + args.seed) % 10000) / 10000 < rate for i in range(len(test))]
        )

    summary_path = args.output.rsplit(".", 1)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved: {args.output} and {summary_path}")

if __name__ == "__main__":
    main()
