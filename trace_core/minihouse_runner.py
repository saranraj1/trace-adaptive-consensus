import time
from typing import Dict, Any, List, Optional
from trace_core.minihouse import MiniHouseEnv, generate_minihouse_tasks
from trace_core.controller import GreedyController, SelfConsistencyController, TrACEController
from trace_core.canonicalizer import canonicalize_action

def run_minihouse_evaluation(
    generator,
    n_tasks: int = 30,
    seed: int = 0,
    conditions: Optional[List[str]] = None,
    max_steps: int = 15,
    progress_callback = None
) -> Dict[str, Any]:
    """
    Evaluate all 5 conditions on MiniHouse benchmark:
    greedy, sc4, sc8, trace4, trace8.
    Matches paper Section 4.2 and Section 5.1-5.3.
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

    tasks = generate_minihouse_tasks(n_tasks=n_tasks, seed=seed)
    results = {
        cond: {
            "successes": 0,
            "total_calls": 0,
            "total_steps": 0,
            "episodes": [],
            "step_agreements": []
        }
        for cond in controllers
    }

    for t_idx, task_info in enumerate(tasks):
        for cond_name, ctrl in controllers.items():
            env = MiniHouseEnv(
                template_id=task_info["template_id"],
                target_object=task_info["target_object"],
                target_receptacle=task_info["target_receptacle"]
            )
            history = []
            obs = env.get_observation()
            episode_calls = 0
            episode_steps = 0
            step_records = []
            done = False
            success = False

            while not done and episode_steps < max_steps:
                episode_steps += 1
                prompt = env.render_prompt(history)

                # Sampling function for this timestep
                def sample_action(temp: float, s: int) -> str:
                    run_seed = seed * 100000 + t_idx * 1000 + episode_steps * 50 + s
                    res = generator.generate(prompt, temperature=temp, max_new_tokens=64, seed=run_seed)
                    raw_text = res["text"].strip().split("\n")[0]
                    return canonicalize_action(raw_text)

                exec_res = ctrl.execute(sample_action, base_seed=0)
                action_chosen = exec_res["action"]
                episode_calls += exec_res["calls"]

                step_records.append({
                    "step": episode_steps,
                    "action": action_chosen,
                    "calls": exec_res["calls"],
                    "agreement": exec_res["agreement"],
                })

                next_obs, success, done = env.step(action_chosen)
                history.append((action_chosen, obs))
                obs = next_obs

            results[cond_name]["successes"] += int(success)
            results[cond_name]["total_calls"] += episode_calls
            results[cond_name]["total_steps"] += episode_steps
            results[cond_name]["episodes"].append({
                "task_id": task_info["task_id"],
                "success": success,
                "steps": episode_steps,
                "calls": episode_calls,
                "step_details": step_records
            })
            for sr in step_records:
                results[cond_name]["step_agreements"].append({
                    "agreement": sr["agreement"],
                    "eventual_success": success
                })

        if progress_callback:
            progress_callback(t_idx + 1, len(tasks))

        # Checkpoint running progress to disk
        try:
            import json, os
            os.makedirs("results", exist_ok=True)
            cur_n = t_idx + 1
            cur_summary = {}
            for c_name in controllers:
                s_rate = results[c_name]["successes"] / max(cur_n, 1)
                m_calls = results[c_name]["total_calls"] / max(cur_n, 1)
                m_steps = results[c_name]["total_steps"] / max(cur_n, 1)
                cur_summary[c_name] = {
                    "success_rate": round(s_rate, 4),
                    "mean_calls": round(m_calls, 3),
                    "total_calls": results[c_name]["total_calls"],
                    "mean_steps": round(m_steps, 2),
                }
            if "sc4" in cur_summary and "trace4" in cur_summary and cur_summary["sc4"]["mean_calls"] > 0:
                cur_summary["trace4"]["call_reduction_pct"] = round(
                    ((cur_summary["sc4"]["mean_calls"] - cur_summary["trace4"]["mean_calls"]) / cur_summary["sc4"]["mean_calls"]) * 100, 1
                )
            if "sc8" in cur_summary and "trace8" in cur_summary and cur_summary["sc8"]["mean_calls"] > 0:
                cur_summary["trace8"]["call_reduction_pct"] = round(
                    ((cur_summary["sc8"]["mean_calls"] - cur_summary["trace8"]["mean_calls"]) / cur_summary["sc8"]["mean_calls"]) * 100, 1
                )
            with open(os.path.join("results", "minihouse_results.json"), "w", encoding="utf-8") as f:
                json.dump({"summary": cur_summary, "per_episode_results": results, "n_tasks": cur_n}, f, indent=2)
        except Exception:
            pass

    # Summary table
    n = len(tasks)
    summary = {}
    for cond_name in controllers:
        succ_rate = results[cond_name]["successes"] / max(n, 1)
        mean_calls = results[cond_name]["total_calls"] / max(n, 1)
        mean_steps = results[cond_name]["total_steps"] / max(n, 1)
        summary[cond_name] = {
            "success_rate": round(succ_rate, 4),
            "mean_calls": round(mean_calls, 3),
            "total_calls": results[cond_name]["total_calls"],
            "mean_steps": round(mean_steps, 2),
        }

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

    return {"summary": summary, "per_episode_results": results, "n_tasks": n}
