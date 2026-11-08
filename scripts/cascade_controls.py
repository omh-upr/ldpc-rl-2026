#!/usr/bin/env python3
"""
Controls for the Min-Sum, agent, Min-Sum cascade (scripts/cascade_minsum_agent.py).

On the same held-out words and the same headline cells, asks whether something
cheaper than the agent does as well where Min-Sum fails:

    1. Other fallbacks at stage 2: Gallager-B and greedy bit flipping, run in the
       environment from the received word on Min-Sum's failures.
    2. Other perturbations before Min-Sum runs again: the first k flips of greedy,
       or k distinct bits flipped uniformly at random (mean over draws), for
       k = 1 to 6. Applied to Min-Sum's failures directly (no agent at all) and
       to the words the agent does not recover (the agent's stage-3 set).
    3. Min-Sum alone with more iterations (50, 100, 200 instead of 20) on
       Min-Sum's failures.

The agent's own numbers are recomputed here for the same sets so the table
compares like with like. Greedy and Gallager-B outcomes are checked against
their recorded evaluations.

Writes docs/ablation/cascade/controls_<timestamp>.json and .md.

Usage:
    PYTHONPATH=. python scripts/cascade_controls.py [--seeds ...] [--draws 20]
"""

import argparse
import copy
import datetime as dt
import json

import numpy as np
import torch

from scripts.analyze_ablation_details import eval_file, load_many, success
from scripts.benchmark_decoders import decode_frame, find_model, load_cell_config, load_test_frames
from scripts.cascade_minsum_agent import CONDITION, K_SWEEP, min_sum_outcome, partial_word
from src.core.baseline_agents import MinSumExpertAgent
from src.core.environment import LDPCEnvironment
from src.core.ldpc_code import LDPC_Code
from src.evaluation.agent_registry import AgentRegistry, AgentSpec
from src.utils.path_utils import resolve_project_path

ITERATIONS = [50, 100, 200]
OUTCOMES = ("recovered", "wrong_codeword", "not_converged")


def perturb_then_min_sum(decoder, H, words, originals, flips_of) -> dict:
    """For each k: outcomes of Min-Sum on each word after flips_of(i, k)."""
    out = {}
    for k in K_SWEEP:
        counts = dict.fromkeys(OUTCOMES, 0)
        for i in words:
            w = originals[i]
            counts[min_sum_outcome(decoder, H, flips_of(i, k), w)] += 1
        out[str(k)] = counts
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--config-root", default="config/ablation_config")
    parser.add_argument("--baseline-root", default="config/baseline_eval")
    parser.add_argument("--cache-root", default="results/ablation_cache")
    parser.add_argument("--output-dir", default="docs/ablation/cascade")
    args = parser.parse_args()
    torch.set_num_threads(1)

    config_root = resolve_project_path(args.config_root)
    cache_root = resolve_project_path(args.cache_root)
    record = {"generated": dt.datetime.now().isoformat(timespec="seconds"),
              "condition": CONDITION, "draws": args.draws, "seeds": {}}

    for seed in args.seeds:
        cell = f"{CONDITION}_s{seed}"
        config = load_cell_config(config_root, cell)
        config["device"] = "cpu"
        code = LDPC_Code.from_default(code_id=config["code_id"])
        H = np.asarray(code.H, dtype=np.int64) % 2
        n = code.n
        env = LDPCEnvironment(code=code, **config["environment"])
        results_dir = resolve_project_path(config["output_paths"]["results_dir"])
        models_dir = resolve_project_path(config["output_paths"]["models_dir"])
        originals, noisies = load_test_frames(resolve_project_path(config["dataset_creation"]["output_path"]))

        registry = AgentRegistry(copy.deepcopy(config), env=env, models_dir=str(models_dir))

        def baseline(name, spec_config):
            return registry.create_agent(AgentSpec(
                name=name, display_name=name, enabled=True,
                agent_class=registry.agent_classes[name], config=spec_config))

        ms_spec = next(b for b in config["evaluation"]["baseline_agents"] if b["name"] == "min_sum_expert")
        decoder = baseline("min_sum_expert", ms_spec).decoder
        greedy = baseline("greedy", {"name": "greedy", "enabled": True})
        gb_dir = resolve_project_path(f"{args.baseline_root}/gallager_b_flat_s{seed}")
        gb_config = load_cell_config(gb_dir.parent, gb_dir.name)
        if gb_config["dataset_creation"]["output_path"] != config["dataset_creation"]["output_path"]:
            raise SystemExit(f"{gb_dir}: scores a different dataset than {cell}")
        gallager = baseline("gallager_b", next(b for b in gb_config["evaluation"]["baseline_agents"]
                                               if b["name"] == "gallager_b"))
        rl = registry.create_agent(AgentSpec(
            name="rl", display_name="RL Agent", enabled=True, agent_class=None, requires_model=True,
            model_path=str(find_model(models_dir)), config=config["training"]))

        # Recorded outcomes, for the checks.
        src = {"greedy": eval_file(results_dir, "greedy"),
               "gallager_b": eval_file(resolve_project_path(gb_config["output_paths"]["results_dir"]),
                                       "gallager_b")}
        rec = load_many([("eval", p) for p in src.values()], cache_root, workers=2)
        recorded = {a: success(rec[str(p)]) for a, p in src.items()}

        # Min-Sum's failures (stage 1), as in the cascade script.
        fail1 = [i for i in range(len(noisies))
                 if min_sum_outcome(decoder, H, noisies[i], originals[i]) != "recovered"]
        print(f"{cell}: Min-Sum fails on {len(fail1)}", flush=True)

        # Stage 2: agent, Gallager-B, greedy; keep the agent's and greedy's trajectories.
        traj, stage2, mismatch = {"rl": {}, "greedy": {}}, {}, {}
        for name, agent in [("rl", rl), ("gallager_b", gallager), ("greedy", greedy)]:
            ok = {}
            for i in fail1:
                reason, actions, _ = decode_frame(agent, env, originals[i], noisies[i])
                ok[i] = reason == "SUCCESS_STOP"
                if name in traj:
                    traj[name][i] = actions
            stage2[name] = int(sum(ok.values()))
            if name in recorded:
                mismatch[name] = int(sum(ok[i] != bool(recorded[name][i]) for i in fail1))
            if name == "rl":
                fail2 = [i for i in fail1 if not ok[i]]
        print(f"  stage 2 on {len(fail1)}: {stage2}; disagreements with records {mismatch}", flush=True)

        # Control 2: perturbations before Min-Sum again, on both sets.
        rng_for = lambda i, d: np.random.default_rng([seed, i, d])

        def random_flips(i, k, d):
            word = noisies[i].copy()
            word[rng_for(i, d).choice(n, size=k, replace=False)] ^= 1
            return word

        def random_mean(words):
            out = {}
            for k in K_SWEEP:
                totals = dict.fromkeys(OUTCOMES, 0)
                for d in range(args.draws):
                    for i in words:
                        totals[min_sum_outcome(decoder, H, random_flips(i, k, d), originals[i])] += 1
                out[str(k)] = {o: totals[o] / args.draws for o in OUTCOMES}
            return out

        perturb = {}
        for set_name, words in [("min_sum_failures", fail1), ("agent_failures", fail2)]:
            perturb[set_name] = {
                "agent": perturb_then_min_sum(decoder, H, words, originals,
                                              lambda i, k: partial_word(noisies[i], traj["rl"][i], k, n)),
                "greedy": perturb_then_min_sum(decoder, H, words, originals,
                                               lambda i, k: partial_word(noisies[i], traj["greedy"][i], k, n)),
                "random": random_mean(words),
            }
            print(f"  perturb then Min-Sum on {set_name} ({len(words)}), recovered by k: "
                  + "; ".join(f"{src_}: " + ", ".join(f"{perturb[set_name][src_][str(k)]['recovered']:g}"
                                                      for k in K_SWEEP)
                              for src_ in ("agent", "greedy", "random")), flush=True)

        # Control 3: Min-Sum alone with more iterations.
        more = {}
        for it in ITERATIONS:
            dec = MinSumExpertAgent(code=code, max_iterations=it).decoder
            counts = dict.fromkeys(OUTCOMES, 0)
            for i in fail1:
                counts[min_sum_outcome(dec, H, noisies[i], originals[i])] += 1
            more[str(it)] = counts
        print(f"  Min-Sum with more iterations on {len(fail1)}: "
              + "; ".join(f"{it}: {more[str(it)]['recovered']}" for it in ITERATIONS), flush=True)

        record["seeds"][str(seed)] = {"min_sum_failures": len(fail1), "agent_failures": len(fail2),
                                      "stage2": stage2, "disagreements_with_record": mismatch,
                                      "perturb_then_min_sum": perturb, "min_sum_iterations": more}

    # Pooled.
    S = list(record["seeds"].values())
    f1 = sum(r["min_sum_failures"] for r in S)
    f2 = sum(r["agent_failures"] for r in S)
    pooled = {
        "min_sum_failures": f1, "agent_failures": f2,
        "stage2": {a: sum(r["stage2"][a] for r in S) for a in ("rl", "gallager_b", "greedy")},
        "perturb_then_min_sum": {
            set_name: {src_: {str(k): {o: sum(r["perturb_then_min_sum"][set_name][src_][str(k)][o] for r in S)
                                       for o in OUTCOMES} for k in K_SWEEP}
                       for src_ in ("agent", "greedy", "random")}
            for set_name in ("min_sum_failures", "agent_failures")},
        "min_sum_iterations": {str(it): {o: sum(r["min_sum_iterations"][str(it)][o] for r in S)
                                         for o in OUTCOMES} for it in ITERATIONS},
    }
    record["pooled"] = pooled

    out_dir = resolve_project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    (out_dir / f"controls_{stamp}.json").write_text(json.dumps(record, indent=1) + "\n")

    fmt = lambda x: f"{x:g}" if isinstance(x, int) or float(x).is_integer() else f"{x:.1f}"
    P = pooled
    L = ["# Controls for the Min-Sum, agent, Min-Sum cascade", "",
         f"Generated by `scripts/cascade_controls.py` on {record['generated']}. Condition "
         f"`{CONDITION}`, seeds {', '.join(record['seeds'])}, each on its own held-out split. "
         f"Min-Sum fails on {f1} words, all of them by not converging; the agent recovers "
         f"{P['stage2']['rl']} of them and misses {f2}. Random flips are the mean over "
         f"{args.draws} draws per word.", "",
         f"## 1. Fallback decoders on Min-Sum's {f1} failures", "",
         "| Fallback | Recovered |", "|---|---|",
         f"| DQN agent | {P['stage2']['rl']} |",
         f"| Gallager-B | {P['stage2']['gallager_b']} |",
         f"| Greedy bit flipping | {P['stage2']['greedy']} |", ""]
    for set_name, title in [("min_sum_failures", f"Min-Sum's {f1} failures, no agent stage"),
                            ("agent_failures", f"the {f2} words the agent misses (the cascade's stage 3)")]:
        L += [f"## 2. First k flips, then Min-Sum again, on {title}", "",
              "Recovered (wrong codewords in parentheses where any):", "",
              "| k | Agent's flips | Greedy's flips | Random flips |", "|---|---|---|---|"]
        for k in map(str, K_SWEEP):
            cells = []
            for src_ in ("agent", "greedy", "random"):
                c = P["perturb_then_min_sum"][set_name][src_][k]
                cells.append(fmt(c["recovered"]) + (f" ({fmt(c['wrong_codeword'])})" if c["wrong_codeword"] else ""))
            L.append(f"| {k} | " + " | ".join(cells) + " |")
        L.append("")
    L += [f"## 3. Min-Sum alone with more iterations, on its {f1} failures at 20", "",
          "| Iterations | Recovered | Wrong codeword | Not converged |", "|---|---|---|---|"]
    for it in ITERATIONS:
        c = P["min_sum_iterations"][str(it)]
        L.append(f"| {it} | {c['recovered']} | {c['wrong_codeword']} | {c['not_converged']} |")
    L += ["", "Checks: greedy and Gallager-B outcomes on these words against their recorded "
          "evaluations, disagreements per seed: "
          + "; ".join(f"s{s}: {r['disagreements_with_record']}" for s, r in record["seeds"].items()) + "."]
    (out_dir / f"controls_{stamp}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
