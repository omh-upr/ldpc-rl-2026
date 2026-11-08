#!/usr/bin/env python3
"""
Min-Sum, then the agent, then Min-Sum again: a cascade on the held-out words.

An exploratory experiment on the headline cells (`NoC_NoE3ep_s<seed>`, final
weights, epoch 3), each on its own held-out split:

    1. Hard-input Min-Sum on every received word, exactly as evaluated.
    2. On the words Min-Sum does not recover, the trained agent, run in the
       environment from the received word.
    3. On the words the agent does not recover either, the word after the
       agent's first k flips (the agent is deterministic, so re-running it gives
       the stage-2 trajectory) is handed to Min-Sum again. k = t_max / 2 = 3 is
       the case asked for; k = 1 to 6 is reported as a sweep.

A word counts as recovered only when the final word equals the transmitted
codeword. Stage 3 separates the two ways Min-Sum can fail: no convergence, which
a decoder detects, and convergence to a different codeword, an undetected error.

Stages 1 and 2 are checked against the recorded evaluations: Min-Sum's
per-frame outcome must equal the recorded one on every word, and the agent's
outcome on the stage-1 failures must equal its recorded outcome there.

Writes docs/ablation/cascade/cascade_<timestamp>.json and .md.

Usage:
    PYTHONPATH=. python scripts/cascade_minsum_agent.py [--seeds 42 43 44 45 46]
"""

import argparse
import copy
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import torch

from scripts.analyze_ablation_details import eval_file, load_many, success
from scripts.benchmark_decoders import decode_frame, find_model, load_cell_config, load_test_frames
from src.core.environment import LDPCEnvironment
from src.core.ldpc_code import LDPC_Code
from src.evaluation.agent_registry import AgentRegistry, AgentSpec
from src.utils.path_utils import resolve_project_path

CONDITION = "NoC_NoE3ep"
T_MAX = 6
K_PRIMARY = T_MAX // 2
K_SWEEP = list(range(1, T_MAX + 1))


def min_sum_outcome(decoder, H: np.ndarray, word: np.ndarray, original: np.ndarray) -> str:
    """'recovered', 'wrong_codeword' or 'not_converged' for one hard word."""
    syndrome = (H @ word) % 2
    _rel, decoded, converged = decoder.decode(word.astype(np.int8), syndrome.astype(np.int8))
    if not converged or np.any((H @ decoded) % 2):
        return "not_converged"
    return "recovered" if np.array_equal(decoded, original) else "wrong_codeword"


def partial_word(noisy: np.ndarray, actions: list, k: int, n: int) -> np.ndarray:
    """The word after the first k flip actions of a trajectory (fewer if it has fewer)."""
    word = noisy.copy()
    flips = [a for a in actions if a < n][:k]
    for a in flips:
        word[a] ^= 1
    return word


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--config-root", default="config/ablation_config")
    parser.add_argument("--cache-root", default="results/ablation_cache")
    parser.add_argument("--output-dir", default="docs/ablation/cascade")
    args = parser.parse_args()
    torch.set_num_threads(1)

    config_root = resolve_project_path(args.config_root)
    cache_root = resolve_project_path(args.cache_root)
    record = {"generated": dt.datetime.now().isoformat(timespec="seconds"),
              "condition": CONDITION, "k_primary": K_PRIMARY, "seeds": {}}

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
        model_path = find_model(models_dir)
        originals, noisies = load_test_frames(resolve_project_path(config["dataset_creation"]["output_path"]))
        print(f"{cell}: {len(noisies):,} held-out words, model {model_path.name}", flush=True)

        registry = AgentRegistry(copy.deepcopy(config), env=env, models_dir=str(models_dir))
        ms_spec = next(b for b in config["evaluation"]["baseline_agents"] if b["name"] == "min_sum_expert")
        ms_agent = registry.create_agent(AgentSpec(
            name="min_sum_expert", display_name="Min-sum", enabled=True,
            agent_class=registry.agent_classes["min_sum_expert"], config=ms_spec))
        decoder = ms_agent.decoder
        rl_agent = registry.create_agent(AgentSpec(
            name="rl", display_name="RL Agent", enabled=True, agent_class=None,
            requires_model=True, model_path=str(model_path), config=config["training"]))

        # Recorded outcomes, for the checks.
        src_ms, src_rl = eval_file(results_dir, "min_sum_expert"), eval_file(results_dir, "rl")
        rec = load_many([("eval", src_ms), ("eval", src_rl)], cache_root, workers=2)
        rec_ms, rec_rl = success(rec[str(src_ms)]), success(rec[str(src_rl)])

        # Stage 1: Min-Sum on every received word.
        stage1 = np.array([min_sum_outcome(decoder, H, noisies[i], originals[i])
                           for i in range(len(noisies))])
        ok1 = stage1 == "recovered"
        mismatch1 = int((ok1 != rec_ms).sum())
        fail1 = np.flatnonzero(~ok1)
        print(f"  stage 1: Min-Sum fails on {len(fail1)} "
              f"({dict(zip(*np.unique(stage1[fail1], return_counts=True)))}); "
              f"{mismatch1} disagreements with the recorded evaluation", flush=True)

        # Stage 2: the agent on Min-Sum's failures.
        stage2 = {}
        for i in fail1.tolist():
            reason, actions, _ = decode_frame(rl_agent, env, originals[i], noisies[i])
            stage2[i] = (reason, actions)
        ok2 = {i: stage2[i][0] == "SUCCESS_STOP" for i in fail1.tolist()}
        mismatch2 = sum(ok2[i] != bool(rec_rl[i]) for i in fail1.tolist())
        fail2 = [i for i in fail1.tolist() if not ok2[i]]
        reasons2 = {}
        for i in fail2:
            reasons2[stage2[i][0]] = reasons2.get(stage2[i][0], 0) + 1
        print(f"  stage 2: agent recovers {sum(ok2.values())} of {len(fail1)}; "
              f"{mismatch2} disagreements with the recorded evaluation; "
              f"endings of the rest {reasons2}", flush=True)

        # Stage 3: first k flips of the agent's trajectory, then Min-Sum again.
        stage3 = {}
        for k in K_SWEEP:
            counts = {"recovered": 0, "wrong_codeword": 0, "not_converged": 0}
            closer = 0
            for i in fail2:
                word = partial_word(noisies[i], stage2[i][1], k, n)
                counts[min_sum_outcome(decoder, H, word, originals[i])] += 1
                closer += int((word != originals[i]).sum() < (noisies[i] != originals[i]).sum())
            counts["closer_than_received"] = closer
            stage3[str(k)] = counts
        print(f"  stage 3, k={K_PRIMARY}: {stage3[str(K_PRIMARY)]}", flush=True)

        record["seeds"][str(seed)] = {
            "model": str(model_path.relative_to(resolve_project_path("."))),
            "words": len(noisies),
            "stage1": {"failures": len(fail1),
                       "endings": {k: int(v) for k, v in zip(*np.unique(stage1[fail1], return_counts=True))},
                       "disagreements_with_record": mismatch1},
            "stage2": {"recovered": int(sum(ok2.values())), "failures": len(fail2),
                       "endings_of_failures": reasons2,
                       "disagreements_with_record": int(mismatch2),
                       "weights_of_failures": {str(w): int(c) for w, c in zip(*np.unique(
                           [(noisies[i] != originals[i]).sum() for i in fail2] or [0],
                           return_counts=True))} if fail2 else {}},
            "stage3": stage3,
        }

    # Pooled summary.
    S = record["seeds"].values()
    words = sum(r["words"] for r in S)
    f1 = sum(r["stage1"]["failures"] for r in S)
    r2 = sum(r["stage2"]["recovered"] for r in S)
    f2 = sum(r["stage2"]["failures"] for r in S)
    pooled3 = {k: {key: sum(r["stage3"][k][key] for r in S) for key in
                   ("recovered", "wrong_codeword", "not_converged", "closer_than_received")}
               for k in map(str, K_SWEEP)}
    record["pooled"] = {"words": words, "stage1_failures": f1, "stage2_recovered": r2,
                        "stage2_failures": f2, "stage3": pooled3}

    out_dir = resolve_project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    (out_dir / f"cascade_{stamp}.json").write_text(json.dumps(record, indent=1) + "\n")

    fer = lambda failures: f"{100 * failures / words:.3f}"
    L = ["# Min-Sum, agent, Min-Sum cascade", "",
         f"Generated by `scripts/cascade_minsum_agent.py` on {record['generated']}. "
         f"Condition `{CONDITION}` (final weights), seeds {', '.join(record['seeds'])}, "
         f"each on its own held-out split; {words:,} words pooled.", "",
         "| Stage | Recovered | Still failing | Frame error rate (%) |",
         "|---|---|---|---|",
         f"| 1. Min-Sum | {words - f1:,} | {f1:,} | {fer(f1)} |",
         f"| 2. Agent on Min-Sum's failures | {r2:,} | {f2:,} | {fer(f2)} |"]
    p = pooled3[str(K_PRIMARY)]
    L.append(f"| 3. Agent's first {K_PRIMARY} flips, then Min-Sum | {p['recovered']:,} | "
             f"{f2 - p['recovered']:,} (of which {p['wrong_codeword']} wrong codewords) | "
             f"{fer(f2 - p['recovered'])} |")
    L += ["", "Stage 3 by the number of the agent's flips kept before Min-Sum runs again:", "",
          "| k | Recovered | Wrong codeword (undetected) | Not converged | Partial word closer than received |",
          "|---|---|---|---|---|"]
    for k in map(str, K_SWEEP):
        q = pooled3[k]
        L.append(f"| {k} | {q['recovered']} | {q['wrong_codeword']} | {q['not_converged']} | "
                 f"{q['closer_than_received']} of {f2} |")
    L += ["", "Checks against the recorded evaluations (disagreements, per seed): "
          + "; ".join(f"s{s}: stage 1 {r['stage1']['disagreements_with_record']}, "
                      f"stage 2 {r['stage2']['disagreements_with_record']}"
                      for s, r in record["seeds"].items()) + "."]
    (out_dir / f"cascade_{stamp}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
