#!/usr/bin/env python3
"""
Extract one held-out decoding episode of the final policy for the manuscript.

Writes docs/ablation/example_episode.json, which the manuscript's figure script
reads; like the other records under docs/ablation/ it is regenerated in full and
never edited by hand.

The episode is chosen by a fixed rule, not by inspection: in the evaluation of
the final policy (flat ordering, unseeded, epoch 3) on seed 42, the first
held-out word, in evaluation order, whose error pattern has weight 6 (the
correction radius), on which the agent ends in a task success and greedy bit
flipping, run on the same word, does not. The record carries both trajectories:
the bit flipped at each step (128 is STOP) and the syndrome weight before the
first step and after every step.

Usage:
    PYTHONPATH=. python scripts/extract_example_episode.py
"""

import gzip
import json

from scripts.analyze_ablation_details import eval_file
from src.utils.path_utils import resolve_project_path

CELL, SEED, WEIGHT = "NoC_NoE3ep", "42", 6
STOP = 128
OUT = "docs/ablation/example_episode.json"


def episodes(path):
    with gzip.open(path, "rt") as fh:
        return json.load(fh)["episodes"]


def trajectory(ep):
    steps = ep["steps"]
    return {
        "actions": [s["action"] for s in steps],
        "syndrome_weight": [steps[0]["state"]["syndrome_weight"]]
        + [s["next_state"]["syndrome_weight"] for s in steps],
        "flip_weight": [steps[0]["state"]["flip_weight"]]
        + [s["next_state"]["flip_weight"] for s in steps],
        "terminal_reason": ep["terminal_reason"],
    }


def main():
    root = resolve_project_path(".")
    cell_dir = root / "results" / "ablation" / f"{CELL}_s{SEED}"
    rl_path, greedy_path = eval_file(cell_dir, "rl"), eval_file(cell_dir, "greedy")
    rl, greedy = episodes(rl_path), episodes(greedy_path)
    if len(rl) != len(greedy):
        raise SystemExit("agent and greedy evaluations differ in length")

    for index, (a, g) in enumerate(zip(rl, greedy)):
        if a["noisy_codeword"] != g["noisy_codeword"]:
            raise SystemExit(f"episode {index}: agent and greedy decoded different words")
        meta = a["error_info"]["meta"]
        if (meta["num_errors"] == WEIGHT and a["terminal_reason"] == "SUCCESS_STOP"
                and g["terminal_reason"] != "SUCCESS_STOP"):
            break
    else:
        raise SystemExit("no episode matches the rule")

    record = {
        "generator": "scripts/extract_example_episode.py",
        "rule": ("first held-out word in evaluation order with an error pattern of weight "
                 f"{WEIGHT} on which the agent ends in a task success and greedy does not"),
        "cell": f"{CELL}_s{SEED}",
        "epoch": 3,
        "sources": [str(p.relative_to(root)) for p in (rl_path, greedy_path)],
        "evaluation_index": index,
        "error_model": meta["error_model"],
        "weight": meta["num_errors"],
        "error_positions": [i for i, b in enumerate(a["error_info"]["error_pattern"]) if b],
        "stop_action": STOP,
        "agent": trajectory(a),
        "greedy": trajectory(g),
    }
    (root / OUT).write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {OUT}: evaluation index {index}, {meta['error_model']}, "
          f"agent {record['agent']['terminal_reason']}, greedy {record['greedy']['terminal_reason']}")


if __name__ == "__main__":
    main()
