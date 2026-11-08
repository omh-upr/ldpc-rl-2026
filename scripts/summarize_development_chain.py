#!/usr/bin/env python3
"""
Summarize the tuning chain that led to the ablation's training configuration.

Each step of the chain is one run on the seed-42 dataset with the same expert
replay, differing from the previous step by one setting; its configuration is
the folder of that name under config/experiments_config/. This script reads the
held-out evaluation files of those runs, which are kept under archive/reference/
(ignored by git), and writes docs/ablation/DEVELOPMENT_CHAIN.json, the record the
paper's development-path table is generated from. Like the other records,
it is regenerated in full and never edited by hand.

Usage:
    PYTHONPATH=. python scripts/summarize_development_chain.py
"""

import argparse
import datetime as dt
import gzip
import json
import sys

import numpy as np

from src.utils.path_utils import resolve_project_path

# (run timestamp, experiment config folder, the change the step makes)
CHAIN = [
    ("20260905_095822", "64x128_Full_AutoEp", "Tapered epsilon, one epoch (starting point)"),
    ("20260906_082214", "64x128_Full_AutoEp_Uniform", "Uniform epsilon across phases"),
    ("20260906_095320", "64x128_Full_Uniform_NoReflip", "Refuse an immediate re-flip"),
    ("20260906_111741", "64x128_Full_Uniform_3ep", "Three epochs, halving epsilon ladder"),
]


def summarize(path) -> dict:
    with gzip.open(path) as f:
        episodes = json.load(f)["episodes"]
    ok = np.array([e["terminal_reason"] == "SUCCESS_STOP" for e in episodes])
    steps = np.array([len(e["steps"]) for e in episodes])
    return {"episodes": len(episodes), "success": float(100.0 * ok.mean()),
            "mean_steps": float(steps.mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--archive", default="archive/reference")
    parser.add_argument("--output", default="docs/ablation/DEVELOPMENT_CHAIN.json")
    args = parser.parse_args()
    archive = resolve_project_path(args.archive)

    steps, previous = [], None
    for stamp, folder, change in CHAIN:
        files = sorted((archive / stamp).glob(f"*_{stamp}_evaluation_agent_rl.json.gz"))
        if len(files) != 1:
            sys.exit(f"{archive / stamp}: expected one RL evaluation file, found {len(files)}")
        row = {"run": stamp, "config_folder": folder, "change": change, "seed": 42,
               **summarize(files[0])}
        row["gain"] = None if previous is None else row["success"] - previous
        previous = row["success"]
        steps.append(row)
        print(f"{stamp}  {row['success']:.2f}%  {change}")

    record = {"generated": dt.date.today().isoformat(),
              "source": "config/experiments_config/<config_folder>",
              "steps": steps}
    output = resolve_project_path(args.output)
    output.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
