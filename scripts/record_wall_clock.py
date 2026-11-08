#!/usr/bin/env python3
"""
Record how long each ablation cell took, from start to its last evaluation file.

A cell's wall-clock time is the span from its run's start, the timestamp in its
result file names, to the modification time of its newest evaluation file, so it
covers dataset creation, training and evaluation. Modification times do not
survive a copy or an archive, so this script reads them once, where the runs
were made, and writes docs/ablation/WALL_CLOCK.json; the paper's generators read
that record. Like the other records, it is regenerated in full and never edited
by hand.

Cells whose span includes a system suspend are flagged and keep their time;
docs/ablation/PROTOCOL.md records the suspend.

Usage:
    PYTHONPATH=. python scripts/record_wall_clock.py
"""

import argparse
import datetime as dt
import json
import re
import sys

from src.utils.path_utils import resolve_project_path

# Cells whose wall-clock span includes a system suspend (PROTOCOL.md, sitting 10).
SUSPENDED = {"NoC_NoE_s46"}


def span(results_dir) -> dict:
    files = sorted(results_dir.glob("*_evaluation_agent_*.json.gz"))
    stamp = re.search(r"_(\d{8}_\d{6})_evaluation", files[0].name).group(1)
    start = dt.datetime.strptime(stamp, "%Y%m%d_%H%M%S")
    end = max(dt.datetime.fromtimestamp(f.stat().st_mtime) for f in files)
    return {"started": start.isoformat(), "finished": end.isoformat(timespec="seconds"),
            "minutes": round((end - start).total_seconds() / 60, 4)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--configs", default="config/ablation_config")
    parser.add_argument("--results", default="results/ablation")
    parser.add_argument("--output", default="docs/ablation/WALL_CLOCK.json")
    args = parser.parse_args()
    configs, results = resolve_project_path(args.configs), resolve_project_path(args.results)

    cells, pending = {}, []
    for cell in sorted(p.name for p in configs.iterdir() if p.is_dir()):
        # A cell still running has no RL evaluation file yet and is left out.
        if not any((results / cell).glob("*_evaluation_agent_rl.json.gz")):
            pending.append(cell)
            continue
        cells[cell] = {**span(results / cell), "suspended": cell in SUSPENDED}
    if not cells:
        sys.exit(f"no finished cells under {results}")

    record = {"generated": dt.date.today().isoformat(),
              "note": ("Local time. Minutes from the run's start stamp to its newest "
                       "evaluation file: dataset creation, training and evaluation."),
              "cells": cells}
    output = resolve_project_path(args.output)
    output.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {args.output}: {len(cells)} cells"
          + (f"; not finished: {', '.join(pending)}" if pending else ""))


if __name__ == "__main__":
    main()
