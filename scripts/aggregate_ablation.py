#!/usr/bin/env python3
"""
Aggregate the five-seed ablation and write docs/ablation/RESULTS.md.

Reads the evaluation files of the twenty cells under results/ablation/ and the
per-seed datasets under data/ablation/, computes the tables specified in
section 7 of docs/ablation/PROTOCOL.md, and writes them as one Markdown
document. The document is generated in full on every run and is not edited by
hand. No database is involved: every number comes from the gzipped evaluation
JSON and the HDF5 dataset files.

The multi-epoch conditions of the training-budget study (Full3ep, OnlyC3ep,
NoC_NoE3ep, and OnlyE3ep once its cells exist) get their own section, one row per epoch: epoch 1 is read
from the one-epoch cell whose network it equals, intermediate epochs from the
evaluation-only cells `<cell>_ep<E>`, and the last epoch from the cell itself.
A condition is reported only when every one of its cells is present, so the
document can be regenerated while a campaign is under way. Full3epFloor is a
control on whichever seeds it was run.

Next to the document the script writes a JSON sidecar (same stem, `.json`)
holding every per-cell summary the tables are computed from, so downstream
consumers, the manuscript's table and figure scripts among them, never parse
Markdown.

Usage:
    PYTHONPATH=. python scripts/aggregate_ablation.py [--output PATH]
                                                     [--seeds 42 43 ...]

Definitions used throughout (see the "Definitions" section of the output):
    success rate      episodes whose terminal reason starts with SUCCESS, i.e.
                      the agent stopped on the transmitted codeword
    frame error rate  episodes whose final word differs from the transmitted
                      codeword; lower than 1 - success because a run can end on
                      the correct word at the step limit without stopping
    final BER         bits still wrong at the end, over all bits of all frames
    mean steps        environment steps per episode, the stop action included
"""

import argparse
import datetime as dt
import glob
import gzip
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

from src.utils.path_utils import resolve_project_path

# Cell name -> (dataset file stem, curriculum flag, expert seeding flag).
CONDITIONS = {
    "NoC_NoE": ("flat", False, False),
    "OnlyC": ("curriculum", True, False),
    "OnlyE": ("flat", False, True),
    "Full": ("curriculum", True, True),
}
CONDITION_ORDER = ["NoC_NoE", "OnlyC", "OnlyE", "Full"]
AGENTS = ["rl", "greedy", "min_sum_expert"]
AGENT_LABEL = {"rl": "RL", "greedy": "Greedy", "min_sum_expert": "Min-sum",
               "gallager_b": "Gallager-B"}
# Baselines scored after the campaign by evaluation-only configs
# (scripts/make_baseline_eval_configs.py), one folder per seed and dataset under
# results/ablation/baseline_eval/. Reported once all ten folders exist.
ADDED_BASELINES = ["gallager_b"]
# Filled in by main(): the baselines every table reports, in column order.
BASELINES = ["greedy", "min_sum_expert"]
WEIGHTS = list(range(7))
# Within-seed contrasts, (name, minuend, subtrahend), from the protocol.
CONTRASTS = [
    ("Full - OnlyC", "Full", "OnlyC"),
    ("OnlyE - NoC_NoE", "OnlyE", "NoC_NoE"),
    ("Full - OnlyE", "Full", "OnlyE"),
    ("OnlyC - NoC_NoE", "OnlyC", "NoC_NoE"),
    ("Full - NoC_NoE", "Full", "NoC_NoE"),
]
CONTRAST_MECHANISM = {
    "Full - OnlyC": "expert seeding, with curriculum",
    "OnlyE - NoC_NoE": "expert seeding, without curriculum",
    "Full - OnlyE": "curriculum, with expert seeding",
    "OnlyC - NoC_NoE": "curriculum, without expert seeding",
    "Full - NoC_NoE": "both mechanisms together",
}

# Multi-epoch conditions: name -> (one-epoch condition whose network equals
# epoch 1, number of epochs). Reported when all of their cells exist.
EPOCH_CONDITIONS = {"Full3ep": ("Full", 3), "OnlyC3ep": ("OnlyC", 3),
                    "NoC_NoE3ep": ("NoC_NoE", 3), "OnlyE3ep": ("OnlyE", 3)}
# Controls reported on whatever seeds they were run.
EPOCH_CONTROLS = {"Full3epFloor": ("Full", 3, "Full3ep")}
# Within-seed contrasts between multi-epoch conditions, one per epoch.
EPOCH_CONTRASTS = [("Full3ep - OnlyC3ep", "Full3ep", "OnlyC3ep",
                    "expert seeding, with curriculum, three epochs"),
                   ("OnlyC3ep - NoC_NoE3ep", "OnlyC3ep", "NoC_NoE3ep",
                    "curriculum, without expert seeding, three epochs"),
                   ("OnlyE3ep - NoC_NoE3ep", "OnlyE3ep", "NoC_NoE3ep",
                    "expert seeding, without curriculum, three epochs"),
                   ("Full3ep - OnlyE3ep", "Full3ep", "OnlyE3ep",
                    "curriculum, with expert seeding, three epochs")]

# Two-sided 97.5% Student t quantiles by degrees of freedom.
T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
         6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


# --------------------------------------------------------------------------
# Per-cell summaries from the evaluation files
# --------------------------------------------------------------------------

def load_episodes(cell_dir: Path, agent: str) -> list:
    files = sorted(cell_dir.glob(f"*_evaluation_agent_{agent}.json.gz"))
    if len(files) != 1:
        sys.exit(f"{cell_dir.name}: expected one evaluation file for {agent}, "
                 f"found {len(files)}")
    with gzip.open(files[0]) as f:
        return json.load(f)["episodes"]


def summarize_agent(episodes: list) -> dict:
    """Headline metrics and breakdowns for one agent on one cell."""
    n = len(episodes)
    n_bits = len(episodes[0]["original_codeword"])
    successes = 0
    frame_errors = 0
    bit_errors = 0
    steps_total = 0
    by_weight = defaultdict(lambda: [0, 0])   # weight -> [successes, count]
    by_model = defaultdict(lambda: [0, 0])    # error model -> [successes, count]
    reasons = Counter()
    patterns = []
    for e in episodes:
        success = e["terminal_reason"].startswith("SUCCESS")
        word = np.array(e["noisy_codeword"], dtype=np.int8)
        for step in e["steps"]:
            action = step["action"]
            if action < n_bits:
                word[action] ^= 1
        wrong = int((word != np.array(e["original_codeword"], dtype=np.int8)).sum())
        successes += success
        frame_errors += wrong > 0
        bit_errors += wrong
        steps_total += len(e["steps"])
        meta = e["error_info"]["meta"]
        by_weight[int(meta["num_errors"])][0] += success
        by_weight[int(meta["num_errors"])][1] += 1
        by_model[meta["error_model"]][0] += success
        by_model[meta["error_model"]][1] += 1
        reasons[e["terminal_reason"]] += 1
        patterns.append(np.packbits(np.array(e["error_info"]["error_pattern"],
                                             dtype=np.uint8)).tobytes())
    return {
        "n": n,
        "success": 100.0 * successes / n,
        "fer": 100.0 * frame_errors / n,
        "ber": 100.0 * bit_errors / (n * n_bits),
        "steps": steps_total / n,
        "by_weight": {w: (100.0 * s / c if c else float("nan"), c)
                      for w, (s, c) in sorted(by_weight.items())},
        "by_model": {m: (100.0 * s / c, c) for m, (s, c) in sorted(by_model.items())},
        "reasons": dict(reasons),
        "patterns": patterns,
    }


def load_cell(results_root: Path, cell: str) -> dict:
    cell_dir = results_root / cell
    if not cell_dir.is_dir():
        sys.exit(f"missing results folder {cell_dir}")
    summary = {}
    for agent in AGENTS:
        summary[agent] = summarize_agent(load_episodes(cell_dir, agent))
    ref = summary["rl"]["patterns"]
    for agent in AGENTS[1:]:
        if summary[agent]["patterns"] != ref:
            sys.exit(f"{cell}: {agent} was evaluated on different episodes than rl")
    return summary


def epoch_cell_names(name: str, seed: int, base: str, epochs: int) -> dict:
    """Epoch -> results folder holding that epoch's evaluation."""
    names = {1: f"{base}_s{seed}", epochs: f"{name}_s{seed}"}
    for e in range(2, epochs):
        names[e] = f"{name}_s{seed}_ep{e}"
    return names


def load_epoch_cell(results_root: Path, name: str, seed: int, base: str, epochs: int,
                    base_summary: dict) -> dict:
    """RL summary per epoch for one multi-epoch cell. Every epoch must have been
    evaluated on the base cell's held-out episodes."""
    out = {}
    for e, folder in sorted(epoch_cell_names(name, seed, base, epochs).items()):
        if e == 1:
            out[e] = base_summary["rl"]
            continue
        out[e] = summarize_agent(load_episodes(results_root / folder, "rl"))
        if out[e]["patterns"] != base_summary["rl"]["patterns"]:
            sys.exit(f"{folder}: evaluated on different episodes than {base}_s{seed}")
    return out


def epoch_cells_present(results_root: Path, name: str, seed: int, base: str,
                        epochs: int) -> bool:
    return all(any((results_root / folder).glob("*_evaluation_agent_rl.json.gz"))
               for folder in epoch_cell_names(name, seed, base, epochs).values())


# --------------------------------------------------------------------------
# Train/test disjointness from the dataset files
# --------------------------------------------------------------------------

def disjointness(dataset_path: Path) -> dict:
    """Fraction of held-out error patterns also present in the training split,
    by error weight. Returns {weight: (percent, count)} plus 'all'."""
    with h5py.File(dataset_path, "r") as h:
        phases = sorted(k for k in h if k.startswith("phase_"))
        patterns = np.concatenate([h[f"{p}/error_patterns"][:] for p in phases])
        models = np.concatenate([h[f"{p}/metadata"]["error_model"] for p in phases])
        train = h["splits/train_indices"][:]
        test = h["splits/test_indices"][:]
    packed = np.packbits(patterns.astype(np.uint8), axis=1)
    keys = packed.view(np.dtype((np.void, packed.shape[1]))).ravel()
    weights = patterns.sum(axis=1)
    train_keys = set(keys[train].tolist())
    seen = np.array([k in train_keys for k in keys[test].tolist()])
    out = {}
    for w in WEIGHTS:
        mask = weights[test] == w
        out[w] = (100.0 * seen[mask].mean() if mask.any() else float("nan"), int(mask.sum()))
    out["all"] = (100.0 * seen.mean(), int(len(test)))
    test_models = np.array([m.decode() for m in models[test]])
    out["by_model"] = {}
    for m in sorted(set(test_models.tolist())):
        mask = test_models == m
        out["by_model"][m] = (100.0 * seen[mask].mean(), int(mask.sum()))
    # Unstructured draws only: the figure the review audit asks for.
    mask = (test_models == "uniform_count") | (test_models == "uniform_ber")
    out["uniform"] = {}
    for w in WEIGHTS:
        wm = mask & (weights[test] == w)
        out["uniform"][w] = (100.0 * seen[wm].mean() if wm.any() else float("nan"), int(wm.sum()))
    return out


# --------------------------------------------------------------------------
# Statistics and formatting
# --------------------------------------------------------------------------

def stats(values: list) -> dict:
    x = np.array(values, dtype=float)
    k = len(x)
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if k > 1 else float("nan")
    if k - 1 in T_975 and k > 1:
        half = T_975[k - 1] * sd / math.sqrt(k)
    else:
        half = float("nan")
    return {"mean": mean, "sd": sd, "half": half, "min": float(x.min()),
            "max": float(x.max()), "n": k}


def fmt(x: float, d: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def fmt_ci(s: dict, d: int = 1) -> str:
    return f"{fmt(s['mean'], d)} ± {fmt(s['half'], d)}"


def table(header: list, rows: list) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Document
# --------------------------------------------------------------------------

def build_document(cells: dict, seeds: list, disjoint: dict, results_root: Path,
                   data_root: Path, epoch_cells: dict, controls: dict) -> str:
    k = len(seeds)
    t = T_975.get(k - 1, float("nan"))
    L = []
    L.append("# Ablation Results: Curriculum and Expert Seeding on the (128,64) Code\n")
    extra = sum(len(d) for d in epoch_cells.values()) + sum(len(d) for d in controls.values())
    extra_text = (f", the {extra} multi-epoch cells of section 9 with their "
                  f"per-epoch evaluations," if extra else "")
    L.append(f"Generated by `scripts/aggregate_ablation.py` on "
             f"{dt.date.today().isoformat()} from the evaluation files of the "
             f"{len(cells)} cells under `{results_root}/`{extra_text} and the datasets under "
             f"`{data_root}/`. Do not edit; rerun the script. The design, the "
             f"checks each cell passed, and the run log are in `PROTOCOL.md`. "
             f"Every per-cell summary behind these tables is also written to the "
             f"JSON file of the same name.\n")

    # -- Definitions
    L.append("## Definitions\n")
    L.append(
        "Each cell is one condition trained on one seed's dataset and evaluated on "
        "that dataset's held-out split of 105,000 error patterns. Percentages are "
        "rates over those episodes, in points. Conditions are aggregated over the "
        f"{k} seeds with the mean, the sample standard deviation $s$ and a 95% "
        f"confidence interval from the $t$ distribution with {k - 1} degrees of "
        f"freedom, $t_{{0.975,{k - 1}}} = {t}$, so the half-width written after "
        f"± is ${t / math.sqrt(k):.2f}\\,s$. Contrasts are differences within a "
        "seed, aggregated the same way, so seed-to-seed offsets cancel. No cell "
        "is bolded.\n")
    L.append(
        "- **Success rate**: episodes whose terminal reason starts with `SUCCESS`, "
        "that is the agent chose STOP on the transmitted codeword.\n"
        "- **Frame error rate (FER)**: episodes whose final word, after every "
        "flip the agent made, differs from the transmitted codeword. It is "
        "slightly below 100 minus the success rate, because a run can reach the "
        "correct word on its last permitted step without stopping.\n"
        "- **Final BER**: bits still wrong at the end of the episode, over all "
        "bits of all held-out frames, in percent to three decimals.\n"
        "- **Mean steps**: environment steps per episode, the STOP action "
        "included, with a limit of 12.\n"
        "- **Weight**: number of flipped bits in the error pattern. Held-out "
        "patterns run from weight 0 to 6, the correction radius of the code.\n"
        "- **Conditions**: `NoC_NoE` neither mechanism, `OnlyC` curriculum only, "
        "`OnlyE` expert seeding only, `Full` both. `OnlyC` and `Full` train and "
        "evaluate on the seed's curriculum dataset, `NoC_NoE` and `OnlyE` on the "
        "seed's flat dataset with the same marginal mixture, so baselines are "
        "identical within each pair and differ slightly between the pairs.\n")

    # -- Headline
    L.append("## 1. Held-out success rate by condition\n")
    rows = []
    for c in CONDITION_ORDER:
        vals = [cells[(c, s)]["rl"]["success"] for s in seeds]
        st = stats(vals)
        rows.append([f"`{c}`", fmt_ci(st), fmt(st["sd"]), fmt(st["min"]), fmt(st["max"])])
    L.append(table(["Condition", "RL success (%)", "s", "min", "max"], rows) + "\n")

    L.append("Per seed:\n")
    rows = []
    for s in seeds:
        rows.append([s] + [fmt(cells[(c, s)]["rl"]["success"], 2) for c in CONDITION_ORDER])
    L.append(table(["Seed"] + [f"`{c}`" for c in CONDITION_ORDER], rows) + "\n")

    # -- Other headline metrics
    L.append("## 2. Frame error rate, final bit error rate and mean steps\n")
    L.append("RL agent, mean over seeds with the 95% interval:\n")
    rows = []
    for c in CONDITION_ORDER:
        fer = stats([cells[(c, s)]["rl"]["fer"] for s in seeds])
        ber = stats([cells[(c, s)]["rl"]["ber"] for s in seeds])
        stp = stats([cells[(c, s)]["rl"]["steps"] for s in seeds])
        rows.append([f"`{c}`", fmt_ci(fer), fmt_ci(ber, 3), fmt_ci(stp)])
    L.append(table(["Condition", "FER (%)", "Final BER (%)", "Mean steps"], rows) + "\n")

    L.append("Baselines on the same held-out splits (identical for the two "
             "conditions sharing a dataset; see section 6 for their spread):\n")
    rows = []
    for ds, cond in [("curriculum", "Full"), ("flat", "OnlyE")]:
        for agent in BASELINES:
            suc = stats([cells[(cond, s)][agent]["success"] for s in seeds])
            fer = stats([cells[(cond, s)][agent]["fer"] for s in seeds])
            ber = stats([cells[(cond, s)][agent]["ber"] for s in seeds])
            stp = stats([cells[(cond, s)][agent]["steps"] for s in seeds])
            rows.append([AGENT_LABEL[agent], ds, fmt_ci(suc), fmt_ci(fer),
                         fmt_ci(ber, 3), fmt_ci(stp)])
    L.append(table(["Agent", "Dataset", "Success (%)", "FER (%)", "Final BER (%)",
                    "Mean steps"], rows) + "\n")
    if "gallager_b" in BASELINES:
        L.append("Gallager-B was scored after the campaign by evaluation-only runs on "
                 "the same held-out splits (`results/ablation/baseline_eval/`), with "
                 "thresholds selected on training words "
                 "(`GALLAGER_B_SELECTION.md`).\n")

    # -- Contrasts
    L.append("## 3. Contrasts\n")
    L.append("Within-seed differences in RL success rate, in points, with the "
             "95% interval over seeds. An interval that excludes zero is a "
             "difference the five seeds resolve.\n")
    rows = []
    for name, a, b in CONTRASTS:
        diffs = [cells[(a, s)]["rl"]["success"] - cells[(b, s)]["rl"]["success"] for s in seeds]
        st = stats(diffs)
        per_seed = ", ".join(f"{d:+.1f}" for d in diffs)
        rows.append([f"`{name}`", CONTRAST_MECHANISM[name], fmt_ci(st), fmt(st["sd"]), per_seed])
    L.append(table(["Contrast", "Isolates", "Difference (points)", "s",
                    "Per seed " + ", ".join(str(s) for s in seeds)], rows) + "\n")

    # -- By weight
    L.append("## 4. Success rate by error weight\n")
    L.append("RL agent, mean over seeds with the 95% interval, by weight of the "
             "held-out pattern. The count is the number of held-out patterns of "
             "that weight in one seed's split (it varies slightly between seeds "
             "and between the curriculum and flat datasets; the value shown is "
             "for seed {} on the curriculum dataset).\n".format(seeds[0]))
    rows = []
    for w in WEIGHTS:
        row = [w, cells[("Full", seeds[0])]["rl"]["by_weight"].get(w, (float("nan"), 0))[1]]
        for c in CONDITION_ORDER:
            st = stats([cells[(c, s)]["rl"]["by_weight"].get(w, (float("nan"),))[0] for s in seeds])
            row.append(fmt_ci(st))
        for agent in BASELINES:
            st = stats([cells[("Full", s)][agent]["by_weight"].get(w, (float("nan"),))[0] for s in seeds])
            row.append(fmt(st["mean"]))
        rows.append(row)
    L.append(table(["Weight", "Count"] + [f"`{c}`" for c in CONDITION_ORDER]
                   + [f"{AGENT_LABEL[a]} (curr.)" for a in BASELINES], rows) + "\n")

    L.append("Contrasts by weight, within-seed differences in points with the "
             "95% interval:\n")
    rows = []
    for w in WEIGHTS:
        row = [w]
        for name, a, b in CONTRASTS:
            diffs = [cells[(a, s)]["rl"]["by_weight"][w][0] - cells[(b, s)]["rl"]["by_weight"][w][0]
                     for s in seeds]
            row.append(fmt_ci(stats(diffs)))
        rows.append(row)
    L.append(table(["Weight"] + [f"`{n}`" for n, _, _ in CONTRASTS], rows) + "\n")

    # -- By error model
    L.append("## 5. Success rate by error model\n")
    L.append("RL agent, mean over seeds with the 95% interval, by the model that "
             "drew the held-out pattern. Counts are for seed {} on the "
             "curriculum dataset.\n".format(seeds[0]))
    models = sorted({m for c in CONDITION_ORDER for s in seeds
                     for m in cells[(c, s)]["rl"]["by_model"]})
    rows = []
    for m in models:
        row = [f"`{m}`", cells[("Full", seeds[0])]["rl"]["by_model"].get(m, (0, 0))[1]]
        for c in CONDITION_ORDER:
            st = stats([cells[(c, s)]["rl"]["by_model"][m][0] for s in seeds])
            row.append(fmt_ci(st))
        rows.append(row)
    L.append(table(["Error model", "Count"] + [f"`{c}`" for c in CONDITION_ORDER], rows) + "\n")

    # -- Baseline spread
    L.append("## 6. Baseline spread across seeds\n")
    L.append("Success rate of the two baselines on each seed's held-out split, "
             "per dataset. The baselines are deterministic, so this spread is "
             "entirely due to the drawn patterns.\n")
    rows = []
    for ds, cond in [("curriculum", "Full"), ("flat", "OnlyE")]:
        for agent in BASELINES:
            vals = [cells[(cond, s)][agent]["success"] for s in seeds]
            st = stats(vals)
            rows.append([AGENT_LABEL[agent], ds] + [fmt(v, 2) for v in vals]
                        + [fmt_ci(st, 2), fmt(st["sd"], 2)])
    L.append(table(["Agent", "Dataset"] + [str(s) for s in seeds] + ["Mean ± 95%", "s"], rows) + "\n")

    # -- Disjointness
    L.append("## 7. Overlap between held-out and training error patterns\n")
    L.append("For each seed and dataset, the percentage of held-out error "
             "patterns that also occur, as the same set of flipped positions, "
             "among that dataset's training patterns, by weight. Low weights "
             "cannot be disjoint: there is one pattern of weight 0 and 128 of "
             "weight 1. The figure that matters is at the weights where the "
             "number of possible patterns exceeds the training set.\n")
    header = ["Seed", "Dataset"] + [f"w={w}" for w in WEIGHTS] + ["All"]
    rows = []
    for s in seeds:
        for ds in ["curriculum", "flat"]:
            d = disjoint[(s, ds)]
            rows.append([s, ds] + [fmt(d[w][0]) for w in WEIGHTS] + [fmt(d["all"][0])])
    L.append(table(header, rows) + "\n")
    counts = disjoint[(seeds[0], "curriculum")]
    L.append("Held-out pattern counts by weight for seed {} on the curriculum "
             "dataset: ".format(seeds[0])
             + ", ".join(f"w={w}: {counts[w][1]:,}" for w in WEIGHTS) + ".\n")

    L.append("The rise at weights 5 and 6 comes from the structured models. A "
             "burst of a fixed length has at most 128 placements and a "
             "systematic pattern with a fixed step and offset set has fewer, so "
             "those draws repeat between the splits by construction. Overlap by "
             "the model that drew the held-out pattern, mean over seeds:\n")
    models = sorted({m for key in disjoint for m in disjoint[key]["by_model"]})
    rows = []
    for m in models:
        row = [f"`{m}`", disjoint[(seeds[0], "curriculum")]["by_model"].get(m, (0, 0))[1]]
        for ds in ["curriculum", "flat"]:
            vals = [disjoint[(s, ds)]["by_model"][m][0] for s in seeds
                    if m in disjoint[(s, ds)]["by_model"]]
            row.append(fmt(float(np.mean(vals))) if vals else "n/a")
        rows.append(row)
    L.append(table(["Error model", "Count", "curriculum", "flat"], rows) + "\n")

    L.append("Overlap restricted to the unstructured draws (`uniform_count` and "
             "`uniform_ber`), by weight, mean over seeds:\n")
    rows = []
    for ds in ["curriculum", "flat"]:
        row = [ds]
        for w in WEIGHTS:
            vals = [disjoint[(s, ds)]["uniform"][w][0] for s in seeds]
            row.append(fmt(float(np.nanmean(vals))))
        rows.append(row)
    L.append(table(["Dataset"] + [f"w={w}" for w in WEIGHTS], rows) + "\n")

    # -- Terminal reasons
    L.append("## 8. Terminal reasons\n")
    L.append("How RL episodes ended, summed over the five seeds of each "
             "condition. `SUCCESS_STOP` is the success rate; `FAILED_STOP` is "
             "a stop on a wrong word; `MAX_STEPS` and `MAX_FLIPS` are the step "
             "and flip limits.\n")
    reasons = sorted({r for c in CONDITION_ORDER for s in seeds for r in cells[(c, s)]["rl"]["reasons"]})
    rows = []
    for c in CONDITION_ORDER:
        total = sum(cells[(c, s)]["rl"]["n"] for s in seeds)
        row = [f"`{c}`"]
        for r in reasons:
            cnt = sum(cells[(c, s)]["rl"]["reasons"].get(r, 0) for s in seeds)
            row.append(f"{100.0 * cnt / total:.2f}")
        rows.append(row)
    L.append(table(["Condition"] + [f"`{r}` (%)" for r in reasons], rows) + "\n")

    if epoch_cells:
        L.append(epoch_section(epoch_cells, controls, cells, seeds))

    return "\n".join(L)


def epoch_section(epoch_cells: dict, controls: dict, cells: dict, seeds: list) -> str:
    """Section 9: the multi-epoch conditions, one row per epoch."""
    L = []
    L.append("## 9. Training budget: multi-epoch conditions\n")
    L.append(
        "A multi-epoch condition is a one-epoch condition trained for more passes "
        "over the same training words, each epoch restarting the phase schedule "
        "and reopening epsilon on a halving ladder (0.5, 0.25, 0.125, each "
        "annealing to 0.01 at the phase midpoint). Its weights are saved at every "
        "epoch boundary and every checkpoint is scored on the same held-out split "
        "as the one-epoch cell. Epoch 1 is read from the one-epoch cell, whose "
        "network it equals tensor for tensor. Increments are within-seed "
        "differences in points, aggregated like the contrasts in section 3. "
        "Success rates here carry two decimals because the increments are of the "
        "order of one point.\n")
    for name, per_seed in epoch_cells.items():
        base, epochs = EPOCH_CONDITIONS[name]
        E = list(range(1, epochs + 1))
        L.append(f"### `{name}`: `{base}` for {epochs} epochs\n")

        rows = []
        for s in seeds:
            v = [per_seed[s][e]["success"] for e in E]
            rows.append([s] + [fmt(x, 2) for x in v]
                        + [f"{v[i + 1] - v[i]:+.2f}" for i in range(epochs - 1)]
                        + [f"{v[-1] - v[0]:+.2f}"])
        head = (["Seed"] + [f"Epoch {e}" for e in E]
                + [f"{e} to {e + 1}" for e in E[:-1]] + ["Total"])
        L.append("RL success rate (%) per seed and epoch:\n")
        L.append(table(head, rows) + "\n")

        rows = []
        for e in E:
            st = {m: stats([per_seed[s][e][m] for s in seeds])
                  for m in ["success", "fer", "ber", "steps"]}
            rows.append([e, fmt_ci(st["success"], 2), fmt(st["success"]["sd"], 2),
                         fmt_ci(st["fer"], 2), fmt_ci(st["ber"], 3), fmt_ci(st["steps"], 2)])
        L.append("Mean over seeds with the 95% interval:\n")
        L.append(table(["Epoch", "Success (%)", "s", "FER (%)", "Final BER (%)",
                        "Mean steps"], rows) + "\n")

        rows = []
        pairs = [(e, e + 1) for e in E[:-1]] + [(1, epochs)]
        for a, b in pairs:
            diffs = [per_seed[s][b]["success"] - per_seed[s][a]["success"] for s in seeds]
            st = stats(diffs)
            rows.append([f"epoch {a} to {b}", fmt_ci(st, 2), fmt(st["sd"], 2),
                         ", ".join(f"{d:+.2f}" for d in diffs)])
        L.append("Increments in success rate, within seed:\n")
        L.append(table(["Increment", "Points", "s",
                        "Per seed " + ", ".join(str(s) for s in seeds)], rows) + "\n")

        rows = []
        for w in WEIGHTS:
            row = [w]
            for e in E:
                row.append(fmt_ci(stats([per_seed[s][e]["by_weight"][w][0] for s in seeds])))
            diffs = [per_seed[s][epochs]["by_weight"][w][0] - per_seed[s][1]["by_weight"][w][0]
                     for s in seeds]
            row.append(fmt_ci(stats(diffs)))
            rows.append(row)
        L.append("Success rate by error weight, with the within-seed gain from "
                 "epoch 1 to the last epoch:\n")
        L.append(table(["Weight"] + [f"Epoch {e}" for e in E] + [f"1 to {epochs}"],
                       rows) + "\n")

        models = sorted(per_seed[seeds[0]][1]["by_model"])
        rows = []
        for m in models:
            row = [f"`{m}`"]
            for e in E:
                row.append(fmt_ci(stats([per_seed[s][e]["by_model"][m][0] for s in seeds])))
            diffs = [per_seed[s][epochs]["by_model"][m][0] - per_seed[s][1]["by_model"][m][0]
                     for s in seeds]
            row.append(fmt_ci(stats(diffs)))
            rows.append(row)
        L.append("Success rate by error model:\n")
        L.append(table(["Error model"] + [f"Epoch {e}" for e in E] + [f"1 to {epochs}"],
                       rows) + "\n")

        reasons = sorted({r for s in seeds for e in E for r in per_seed[s][e]["reasons"]})
        rows = []
        for e in E:
            total = sum(per_seed[s][e]["n"] for s in seeds)
            rows.append([e] + [f"{100.0 * sum(per_seed[s][e]['reasons'].get(r, 0) for s in seeds) / total:.2f}"
                               for r in reasons])
        L.append("Terminal reasons, summed over seeds:\n")
        L.append(table(["Epoch"] + [f"`{r}` (%)" for r in reasons], rows) + "\n")

    for label, a, b, isolates in EPOCH_CONTRASTS:
        if a not in epoch_cells or b not in epoch_cells:
            continue
        epochs = EPOCH_CONDITIONS[a][1]
        rows = []
        for e in range(1, epochs + 1):
            diffs = [epoch_cells[a][s][e]["success"] - epoch_cells[b][s][e]["success"]
                     for s in seeds]
            st = stats(diffs)
            rows.append([e, fmt_ci(st, 2), fmt(st["sd"], 2),
                         ", ".join(f"{d:+.2f}" for d in diffs)])
        L.append(f"### `{label}`: {isolates}\n")
        L.append("Within-seed difference in success rate at each epoch:\n")
        L.append(table(["Epoch", "Points", "s",
                        "Per seed " + ", ".join(str(s) for s in seeds)], rows) + "\n")

    for name, per_seed in controls.items():
        base, epochs, reference = EPOCH_CONTROLS[name]
        if reference not in epoch_cells:
            continue
        L.append(f"### Control `{name}`\n")
        L.append(f"Identical to `{reference}` except that epochs 2 and later open "
                 "at the epsilon floor and hold it, so the later passes add no "
                 "exploration. Reported on the seeds it was run on; a control, not "
                 "a result.\n")
        rows = []
        for s in sorted(per_seed):
            ctl = [per_seed[s][e]["success"] for e in range(1, epochs + 1)]
            ref = [epoch_cells[reference][s][e]["success"] for e in range(1, epochs + 1)]
            rows.append([s, f"`{reference}`"] + [fmt(x, 2) for x in ref] + [f"{ref[-1] - ref[0]:+.2f}"])
            rows.append([s, f"`{name}`"] + [fmt(x, 2) for x in ctl] + [f"{ctl[-1] - ctl[0]:+.2f}"])
            share = (ctl[-1] - ctl[0]) / (ref[-1] - ref[0]) if ref[-1] != ref[0] else float("nan")
            rows.append([s, "share of the gain"] + [""] * epochs + [f"{100 * share:.0f}%"])
        L.append(table(["Seed", "Condition"] + [f"Epoch {e}" for e in range(1, epochs + 1)]
                       + ["Total"], rows) + "\n")

    return "\n".join(L)


def sidecar(cells: dict, epoch_cells: dict, controls: dict, disjoint: dict, seeds: list) -> dict:
    """Every per-cell summary behind the tables, without the pattern lists."""
    def strip(summary: dict) -> dict:
        out = {k: v for k, v in summary.items() if k != "patterns"}
        out["by_weight"] = {str(w): list(v) for w, v in summary["by_weight"].items()}
        out["by_model"] = {m: list(v) for m, v in summary["by_model"].items()}
        return out

    def dis(d: dict) -> dict:
        return {"by_weight": {str(w): list(d[w]) for w in WEIGHTS}, "all": list(d["all"]),
                "by_model": {m: list(v) for m, v in d["by_model"].items()},
                "uniform_by_weight": {str(w): list(v) for w, v in d["uniform"].items()}}

    return {
        "generated": dt.date.today().isoformat(),
        "seeds": seeds,
        "conditions": {c: {str(s): {a: strip(cells[(c, s)][a]) for a in ["rl"] + BASELINES} for s in seeds}
                       for c in CONDITION_ORDER},
        "epoch_conditions": {n: {str(s): {str(e): strip(v) for e, v in per.items()}
                                 for s, per in d.items()} for n, d in epoch_cells.items()},
        "epoch_controls": {n: {str(s): {str(e): strip(v) for e, v in per.items()}
                               for s, per in d.items()} for n, d in controls.items()},
        "overlap": {f"{s}/{ds}": dis(v) for (s, ds), v in disjoint.items()},
    }


# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results-root", default="results/ablation",
                        help="folder holding one results folder per cell")
    parser.add_argument("--data-root", default="data/ablation",
                        help="folder holding s<seed>/curriculum.h5 and flat.h5")
    parser.add_argument("--output", default="docs/ablation/RESULTS.md")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = parser.parse_args()

    results_root = resolve_project_path(args.results_root)
    data_root = resolve_project_path(args.data_root)
    output = resolve_project_path(args.output)
    seeds = args.seeds

    cells = {}
    for s in seeds:
        for c in CONDITION_ORDER:
            cell = f"{c}_s{s}"
            print(f"reading {cell} ...", flush=True)
            cells[(c, s)] = load_cell(results_root, cell)

    # The two cells sharing a dataset must show identical baselines (protocol §6).
    for s in seeds:
        for a, b in [("OnlyC", "Full"), ("NoC_NoE", "OnlyE")]:
            for agent in ["greedy", "min_sum_expert"]:
                if cells[(a, s)][agent]["patterns"] != cells[(b, s)][agent]["patterns"] or \
                        cells[(a, s)][agent]["success"] != cells[(b, s)][agent]["success"]:
                    sys.exit(f"seed {s}: {agent} differs between {a} and {b}, "
                             f"which should share a dataset")

    for agent in ADDED_BASELINES:
        folders = {(s, split): results_root / "baseline_eval" / f"{agent}_{split}_s{s}"
                   for s in seeds for split in ("curriculum", "flat")}
        missing = [f for f in folders.values()
                   if not any(f.glob(f"*_evaluation_agent_{agent}.json.gz"))]
        if missing:
            print(f"{agent}: {len(folders) - len(missing)} of {len(folders)} evaluations "
                  "present, not reported")
            continue
        for (s, split), folder in folders.items():
            print(f"reading {folder.relative_to(results_root)} ...", flush=True)
            summary = summarize_agent(load_episodes(folder, agent))
            for c, (stem, _, _) in CONDITIONS.items():
                if stem != split:
                    continue
                if summary["patterns"] != cells[(c, s)]["rl"]["patterns"]:
                    sys.exit(f"{folder.name}: evaluated on different episodes than {c}_s{s}")
                cells[(c, s)][agent] = summary
        BASELINES.append(agent)

    disjoint = {}
    for s in seeds:
        for ds in ["curriculum", "flat"]:
            path = data_root / f"s{s}" / f"{ds}.h5"
            print(f"reading {path.relative_to(resolve_project_path('.'))} ...", flush=True)
            disjoint[(s, ds)] = disjointness(path)

    epoch_cells = {}
    for name, (base, epochs) in EPOCH_CONDITIONS.items():
        present = [s for s in seeds
                   if epoch_cells_present(results_root, name, s, base, epochs)]
        if len(present) < len(seeds):
            print(f"{name}: {len(present)} of {len(seeds)} seeds present, not reported")
            continue
        epoch_cells[name] = {}
        for s in seeds:
            print(f"reading {name}_s{s} by epoch ...", flush=True)
            epoch_cells[name][s] = load_epoch_cell(results_root, name, s, base, epochs,
                                                   cells[(base, s)])

    controls = {}
    for name, (base, epochs, _reference) in EPOCH_CONTROLS.items():
        present = [s for s in seeds
                   if epoch_cells_present(results_root, name, s, base, epochs)]
        if present:
            controls[name] = {}
            for s in present:
                print(f"reading {name}_s{s} by epoch ...", flush=True)
                controls[name][s] = load_epoch_cell(results_root, name, s, base, epochs,
                                                    cells[(base, s)])

    doc = build_document(cells, seeds, disjoint, Path(args.results_root), Path(args.data_root),
                         epoch_cells, controls)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(doc)
    print(f"wrote {output}")
    side = output.with_suffix(".json")
    side.write_text(json.dumps(sidecar(cells, epoch_cells, controls, disjoint, seeds), indent=1))
    print(f"wrote {side}")


if __name__ == "__main__":
    main()
