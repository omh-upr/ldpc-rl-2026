#!/usr/bin/env python3
"""
Detailed analyses of the ablation and training-budget cells.

Writes docs/ablation/DETAILS.md, a JSON sidecar of the same stem, and
docs/ablation/learning_curves.csv. Like RESULTS.md, the document is generated in
full on every run and never edited by hand. It complements RESULTS.md with the
analyses the headline tables do not carry:

    1. Success and exact recovery: success (STOP on a zero syndrome within the
       flip budget) against whether the final word equals the transmitted one,
       for every cell and agent.
    2. Seen and unseen error patterns: RL success by weight and by whether the
       held-out pattern occurs in the training split, only in the expert replay,
       or in neither.
    3. Agreement with Min-Sum and greedy: frame-aligned contingency tables.
    4. The disagreements: weight, error model and the other decoder's ending.
    5. Agreement with the greedy rule: how often the agent flips a bit of
       maximal penalty, as greedy would.
    6. Training: learning curves, transitions written per epoch, and the share
       of the replay buffer the expert transitions hold.
    7. Resolution of a single run: the seed noise of each cell, the smallest
       difference one run per configuration can detect, and the seeds needed
       for smaller ones.

Evaluation and training files are parsed once into compact arrays cached under
results/ablation_cache/ (keyed by file size and modification time), so a rerun
reads the cache.

Usage:
    PYTHONPATH=. python scripts/analyze_ablation_details.py [--workers 6]
"""

import argparse
import csv
import datetime as dt
import gzip
import json
import math
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

from scripts.aggregate_ablation import (ADDED_BASELINES, CONDITIONS, CONDITION_ORDER, EPOCH_CONDITIONS,
                                        EPOCH_CONTROLS, T_975, epoch_cell_names,
                                        epoch_cells_present, fmt, fmt_ci, stats, table)
from src.utils.path_utils import resolve_project_path

REASONS = ["SUCCESS_STOP", "FAILED_STOP", "MAX_STEPS", "MAX_FLIPS"]
MODELS = ["burst", "cluster", "systematic", "uniform_ber", "uniform_count"]
MAX_STEPS = 12
N_BITS = 128
EXPERT_TRANSITIONS = 500_000
BUFFER_CAPACITY = 5_000_000
BINS_PER_EPOCH = 100
# Differences whose detectability section 7 reports: the gains of the re-flip
# mask and of the uniform epsilon schedule in the development chain, and a
# middle value.
RESOLUTION_DELTAS = [1.43, 0.5, 0.2]
H_PATH = "data/H_64x128_dmin14.npy"


# --------------------------------------------------------------------------
# Parsing and cache
# --------------------------------------------------------------------------

def _reason_code(reason: str) -> int:
    return REASONS.index(reason) if reason in REASONS else len(REASONS)


def parse_eval(path: str) -> dict:
    with gzip.open(path) as f:
        episodes = json.load(f)["episodes"]
    n = len(episodes)
    actions = np.full((n, MAX_STEPS), -1, dtype=np.int16)
    n_steps = np.zeros(n, dtype=np.int8)
    reason = np.zeros(n, dtype=np.int8)
    weight = np.zeros(n, dtype=np.int8)
    model = np.zeros(n, dtype=np.int8)
    pattern = np.zeros((n, N_BITS), dtype=np.uint8)
    noisy = np.zeros((n, N_BITS), dtype=np.uint8)
    original = np.zeros((n, N_BITS), dtype=np.uint8)
    for i, e in enumerate(episodes):
        acts = [s["action"] for s in e["steps"]]
        actions[i, :len(acts)] = acts
        n_steps[i] = len(acts)
        reason[i] = _reason_code(e["terminal_reason"])
        meta = e["error_info"]["meta"]
        weight[i] = int(meta["num_errors"])
        model[i] = MODELS.index(meta["error_model"])
        pattern[i] = e["error_info"]["error_pattern"]
        noisy[i] = e["noisy_codeword"]
        original[i] = e["original_codeword"]
    return {"actions": actions, "n_steps": n_steps, "reason": reason, "weight": weight,
            "model": model, "pattern": np.packbits(pattern, axis=1),
            "noisy": np.packbits(noisy, axis=1), "original": np.packbits(original, axis=1)}


def parse_train(path: str) -> dict:
    with gzip.open(path) as f:
        episodes = json.load(f)["training_episodes"]
    return {
        "episode_id": np.array([e["episode_id"] for e in episodes], dtype=np.int64),
        "epoch": np.array([e["epoch"] for e in episodes], dtype=np.int8),
        "phase": np.array([e["phase"] for e in episodes], dtype=np.int8),
        "n_steps": np.array([len(e["steps"]) for e in episodes], dtype=np.int8),
        "reason": np.array([_reason_code(e["terminal_reason"]) for e in episodes], dtype=np.int8),
    }


def cache_path(cache_root: Path, src: Path) -> Path:
    st = src.stat()
    return cache_root / src.parent.name / f"{src.name}.{st.st_size}.{int(st.st_mtime)}.npz"


def _load_job(args):
    kind, src, dst = args
    dst = Path(dst)
    if dst.exists():
        return str(dst)
    data = parse_eval(src) if kind == "eval" else parse_train(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **data)
    tmp.rename(dst)
    return str(dst)


def load_many(jobs: list, cache_root: Path, workers: int) -> dict:
    """jobs: list of (kind, source path). Returns {source path: arrays}."""
    todo = [(kind, str(src), str(cache_path(cache_root, src))) for kind, src in jobs]
    missing = [j for j in todo if not Path(j[2]).exists()]
    if missing:
        print(f"parsing {len(missing)} files into the cache ...", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for k, done in enumerate(pool.map(_load_job, missing), 1):
                print(f"  [{k}/{len(missing)}] {Path(done).parent.name}/{Path(done).name.split('.json')[0]}",
                      flush=True)
    return {src: dict(np.load(dst)) for _kind, src, dst in todo}


def eval_file(cell_dir: Path, agent: str) -> Path:
    files = sorted(cell_dir.glob(f"*_evaluation_agent_{agent}.json.gz"))
    if len(files) != 1:
        sys.exit(f"{cell_dir}: expected one evaluation file for {agent}, found {len(files)}")
    return files[0]


def train_files(cell_dir: Path) -> list:
    files = sorted(cell_dir.glob("*_training_epoch*_phase*.json.gz"))
    if not files:
        sys.exit(f"{cell_dir}: no training files")
    return files


# --------------------------------------------------------------------------
# Derived quantities
# --------------------------------------------------------------------------

def final_wrong_bits(ev: dict) -> np.ndarray:
    word = np.unpackbits(ev["noisy"], axis=1)[:, :N_BITS].astype(np.uint8)
    rows = np.arange(len(word))
    for t in range(MAX_STEPS):
        a = ev["actions"][:, t]
        flip = (a >= 0) & (a < N_BITS)
        word[rows[flip], a[flip]] ^= 1
    orig = np.unpackbits(ev["original"], axis=1)[:, :N_BITS]
    return (word != orig).sum(axis=1)


def success(ev: dict) -> np.ndarray:
    return ev["reason"] == 0


def keys(packed: np.ndarray) -> np.ndarray:
    return packed.view(np.dtype((np.void, packed.shape[1]))).ravel()


def dataset_patterns(path: Path, split: str = None, first: int = None) -> set:
    with h5py.File(path, "r") as h:
        phases = sorted((k for k in h if k.startswith("phase_")), key=lambda k: int(k.split("_")[1]))
        patterns = np.concatenate([h[f"{p}/error_patterns"][:] for p in phases])
        if split is not None:
            patterns = patterns[h[f"splits/{split}_indices"][:]]
    if first is not None:
        patterns = patterns[:first]
    return set(keys(np.packbits(patterns.astype(np.uint8), axis=1)).tolist())


def expert_episode_count(path: Path) -> int:
    """Episodes of the expert dataset, in stored order, that the 500,000-transition
    replay holds. Each oracle episode writes weight + 1 transitions (the flips and
    the STOP), and generation stops as soon as the target is reached."""
    with h5py.File(path, "r") as h:
        phases = sorted((k for k in h if k.startswith("phase_")), key=lambda k: int(k.split("_")[1]))
        weights = np.concatenate([h[f"{p}/error_patterns"][:].sum(axis=1) for p in phases])
    cum = np.cumsum(weights + 1)
    return int(np.searchsorted(cum, EXPERT_TRANSITIONS) + 1)


def greedy_agreement(ev: dict, H: np.ndarray) -> dict:
    """For every flip the agent made, whether it chose a bit of maximal penalty,
    and whether it chose greedy's exact action (lowest index among the maxima,
    after removing the bit flipped on the previous step)."""
    n = len(ev["actions"])
    word = np.unpackbits(ev["noisy"], axis=1)[:, :N_BITS].astype(np.int64)
    last = np.full(n, -1)
    counts = Counter()
    by_step = {}
    for t in range(MAX_STEPS):
        a = ev["actions"][:, t].astype(np.int64)
        flip = (a >= 0) & (a < N_BITS)
        if not flip.any():
            break
        idx = np.flatnonzero(flip)
        z = (word[idx] @ H.T) % 2
        p = z @ H                                   # unsatisfied checks per bit
        prev = last[idx]
        has_prev = prev >= 0
        p_legal = p.copy()
        p_legal[np.flatnonzero(has_prev), prev[has_prev]] = -1
        pmax = p_legal.max(axis=1)
        chosen = p[np.arange(len(idx)), a[idx]]
        is_max = chosen == pmax
        is_greedy = a[idx] == p_legal.argmax(axis=1)
        zero_pen = chosen == 0
        counts["flips"] += len(idx)
        counts["max_penalty"] += int(is_max.sum())
        counts["greedy_action"] += int(is_greedy.sum())
        counts["zero_penalty"] += int(zero_pen.sum())
        by_step[t + 1] = (len(idx), float(is_max.mean()), float(is_greedy.mean()))
        word[idx, a[idx]] ^= 1
        last[idx] = a[idx]
    return {"counts": dict(counts), "by_step": by_step}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results-root", default="results/ablation")
    parser.add_argument("--data-root", default="data/ablation")
    parser.add_argument("--cache-root", default="results/ablation_cache")
    parser.add_argument("--output", default="docs/ablation/DETAILS.md")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--skip-training", action="store_true",
                        help="omit section 6, which parses every training file")
    args = parser.parse_args()

    results_root = resolve_project_path(args.results_root)
    data_root = resolve_project_path(args.data_root)
    cache_root = resolve_project_path(args.cache_root)
    output = resolve_project_path(args.output)
    seeds = args.seeds
    H = np.load(resolve_project_path(H_PATH)).astype(np.int64) % 2

    # Which multi-epoch conditions are complete.
    epoch_conds = {name: spec for name, spec in EPOCH_CONDITIONS.items()
                   if all(epoch_cells_present(results_root, name, s, *spec) for s in seeds)}

    # ---- evaluation files
    # Baselines added after the campaign are read from their evaluation-only
    # folders, one per seed and dataset, once all of them exist.
    added = [a for a in ADDED_BASELINES
             if all(any((results_root / "baseline_eval" / f"{a}_{split}_s{s}")
                        .glob(f"*_evaluation_agent_{a}.json.gz"))
                    for s in seeds for split in ("curriculum", "flat"))]
    baselines = ["greedy", "min_sum_expert"] + added
    jobs, ev_index = [], {}
    for c in CONDITION_ORDER:
        for s in seeds:
            for agent in ["rl"] + baselines:
                folder = (results_root / "baseline_eval" / f"{agent}_{CONDITIONS[c][0]}_s{s}"
                          if agent in added else results_root / f"{c}_s{s}")
                src = eval_file(folder, agent)
                jobs.append(("eval", src))
                ev_index[(c, s, 1, agent)] = str(src)
    for name, (base, epochs) in epoch_conds.items():
        for s in seeds:
            for e, folder in epoch_cell_names(name, s, base, epochs).items():
                if e == 1:
                    for agent in ["rl"] + baselines:
                        ev_index[(name, s, 1, agent)] = ev_index[(base, s, 1, agent)]
                    continue
                src = eval_file(results_root / folder, "rl")
                jobs.append(("eval", src))
                ev_index[(name, s, e, "rl")] = str(src)
                for agent in baselines:
                    ev_index[(name, s, e, agent)] = ev_index[(base, s, 1, agent)]
    ev = load_many(jobs, cache_root, args.workers)
    get = lambda cond, s, e, agent: ev[ev_index[(cond, s, e, agent)]]

    # Every epoch was scored on the base cell's episodes, in the same order.
    for (cond, s, e, agent), src in ev_index.items():
        base = EPOCH_CONDITIONS.get(cond, (cond,))[0]
        if not np.array_equal(ev[src]["pattern"], get(base, s, 1, "rl")["pattern"]):
            sys.exit(f"{cond} seed {s} epoch {e} {agent}: episodes not aligned with {base}")

    # Rows reported: the four one-epoch conditions, then each epoch of the
    # multi-epoch conditions after the first (epoch 1 is the base condition).
    rows_spec = [(c, 1, f"`{c}`") for c in CONDITION_ORDER]
    for name, (base, epochs) in epoch_conds.items():
        rows_spec += [(name, e, f"`{name}` epoch {e}") for e in range(2, epochs + 1)]

    record = {"generated": dt.date.today().isoformat(), "seeds": seeds}
    L = ["# Ablation Details: Recovery, Unseen Patterns, Disagreement and Training\n",
         f"Generated by `scripts/analyze_ablation_details.py` on {record['generated']} "
         f"from the evaluation and training files under `{args.results_root}/` and the "
         f"datasets under `{args.data_root}/`. Do not edit; rerun the script. "
         "Definitions, conditions and the headline tables are in `RESULTS.md`; the "
         "numbers here are also in the JSON file of the same name.\n"]

    # ---- 1. success and exact recovery
    L.append("## 1. Success and exact recovery\n")
    L.append("Success is STOP on a zero syndrome within the flip budget. Exact recovery "
             "is a final word equal to the transmitted one, whether or not the decoder "
             "stopped. With $d_{\\min} = 14$, error weight at most 6 and flip budget 7, "
             "a success is exact recovery by construction, so the first count must be "
             "zero. The second is the frames corrected without a STOP, typically on the "
             "last permitted step. Counts summed over seeds.\n")
    rows, rec1 = [], {}
    for cond, e, label in rows_spec:
        for agent in (["rl"] + baselines if e == 1 else ["rl"]):
            succ_wrong = recovered_no_success = total = succ = 0
            for s in seeds:
                x = get(cond, s, e, agent)
                ok = success(x)
                wrong = final_wrong_bits(x)
                succ_wrong += int((ok & (wrong > 0)).sum())
                recovered_no_success += int((~ok & (wrong == 0)).sum())
                succ += int(ok.sum())
                total += len(ok)
            rec1[f"{cond}/{e}/{agent}"] = {"success": succ, "success_not_recovered": succ_wrong,
                                           "recovered_without_success": recovered_no_success,
                                           "episodes": total}
            if agent != "rl" and cond not in ("Full", "OnlyE"):
                continue
            name = label if agent == "rl" else f"{label} {agent}"
            rows.append([name, f"{total:,}", f"{succ:,}", succ_wrong, f"{recovered_no_success:,}",
                         f"{100.0 * recovered_no_success / total:.2f}"])
    L.append(table(["Cell and agent", "Episodes", "Successes", "Success, wrong word",
                    "Recovered, no success", "Recovered, no success (%)"], rows) + "\n")
    record["exact_recovery"] = rec1
    if any(v["success_not_recovered"] for v in rec1.values()):
        print("WARNING: a success ended on a wrong word", file=sys.stderr)

    # ---- 2. seen and unseen patterns
    print("computing pattern overlap ...", flush=True)
    seen_sets = {}
    expert_n = {}
    for s in seeds:
        expert_path = data_root / f"s{s}" / "expert.h5"
        expert_n[s] = expert_episode_count(expert_path)
        expert_keys = dataset_patterns(expert_path, first=expert_n[s])
        for kind in ["curriculum", "flat"]:
            seen_sets[(s, kind)] = (dataset_patterns(data_root / f"s{s}" / f"{kind}.h5", "train"),
                                    expert_keys)
    record["expert_episodes_in_replay"] = {str(s): n for s, n in expert_n.items()}

    def category(cond, s):
        """0 in training split, 1 only in the expert replay, 2 in neither."""
        base = EPOCH_CONDITIONS.get(cond, (cond,))[0]
        kind = CONDITIONS[base][0]
        train_keys, expert_keys = seen_sets[(s, kind)]
        k = keys(get(base, s, 1, "rl")["pattern"]).tolist()
        in_train = np.array([x in train_keys for x in k])
        in_expert = np.array([x in expert_keys for x in k])
        return np.where(in_train, 0, np.where(in_expert, 1, 2))

    cats = {(c, s): category(c, s) for c in CONDITION_ORDER for s in seeds}
    CAT = ["in training split", "only in expert replay", "in neither"]
    L.append("## 2. Seen and unseen error patterns\n")
    L.append("Every held-out pattern is placed in one of three groups: it occurs, as the "
             "same set of flipped positions, in the training split of the dataset the "
             "cell trained on; it occurs only among the patterns of the expert replay "
             f"(the first {min(expert_n.values()):,} to {max(expert_n.values()):,} words "
             "of the seed's expert dataset, which is what 500,000 oracle transitions "
             "cover); or it occurs in neither. RL success rate by group and weight, mean "
             "over seeds with the 95% interval. For conditions without expert seeding the "
             "second group is still shown; the replay was never loaded there, so it is a "
             "control for the grouping itself. The count is seed 42 on the curriculum "
             "dataset. Weights below 3 are omitted: nearly every such pattern is in the "
             "training split (RESULTS.md section 7).\n")
    rows, rec2 = [], {}
    for w in [3, 4, 5, 6]:
        for g, gname in enumerate(CAT):
            cnt = int(((cats[("Full", seeds[0])] == g)
                       & (get("Full", seeds[0], 1, "rl")["weight"] == w)).sum())
            row = [w, gname, f"{cnt:,}"]
            for cond, e, label in rows_spec:
                base = EPOCH_CONDITIONS.get(cond, (cond,))[0]
                vals = []
                for s in seeds:
                    x = get(cond, s, e, "rl")
                    m = (cats[(base, s)] == g) & (x["weight"] == w)
                    vals.append(100.0 * success(x)[m].mean() if m.sum() >= 20 else float("nan"))
                vals_ok = [v for v in vals if not math.isnan(v)]
                st = stats(vals_ok) if len(vals_ok) >= 2 else None
                row.append(fmt_ci(st) if st else "n/a")
                rec2[f"{cond}/{e}/w{w}/{g}"] = vals
            for agent in ["greedy", "min_sum_expert"]:
                # Baselines on each dataset's held-out words; only the curriculum
                # ones are shown, the flat ones go to the record.
                for ref in ("Full", "NoC_NoE"):
                    vals = []
                    for s in seeds:
                        x = get(ref, s, 1, agent)
                        m = (cats[(ref, s)] == g) & (x["weight"] == w)
                        vals.append(100.0 * success(x)[m].mean() if m.sum() >= 20 else float("nan"))
                    rec2[f"{ref}/1/{agent}/w{w}/{g}"] = vals
                    if ref == "Full":
                        vals_ok = [v for v in vals if not math.isnan(v)]
                        row.append(fmt(float(np.mean(vals_ok))) if vals_ok else "n/a")
            rows.append(row)
    L.append(table(["Weight", "Pattern", "Count"] + [lab for _, _, lab in rows_spec]
                   + ["Greedy", "Min-sum"], rows) + "\n")
    L.append("Groups with fewer than 20 patterns in a seed are left out of that seed's "
             "value; `n/a` means fewer than two seeds remained.\n")

    # Within-seed gap between unseen and training-split patterns, weights 4 to 6.
    L.append("Within-seed difference between patterns in neither set and patterns in the "
             "training split, weights 4 to 6 pooled, in points (negative means worse on "
             "unseen patterns):\n")
    rows = []
    for cond, e, label in rows_spec:
        base = EPOCH_CONDITIONS.get(cond, (cond,))[0]
        diffs = []
        for s in seeds:
            x = get(cond, s, e, "rl")
            heavy = x["weight"] >= 4
            a = success(x)[heavy & (cats[(base, s)] == 2)].mean()
            b = success(x)[heavy & (cats[(base, s)] == 0)].mean()
            diffs.append(100.0 * (a - b))
        rows.append([label, fmt_ci(stats(diffs)), ", ".join(f"{d:+.1f}" for d in diffs)])
        rec2[f"{cond}/{e}/gap_w4_6"] = diffs
    L.append(table(["Cell", "Unseen minus seen (points)",
                    "Per seed " + ", ".join(map(str, seeds))], rows) + "\n")
    record["seen_unseen"] = rec2

    # The pooled gap mixes exposure with error model: burst and systematic
    # patterns are all in the training split by construction and heavy uniform
    # patterns almost never are. Compare within one model and weight instead.
    L.append("The pooled gap above confounds exposure with error model: burst and "
             "systematic patterns all occur in training by construction, while heavy "
             "unstructured patterns almost never do, and the models differ in "
             "difficulty. Greedy, which learns nothing, shows the confound on its own. "
             "Within one error model and weight the comparison is clean. Below, every "
             "(model, weight) cell in which each seed has at least 20 held-out patterns "
             "both inside and outside the training split: success on patterns outside "
             "minus inside, within seed, in points. Counts are seed 42 on the curriculum "
             "dataset. Burst and systematic patterns never appear here, because none of "
             "them is unseen.\n")

    def within_model_table(kind: str, ref: str) -> tuple:
        """Unseen minus seen within one (model, weight) cell, for the conditions
        trained on one dataset; `ref` is a one-epoch condition on that dataset,
        whose episodes and baselines every such condition shares."""
        rows, rec = [], {}
        for mi, mname in enumerate(MODELS):
            for w in range(1, 7):
                sel = {s: (get(ref, s, 1, "rl")["model"] == mi) & (get(ref, s, 1, "rl")["weight"] == w)
                       for s in seeds}
                in_tr = {s: cats[(ref, s)] == 0 for s in seeds}
                if not all((sel[s] & in_tr[s]).sum() >= 20 and (sel[s] & ~in_tr[s]).sum() >= 20
                           for s in seeds):
                    continue
                s0 = seeds[0]
                row = [f"`{mname}`", w, f"{int((sel[s0] & in_tr[s0]).sum()):,} / {int((sel[s0] & ~in_tr[s0]).sum()):,}"]
                for cond, e, _label in rows_spec:
                    if CONDITIONS[EPOCH_CONDITIONS.get(cond, (cond,))[0]][0] != kind:
                        continue
                    diffs = []
                    for s in seeds:
                        ok = success(get(cond, s, e, "rl"))
                        diffs.append(100.0 * (ok[sel[s] & ~in_tr[s]].mean() - ok[sel[s] & in_tr[s]].mean()))
                    row.append(fmt_ci(stats(diffs)))
                    rec[f"{mname}/w{w}/{cond}/{e}"] = diffs
                for agent in ["greedy", "min_sum_expert"]:
                    diffs = []
                    for s in seeds:
                        ok = success(get(ref, s, 1, agent))
                        diffs.append(100.0 * (ok[sel[s] & ~in_tr[s]].mean() - ok[sel[s] & in_tr[s]].mean()))
                    row.append(fmt(float(np.mean(diffs))))
                    rec[f"{mname}/w{w}/{agent}"] = diffs
                rows.append(row)
        labels = [lab for cond, e, lab in rows_spec
                  if CONDITIONS[EPOCH_CONDITIONS.get(cond, (cond,))[0]][0] == kind]
        return table(["Error model", "Weight", "Seen / unseen"] + labels
                     + ["Greedy", "Min-sum"], rows), rec

    text, record["seen_unseen_within_model"] = within_model_table("curriculum", "Full")
    L.append(text + "\n")
    L.append("The same comparison for the conditions trained on the flat dataset, whose "
             "held-out split holds different words drawn from the same mixture. Counts "
             "are seed 42 on the flat dataset.\n")
    text, record["seen_unseen_within_model_flat"] = within_model_table("flat", "NoC_NoE")
    L.append(text + "\n")

    # ---- 3. contingency
    L.append("## 3. Agreement with Min-Sum and greedy\n")
    L.append("Frame-aligned outcomes of the RL agent against each baseline on the same "
             "held-out words, summed over seeds. Success as in `RESULTS.md`.\n")
    rows, rec3 = [], {}
    # The one-epoch cell under each multi-epoch condition, then its later epochs.
    bases = [b for b, _e in epoch_conds.values()] or ["Full"]
    cont_rows = ([(b, 1, f"`{b}`") for b in CONDITION_ORDER if b in bases]
                 + [r for r in rows_spec if r[0] in epoch_conds])
    for cond, e, label in cont_rows:
        for agent in ["min_sum_expert"] + added + ["greedy"]:
            tab = Counter()
            per_seed = []
            for s in seeds:
                r = success(get(cond, s, e, "rl"))
                b = success(get(cond, s, e, agent))
                c = {"both": int((r & b).sum()), "rl_only": int((r & ~b).sum()),
                     "baseline_only": int((~r & b).sum()), "neither": int((~r & ~b).sum())}
                per_seed.append(c)
                tab.update(c)
            rec3[f"{cond}/{e}/{agent}"] = {"pooled": dict(tab), "per_seed": per_seed}
            total = sum(tab.values())
            rows.append([label, {"min_sum_expert": "Min-sum", "greedy": "Greedy",
                                 "gallager_b": "Gallager-B"}[agent],
                         f"{tab['both']:,}", f"{tab['rl_only']:,}", f"{tab['baseline_only']:,}",
                         f"{tab['neither']:,}",
                         ", ".join(str(c["rl_only"]) for c in per_seed)])
    L.append(table(["RL cell", "Baseline", "Both succeed", "RL only", "Baseline only",
                    "Neither", "RL only, per seed"], rows) + "\n")
    record["contingency"] = rec3

    # ---- 4. disagreements characterised, at the last epoch of every
    # multi-epoch condition (the one-epoch Full cell if there is none).
    finals = ([(name, epochs, f"`{name}` epoch {epochs}") for name, (_b, epochs) in epoch_conds.items()]
              or [("Full", 1, "`Full`")])
    L.append("## 4. The disagreements with Min-Sum\n")
    L.append("Pooled over seeds, at the last epoch of each multi-epoch condition. *RL "
             "only* are frames the agent recovers and Min-Sum does not; *Min-sum only* "
             "the reverse. The ending is the terminal reason of the decoder that failed: "
             "for Min-Sum, `FAILED_STOP` means the native decoder did not reach a zero "
             "syndrome within 20 iterations, so the adapter stopped at once, and "
             "`MAX_FLIPS` means it converged to a word more than seven flips away, which "
             "is a wrong codeword.\n")
    record["disagreements"] = {}
    for cond, e, label in finals:
        L.append(f"### {label}\n")
        rec4 = {}
        for side, mask_fn, failer in [
                ("RL only", lambda r, b: r & ~b, "min_sum_expert"),
                ("Min-sum only", lambda r, b: ~r & b, "rl"),
                ("neither", lambda r, b: ~r & ~b, "min_sum_expert")]:
            w_cnt, m_cnt, why = Counter(), Counter(), Counter()
            why_rl = Counter()
            for s in seeds:
                x_rl, x_b = get(cond, s, e, "rl"), get(cond, s, e, "min_sum_expert")
                m = mask_fn(success(x_rl), success(x_b))
                w_cnt.update(x_rl["weight"][m].tolist())
                m_cnt.update(MODELS[i] for i in x_rl["model"][m].tolist())
                fx = x_b if failer == "min_sum_expert" else x_rl
                why.update(REASONS[i] if i < len(REASONS) else "OTHER" for i in fx["reason"][m].tolist())
                if side == "neither":
                    why_rl.update(REASONS[i] for i in x_rl["reason"][m].tolist())
            n = sum(w_cnt.values())
            rec4[side] = {"n": n, "weight": dict(w_cnt), "model": dict(m_cnt), "ending": dict(why),
                          "rl_ending": dict(why_rl)}
            L.append(f"**{side}** ({n:,} frames). Weight: "
                     + ", ".join(f"{w}: {c}" for w, c in sorted(w_cnt.items()))
                     + ". Model: " + ", ".join(f"`{k}` {c}" for k, c in sorted(m_cnt.items()))
                     + f". Ending of {'Min-Sum' if failer == 'min_sum_expert' else 'the agent'}: "
                     + ", ".join(f"`{k}` {c}" for k, c in sorted(why.items()))
                     + (". Ending of the agent: " + ", ".join(f"`{k}` {c}" for k, c in sorted(why_rl.items()))
                        if why_rl else "") + ".\n")
        # Persistence of RL-only frames across epochs of the same cell.
        if cond in epoch_conds:
            keep, first_total = 0, 0
            for s in seeds:
                r1 = success(get(cond, s, 1, "rl")) & ~success(get(cond, s, 1, "min_sum_expert"))
                rl = success(get(cond, s, e, "rl")) & ~success(get(cond, s, e, "min_sum_expert"))
                keep += int((r1 & rl).sum())
                first_total += int(r1.sum())
            rec4["rl_only_persisting_from_epoch1"] = [keep, first_total]
            L.append(f"Of the {first_total} RL-only frames at epoch 1, {keep} are still RL-only "
                     f"at epoch {e} (same held-out words).\n")
        record["disagreements"][f"{cond}/{e}"] = rec4

    # ---- 5. agreement with the greedy rule
    print("computing agreement with the greedy rule ...", flush=True)
    L.append("## 5. Agreement with the greedy rule\n")
    L.append("For every flip the agent made on the held-out words, whether the flipped bit "
             "had the largest number of unsatisfied checks among the legal bits (the "
             "greedy criterion), whether it was greedy's exact choice (lowest index among "
             "those), and whether it had no unsatisfied check at all. Percentages of "
             "flips, pooled over seeds; the greedy decoder itself is 100, 100 and 0 by "
             "construction.\n")
    rows, rec5 = [], {}
    for cond, e, label in rows_spec:
        tot = Counter()
        for s in seeds:
            tot.update(greedy_agreement(get(cond, s, e, "rl"), H)["counts"])
        rec5[f"{cond}/{e}"] = dict(tot)
        rows.append([label, f"{tot['flips']:,}",
                     f"{100.0 * tot['max_penalty'] / tot['flips']:.1f}",
                     f"{100.0 * tot['greedy_action'] / tot['flips']:.1f}",
                     f"{100.0 * tot['zero_penalty'] / tot['flips']:.2f}"])
    L.append(table(["Cell", "Flips", "Maximal penalty (%)", "Greedy's action (%)",
                    "Zero penalty (%)"], rows) + "\n")
    record["greedy_agreement"] = rec5

    # ---- 6. training
    if not args.skip_training:
        print("reading training files ...", flush=True)
        train_cells = [(c, s) for c in CONDITION_ORDER for s in seeds]
        train_cells += [(name, s) for name in epoch_conds for s in seeds]
        for name, (base, epochs, _ref) in EPOCH_CONTROLS.items():
            train_cells += [(name, s) for s in seeds
                            if (results_root / f"{name}_s{s}").is_dir()]
        jobs, tf_index = [], {}
        for cond, s in train_cells:
            files = train_files(results_root / f"{cond}_s{s}")
            tf_index[(cond, s)] = [str(f) for f in files]
            jobs += [("train", f) for f in files]
        tr = load_many(jobs, cache_root, args.workers)

        curve_rows, rec6 = [], {}
        for cond, s in train_cells:
            parts = [tr[f] for f in tf_index[(cond, s)]]
            cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
            order = np.argsort(cat["episode_id"], kind="stable")
            cat = {k: v[order] for k, v in cat.items()}
            seeded = CONDITIONS[EPOCH_CONDITIONS.get(cond, EPOCH_CONTROLS.get(cond, (cond,)))[0]][2]
            per_epoch = {}
            cumulative = 0
            for ep in sorted(set(cat["epoch"].tolist())):
                m = cat["epoch"] == ep
                steps = cat["n_steps"][m].astype(np.int64)
                succ = cat["reason"][m] == 0
                transitions = int(steps.sum())
                cumulative += transitions
                in_buffer = EXPERT_TRANSITIONS if seeded else 0
                if seeded and EXPERT_TRANSITIONS + cumulative > BUFFER_CAPACITY:
                    in_buffer = max(0, BUFFER_CAPACITY - cumulative)
                per_epoch[int(ep)] = {
                    "episodes": int(m.sum()), "transitions": transitions,
                    "training_success": float(100.0 * succ.mean()),
                    "expert_transitions_left_at_end": in_buffer,
                }
                n_ep = int(m.sum())
                edges = np.linspace(0, n_ep, BINS_PER_EPOCH + 1).astype(int)
                ph = cat["phase"][m]
                for b in range(BINS_PER_EPOCH):
                    sl = slice(edges[b], edges[b + 1])
                    curve_rows.append([cond, s, int(ep), b, edges[b + 1] - edges[b],
                                       f"{100.0 * succ[sl].mean():.3f}",
                                       f"{steps[sl].mean():.3f}", int(np.bincount(ph[sl]).argmax())])
            rec6[f"{cond}/{s}"] = per_epoch
        record["training"] = rec6

        curves = output.parent / "learning_curves.csv"
        with open(curves, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["condition", "seed", "epoch", "bin", "episodes",
                        "training_success_pct", "mean_steps", "phase"])
            w.writerows(curve_rows)

        L.append("## 6. Training\n")
        L.append("Per epoch, summed or averaged over seeds. *Training success* is the "
                 "success rate of the training episodes themselves, under exploration and "
                 "on the phase mixture in force, so it is a trace of the run rather than a "
                 "measure of the policy. *Transitions* is what the epoch wrote into the "
                 f"{BUFFER_CAPACITY:,}-transition circular buffer. The buffer overwrites its "
                 "oldest entries first, which are the expert transitions, so *expert left* "
                 "is how many of the 500,000 remain at the end of the epoch. Per-bin curves "
                 f"({BINS_PER_EPOCH} bins per epoch) are in `learning_curves.csv`.\n")
        rows = []
        for cond in list(CONDITION_ORDER) + list(epoch_conds) + list(EPOCH_CONTROLS):
            cells = [k for k in rec6 if k.split("/")[0] == cond]
            if not cells:
                continue
            epochs_here = sorted({ep for k in cells for ep in rec6[k]})
            for ep in epochs_here:
                tr_s = [rec6[k][ep]["training_success"] for k in cells]
                trans = [rec6[k][ep]["transitions"] for k in cells]
                left = [rec6[k][ep]["expert_transitions_left_at_end"] for k in cells]
                rows.append([f"`{cond}`", ep, len(cells),
                             fmt_ci(stats(tr_s)) if len(cells) > 1 else fmt(tr_s[0]),
                             f"{np.mean(trans):,.0f}", f"{np.mean(left):,.0f}"])
        L.append(table(["Condition", "Epoch", "Seeds", "Training success (%)",
                        "Transitions (mean)", "Expert left (mean)"], rows) + "\n")

    # ---- 7. resolution of a single run
    L.append("## 7. Resolution of a single run\n")
    L.append("How large a difference one training run can detect. *s* is the standard "
             "deviation of held-out RL success over the seeds: the noise a single run "
             "carries. Two single runs, one per configuration, then differ by seed noise "
             "alone with standard deviation about √2·s, if their seed effects are "
             "independent; the second table checks that against contrasts actually "
             "measured. *Resolution* is 1.96·√2·s, the smallest difference that seed "
             "noise alone exceeds in fewer than 5% of such pairs. *Seeds for δ* is the "
             "number of seeds per arm at which the 95% interval of the mean difference "
             "narrows to ±δ, (1.96·√2·s/δ)² rounded up; it is a normal approximation, "
             "and with Student t and few seeds the number is larger. "
             f"δ = {RESOLUTION_DELTAS[0]} is the gain of the re-flip mask in the "
             "development chain (`DEVELOPMENT_CHAIN.json`), "
             f"δ = {RESOLUTION_DELTAS[-1]} that of the uniform epsilon schedule.\n")
    rate = {(cond, s, e): 100.0 * success(get(cond, s, e, "rl")).mean()
            for cond, e, _l in rows_spec for s in seeds}
    for name, (base, _epochs) in epoch_conds.items():
        for s in seeds:
            rate[(name, s, 1)] = rate[(base, s, 1)]
    rows, rec7 = [], {}
    for cond, e, label in rows_spec:
        sd = stats([rate[(cond, s, e)] for s in seeds])["sd"]
        sigma = math.sqrt(2.0) * sd
        need = {d: max(1, math.ceil((1.96 * sigma / d) ** 2)) for d in RESOLUTION_DELTAS}
        rec7[f"{cond}/{e}"] = {"sd": sd, "resolution": 1.96 * sigma,
                               "seeds_for": {str(d): n for d, n in need.items()}}
        rows.append([label, f"{sd:.2f}", f"{1.96 * sigma:.2f}"] + [f"{need[d]:,}" for d in RESOLUTION_DELTAS])
    L.append(table(["Cell", "s (points)", "Resolution (points)"]
                   + [f"Seeds for δ = {d}" for d in RESOLUTION_DELTAS], rows) + "\n")

    L.append("Measured within-seed contrasts between conditions that differ only by the "
             "curriculum, the one factor that changes little, against the √2·s the table "
             "above assumes. Where the measured spread is near or above the prediction, "
             "running both configurations on the same seed buys nothing, and the table "
             "above, if anything, understates the noise of a paired comparison.\n")
    checks = [("Full", "OnlyE", 1), ("OnlyC", "NoC_NoE", 1)]
    if {"OnlyC3ep", "NoC_NoE3ep"} <= set(epoch_conds):
        checks += [("OnlyC3ep", "NoC_NoE3ep", e) for e in range(2, epoch_conds["OnlyC3ep"][1] + 1)]
    rows, rec7b = [], {}
    for a, b, e in checks:
        diffs = [rate[(a, s, e)] - rate[(b, s, e)] for s in seeds]
        measured = stats(diffs)["sd"]
        predicted = math.sqrt(2.0) * math.sqrt(0.5 * (rec7[f"{a}/{e}"]["sd"] ** 2 + rec7[f"{b}/{e}"]["sd"] ** 2))
        rec7b[f"{a} - {b}/{e}"] = {"measured_sd": measured, "predicted_sd": predicted}
        rows.append([f"`{a} - {b}`", e, f"{measured:.2f}", f"{predicted:.2f}", f"{1.96 * measured:.2f}"])
    L.append(table(["Contrast", "Epoch", "Measured s (points)", "Predicted √2·s (points)",
                    "Resolution from measured s"], rows) + "\n")
    record["resolution"] = {"cells": rec7, "contrasts": rec7b, "deltas": RESOLUTION_DELTAS}

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(L))
    output.with_suffix(".json").write_text(json.dumps(record, indent=1, default=int))
    print(f"wrote {output} and {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
