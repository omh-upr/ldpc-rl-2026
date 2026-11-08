#!/usr/bin/env python3
"""Write the paper's tables to <out>/tables/ (see paperlib for <out>).

Every table is a Quarto pipe table with its caption and cross-reference label,
included by the section files. Three tables are in the main text (`results`,
`contrasts`, `performance`); every other table is included by the supplement.
Tables name conditions by ordering and seeding (`P.LABEL`), never by
repository ID. Run from the project root:

    PYTHONPATH=. python scripts/paper/make_tables.py [--out DIR]

Tables that need DETAILS.json are skipped, with a message, until it exists.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paperlib as P  # noqa: E402
from src.core.ldpc_utils import get_ldpc, load_matrices  # noqa: E402

SRC = "scripts/paper/make_tables.py"
HEAD, EPOCH = P.HEADLINE
BASE = HEAD_BASE = P.HEADLINE_BASE
# The four contrasts of the 2x2, as (the aid switched on, the level at which the
# other aid is held fixed, minuend, subtrahend), named by the one-epoch
# conditions; the main table continues each through the epochs. Two columns, not
# one label: "Curriculum, seeded" would read as the condition of that name.
# Reader-facing names of the error models, as in the error-model table.
MODEL_NAME = {"uniform_count": "Uniform count", "uniform_ber": "Uniform BER", "burst": "Burst",
              "cluster": "Cluster", "systematic": "Systematic"}
CONTRASTS = [("Seeding", "curriculum", "Full", "OnlyC"),
             ("Seeding", "flat", "OnlyE", "NoC_NoE"),
             ("Curriculum", "seeded", "Full", "OnlyE"),
             ("Curriculum", "unseeded", "OnlyC", "NoC_NoE")]
# A seeded curriculum cell: it carries every setting the tables describe, the
# curriculum and the expert replay included. The other cells differ only there.
CONFIG_CELL = P.ROOT / "config/ablation_config/Full3ep_s42"


def girth(H: np.ndarray) -> int:
    m, n = H.shape
    adj_v = [np.flatnonzero(H[:, j]) for j in range(n)]
    adj_c = [np.flatnonzero(H[i]) for i in range(m)]
    best = 10 ** 9
    for s in range(n):
        dist, parent, queue = {("v", s): 0}, {("v", s): None}, [("v", s)]
        for u in queue:
            nbrs = ([("c", i) for i in adj_v[u[1]]] if u[0] == "v"
                    else [("v", j) for j in adj_c[u[1]]])
            for w in nbrs:
                if w == parent[u]:
                    continue
                if w in dist:
                    best = min(best, dist[u] + dist[w] + 1)
                else:
                    dist[w], parent[w] = dist[u] + 1, u
                    queue.append(w)
    return best


def code_table():
    H, _ = load_matrices("64x128")
    H = np.asarray(H) % 2
    m, n = H.shape
    d = get_ldpc("64x128")["d_min"]
    cols = np.bincount(H.sum(axis=0).astype(int))
    col_desc = ", ".join(f"{w} ({c})" for w, c in enumerate(cols) if c)
    rows = [["Length $n$, dimension $k$, rate", f"{n}, {n - m}, {(n - m) / n:g}"],
            ["Construction", "CCSDS telecommand, quasi-cyclic, $16\\times 16$ circulants"],
            ["Edges", f"{int(H.sum())}"],
            ["Check degree", f"{int(H.sum(axis=1).min())} (all checks)"],
            ["Variable degrees (count)", col_desc],
            ["Girth", f"{girth(H)}"],
            ["$d_{\\min}$, $t_{\\max}$", f"{d} (certified), {(d - 1) // 2}"],
            ["State dimension $3n+m+3$, actions $n+1$", f"{3 * n + m + 3}, {n + 1}"]]
    P.write_table("code", P.table(["Property", "Value"], rows,
                                  "The (128,64) code. $d_{\\min}$ is certified by enumerating, "
                                  "through two disjoint information sets (sets of $k$ positions "
                                  "that determine a codeword), every codeword that could have "
                                  f"weight {d - 1} or less: there is none.", "tbl-code", "ll"), SRC)


def cond_name(cond: str, epoch: str = None) -> str:
    """A condition as the reader sees it. Repository IDs stay out of the paper; the
    release's own docs (docs/ablation/PROTOCOL.md, RESULTS.md) map them."""
    return P.LABEL[cond] + (f", epoch {epoch}" if epoch else "")


def describe(model: dict) -> str:
    t, p = model["type"], model["params"]
    if t == "uniform_count":
        return f"count {p['num_errors']}"
    if t == "uniform_ber":
        return f"BER {p['ber']}"
    if t == "burst":
        return "burst " + "/".join(map(str, p["length"]))
    if t == "cluster":
        return f"cluster {p['num_errors']}"
    if t == "systematic":
        return "systematic"
    return t


def error_models_table():
    """The five error models with every parameter the dataset configuration uses,
    over all six phases, so the table cannot drift from the data."""
    cfg = json.loads((CONFIG_CELL / "main_dataset_config.json").read_text())
    n = 128
    by = defaultdict(list)
    for phase in cfg["curriculum"]:
        for model in phase["error_models"]:
            by[model["type"]].append(model["params"])
    fmt_set = lambda xs: "\\{" + ", ".join(f"{x:g}" for x in sorted(set(xs))) + "\\}"
    counts = [p["num_errors"] for p in by["uniform_count"]]
    bers = [p["ber"] for p in by["uniform_ber"]]
    caps = {p.get("max_errors") for p in by["uniform_ber"]}
    lengths = [x for p in by["burst"] for x in (p["length"] if isinstance(p["length"], list) else [p["length"]])]
    clusters = sorted({(p["num_errors"], p["radius"]) for p in by["cluster"]})
    steps = sorted({p["step"] for p in by["systematic"]})
    sys_weights = sorted({-(-(n - o) // p["step"]) for p in by["systematic"]
                          for o in (p["offset"] if isinstance(p["offset"], list) else [p["offset"]])})
    rows = [["Uniform count", f"$w$ positions chosen uniformly, $w \\in {fmt_set(counts)}$"],
            ["Uniform BER", f"each bit flipped with probability $p \\in {fmt_set(bers)}$, "
                            f"redrawn above weight {max(caps)}"],
            ["Burst", f"$w$ consecutive positions, $w \\in {fmt_set(lengths)}$"],
            ["Cluster", "$w$ positions in a cyclic window of radius $r$ around a random center, "
                        "$(w, r) \\in \\{" + ", ".join(f"({w}, {r})" for w, r in clusters) + "\\}$"],
            ["Systematic", "every $s$-th position from an offset drawn from a listed set, "
                           f"$s \\in {fmt_set(steps)}$, weight {' or '.join(map(str, sys_weights))}"]]
    P.write_table("error-models", P.table(
        ["Model", "Pattern"], rows,
        "Error models, with every parameter used in any phase (@tbl-curriculum gives "
        "the mixture per phase).", "tbl-error-models", "ll"), SRC)


def curriculum_table():
    cfg = json.loads((CONFIG_CELL / "main_dataset_config.json").read_text())
    exp = yaml.safe_load((CONFIG_CELL / "experiment_config.yml").read_text())
    total = sum(p["num_samples"] for p in cfg["curriculum"])
    train_total = round(cfg["total_samples"] * (1 - exp["dataset_creation"]["test_size"]))
    rows = []
    for i, phase in enumerate(cfg["curriculum"], 1):
        mix = defaultdict(float)
        for model in phase["error_models"]:
            mix[describe(model)] += model["weight"]
        # Non-breaking spaces keep each entry ("count 6 (20%)") on one line.
        text = ", ".join(f"{k} ({100 * v:.0f}%)".replace(" ", "\u00a0") for k, v in mix.items())
        rows.append([i, f"{round(train_total * phase['num_samples'] / total):,}", text])
    P.write_table("curriculum", P.table(
        ["Phase", "Training words", "Error mixture (share of the phase)"], rows,
        "Curriculum phases. *count w* is a uniform pattern of weight $w$; burst "
        "lengths and cluster weights are listed; BER patterns are redrawn above "
        "weight 6. The flat ordering keeps the phase sizes and draws every phase "
        "from the overall mixture.", "tbl-curriculum", "crl", cap=80), SRC)


def hyperparameter_table():
    exp = yaml.safe_load((CONFIG_CELL / "experiment_config.yml").read_text())
    t, env = exp["training"], exp["environment"]
    replay = yaml.safe_load((CONFIG_CELL / "expert_experience_config.yml").read_text())["num_experiences"]
    ladder = ", ".join(f"{row[0]:g}" for row in t["epsilon_start"])
    rows = [["Optimizer", f"Adam, learning rate {t['learning_rate']:g}, L2 weight decay {t['weight_decay']:g}"],
            ["Loss, discount", "mean squared TD error, $\\gamma = 0.99$"],
            ["Minibatch, updates per step", f"{t['batch_size']}, {t['update_frequency']}"],
            ["Target network", f"hard copy every {t['target_update_freq']:,} updates"],
            ["Replay buffer", f"{t['replay_buffer_size']:,} transitions, circular, uniform sampling"],
            ["Replay seeding (seeded conditions)", f"{replay:,} oracle transitions, loaded before training"],
            ["Exploration", f"linear in phase episodes to {t['epsilon_min']:g} at "
                            f"{100 * t['epsilon_anneal_fraction']:.0f}% of each phase, then held"],
            ["$\\epsilon$ at phase start", f"0.5 (one epoch); {ladder} (three epochs)"],
            ["Budgets", f"{env['max_steps']} steps, net flip weight {env['max_flips_vector']}, "
                        "no immediate re-flip"],
            ["Network", "640, 480, 360, 270; LayerNorm, leaky ReLU 0.01, dropout "
                        f"{t['network']['dropout_rate']:g}; 906,049 parameters"]]
    P.write_table("hyperparameters", P.table(["Setting", "Value"], rows,
                  "Training configuration, shared by every condition.", "tbl-hyper", "ll"), SRC)


def ablation_tables(res):
    rows = []
    for c in P.CONDITIONS:
        st = P.over_seeds(res, lambda s: P.cell(res, c, s)["success"])
        fer = P.over_seeds(res, lambda s: P.cell(res, c, s)["fer"])
        stp = P.over_seeds(res, lambda s: P.cell(res, c, s)["steps"])
        rows.append([cond_name(c), P.ci(st, 2), P.num(st["sd"], 2),
                     f"{P.num(st['min'], 2)} to {P.num(st['max'], 2)}", P.ci(fer, 2), P.ci(stp, 2)])
    P.write_table("ablation", P.table(
        ["Condition", "Task success (%)", "s", "Range over seeds",
         "FER (%)", "Mean steps"], rows,
        "Held-out results of the $2\\times 2$ ablation, one epoch: mean over five "
        "seeds with the 95% interval, the seed standard deviation *s* and the range.",
        "tbl-ablation", "lrrrrr"), SRC)

    reasons = ["SUCCESS_STOP", "MAX_STEPS", "MAX_FLIPS", "FAILED_STOP"]
    rows = []
    for c in P.CONDITIONS:
        tot = sum(P.cell(res, c, s)["n"] for s in P.seeds(res))
        rows.append([cond_name(c, "1")] + [
            f"{100 * sum(P.cell(res, c, s)['reasons'].get(r, 0) for s in P.seeds(res)) / tot:.2f}"
            for r in reasons])
    for cond in filter(None, (P.three_epoch(res, c) for c in P.CONDITIONS)):
        tot = sum(P.cell(res, cond, s, epoch="3")["n"] for s in P.seeds(res))
        rows.append([cond_name(cond, "3")] + [
            f"{100 * sum(P.cell(res, cond, s, epoch='3')['reasons'].get(r, 0) for s in P.seeds(res)) / tot:.2f}"
            for r in reasons])
    P.write_table("terminal", P.table(
        ["Condition", "Task success", "Step budget", "Flip budget", "Wrong stop"], rows,
        "How held-out episodes of the agent end, in percent, pooled over seeds. A "
        "*wrong stop* is STOP on a word with a nonzero syndrome.",
        "tbl-terminal"), SRC)


def epoch_table(res):
    rows = []
    for cond in filter(None, (P.three_epoch(res, c) for c in P.CONDITIONS)):
        for e in ["1", "2", "3"]:
            g = lambda key: P.over_seeds(res, lambda s: P.cell(res, cond, s, epoch=e)[key])
            succ = g("success")
            if e == "1":
                inc = ""
            else:
                prev = str(int(e) - 1)
                inc = P.ci(P.stats([P.cell(res, cond, s, epoch=e)["success"]
                                    - P.cell(res, cond, s, epoch=prev)["success"]
                                    for s in P.seeds(res)]), 2)
            rows.append([cond_name(cond) if e == "1" else "", e, P.ci(succ, 2), P.num(succ["sd"], 2),
                         inc, P.ci(g("fer"), 2), P.ci(g("steps"), 2)])
    P.write_table("epochs", P.table(
        ["Condition", "Epoch", "Task success (%)", "s", "Gain on previous (points)", "FER (%)",
         "Mean steps"], rows,
        "Held-out results after each epoch, five seeds. Epoch 1 is the one-epoch run "
        "of the same condition; the gain is paired within seed.",
        "tbl-epochs", "lrrrrrr"), SRC)


def main_tables(res):
    """The two result tables of the main text: task success by condition and
    epoch, and the within-seed contrasts at each epoch. A three-epoch condition
    that has not run shows a dash, so the tables regenerate while it runs."""
    S = P.seeds(res)

    def success(cond, e):
        """Per-seed success of a condition after epoch e, or None if not run."""
        if e == "1":
            return [P.cell(res, cond, s)["success"] for s in S]
        cont = P.three_epoch(res, cond)
        return None if cont is None else [P.cell(res, cont, s, epoch=e)["success"] for s in S]

    rows = []
    for c in P.CONDITIONS:
        row = [P.LABEL[c]]
        for e in ["1", "2", "3"]:
            vals = success(c, e)
            row.append("—" if vals is None else P.ci(P.stats(vals), 2))
            if e in ("1", "3"):
                row.append("—" if vals is None else P.num(P.stats(vals)["sd"], 2))
        rows.append(row)
    P.write_table("results", P.table(
        ["Condition", "Epoch 1", "s", "Epoch 2", "Epoch 3", "s"], rows,
        "Held-out task success (%) after each pass over the training data: mean over "
        "five seeds with the 95% interval, and the seed standard deviation *s* in "
        "points. Epoch 1 is the one-epoch ablation; the three-epoch runs repeat it "
        "exactly and continue.", "tbl-results", "lrrrrr"), SRC)

    rows = []
    for aid, held, a, b in CONTRASTS:
        row = [aid, held]
        for e in ["1", "2", "3"]:
            va, vb = success(a, e), success(b, e)
            row.append("—" if va is None or vb is None
                       else P.ci(P.stats([x - y for x, y in zip(va, vb)]), 2))
        rows.append(row)
    P.write_table("contrasts", P.table(
        ["Aid", "Held fixed", "Epoch 1", "Epoch 2", "Epoch 3"], rows,
        "Within-seed differences in held-out task success, in points: the run with the "
        "aid minus the run without it, the other aid held fixed, with the 95% "
        "interval over five seeds; an interval that excludes zero is a difference the "
        "seeds resolve. The seeding contrasts compare runs on the same training and "
        "held-out words; the curriculum contrasts compare two datasets drawn from the "
        "same overall distribution.", "tbl-contrasts", "llrrr"), SRC)

    # The same contrasts seed by seed, so that "on every seed" can be read off.
    rows = []
    for aid, held, a, b in CONTRASTS:
        for e in ["1", "2", "3"]:
            va, vb = success(a, e), success(b, e)
            row = [aid if e == "1" else "", held if e == "1" else "", e]
            row += (["—"] * len(S) if va is None or vb is None
                    else [f"{x - y:+.2f}" for x, y in zip(va, vb)])
            rows.append(row)
    P.write_table("contrasts-seeds", P.table(
        ["Aid", "Held fixed", "Epoch"] + [f"Seed {s}" for s in S], rows,
        "The contrasts of @tbl-contrasts on each seed: the run with the aid minus the "
        "run without it, in points of held-out task success.", "tbl-contrasts-seeds",
        "llr" + "r" * len(S)), SRC)


def performance_tables(res, bench):
    ref = BASE   # baselines scored on the held-out split the headline uses
    cost = bench["summary"]
    rows = []
    decoders = [("DQN agent", "rl", lambda s: P.cell(res, HEAD, s, epoch=EPOCH)),
                ("Greedy bit flipping", "greedy", lambda s: P.cell(res, ref, s, "greedy"))]
    if P.has_baseline(res, "gallager_b"):
        decoders.append(("Gallager-B", "gallager_b", lambda s: P.cell(res, ref, s, "gallager_b")))
    decoders.append(("Hard-input Min-Sum", "min_sum_expert",
                     lambda s: P.cell(res, ref, s, "min_sum_expert")))
    for label, key, get in decoders:
        succ = P.over_seeds(res, lambda s: get(s)["success"])
        recov = P.over_seeds(res, lambda s: 100.0 - get(s)["fer"])
        c = cost[f"{key}@cpu"]
        ops = (f"{c['ops_per_frame'] / 1e6:.2f} M" if c["ops_per_frame"] >= 1e5
               else f"{c['ops_per_frame'] / 1e3:.1f} k")
        rows.append([label, P.ci(succ, 2), P.ci(recov, 2), ops,
                     f"{1e3 * c['latency_median_s']:.2f} / {1e3 * c['latency_p99_s']:.2f}"])
    P.write_table("performance", P.table(
        ["Decoder", "Task success (%)", "Exact recovery (%)", "Operations per frame",
         "CPU latency p50 / p99 (ms)"], rows,
        f"The final policy (flat ordering, unseeded, three epochs) against the baselines "
        f"on the same held-out words. Task success and exact recovery ($1-$FER) are "
        f"means over five seeds of {P.num(105000, 0)} words each, with the 95% interval. "
        f"Cost is measured on a sample of 5,000 of those words per seed, one frame at a "
        f"time on one CPU thread; operations are estimated as described with "
        f"@tbl-cost.", "tbl-performance", "lrrrr"), SRC)


def by_weight_table(res):
    ref = BASE
    rows = []
    s0 = P.seeds(res)[0]
    for w in [str(x) for x in range(7)]:
        count = P.cell(res, ref, s0)["by_weight"][w][1]
        rl = P.over_seeds(res, lambda s: P.cell(res, HEAD, s, epoch=EPOCH)["by_weight"][w][0])
        base = [a for a in ("greedy", "gallager_b", "min_sum_expert") if P.has_baseline(res, a)]
        vals = [P.num(P.over_seeds(res, lambda s: P.cell(res, ref, s, a)["by_weight"][w][0])["mean"])
                for a in base]
        rows.append([w, f"{count:,}", P.ci(rl)] + vals)
    names = {"greedy": "Greedy", "gallager_b": "Gallager-B", "min_sum_expert": "Min-Sum"}
    P.write_table("by-weight", P.table(
        ["Weight", "Words (seed 42)", "DQN agent"] + [names[a] for a in base], rows,
        "Task success (%) of the final policy by error weight. The agent's interval is over seeds; the "
        "baselines are deterministic and vary with the drawn words by less than 0.2 "
        "points.", "tbl-by-weight", whole_headers=True), SRC)


def detail_tables(det):
    if det is None:
        print("DETAILS.json missing: writing placeholders for seen-unseen and contingency")
        for name in ["seen-unseen", "seen-unseen-pooled", "contingency"]:
            P.write_table(name, "*Table pending: run scripts/analyze_ablation_details.py.*\n", SRC)
        return
    cells = [(BASE, "1", "Epoch 1"), (HEAD, EPOCH, f"Epoch {EPOCH}")]
    within = det["seen_unseen_within_model_flat"]
    combos = sorted({tuple(k.split("/")[:2]) for k in within},
                    key=lambda mw: (mw[0], int(mw[1][1:])))
    rows = []
    for model, w in combos:
        row = [MODEL_NAME.get(model, model), w[1:]]
        for cond, e, _ in cells:
            row.append(P.ci(P.stats(within[f"{model}/{w}/{cond}/{e}"]), sign=True))
        for agent in ["greedy", "min_sum_expert"]:
            row.append(P.signed(float(np.mean(within[f"{model}/{w}/{agent}"]))))
        rows.append(row)
    P.write_table("seen-unseen", P.table(
        ["Error model", "Weight"] + [lab for _c, _e, lab in cells] + ["Greedy", "Min-Sum"], rows,
        "Task success on held-out patterns absent from the training split minus "
        "success on patterns present in it, for the final recipe (flat, unseeded) after "
        "one and three epochs, within one error model and weight, in points (mean "
        "within-seed difference and 95% interval). Only strata with at least 20 "
        "patterns of each kind in every seed are shown; burst and systematic patterns "
        "all occur in training by construction and cannot be compared.",
        "tbl-seen-unseen", "lrrrrr"), SRC)

    groups = ["in training split", "only in oracle replay", "not in training split"]
    rows = []
    for w in ["4", "5", "6"]:
        for g in (0, 2):
            row = [w, groups[g]]
            for cond, e, _ in cells:
                vals = [v for v in det["seen_unseen"][f"{cond}/{e}/w{w}/{g}"] if v == v]
                row.append(P.ci(P.stats(vals)) if len(vals) >= 2 else "n/a")
            for agent in ["greedy", "min_sum_expert"]:
                vals = [v for v in det["seen_unseen"][f"{BASE}/1/{agent}/w{w}/{g}"] if v == v]
                row.append(P.num(float(np.mean(vals))) if vals else "n/a")
            rows.append(row)
    P.write_table("seen-unseen-pooled", P.table(
        ["Weight", "Held-out pattern"] + [lab for _c, _e, lab in cells] + ["Greedy", "Min-Sum"], rows,
        "Task success (%) of the final recipe at weights 4 to 6 by whether the held-out "
        "pattern occurs in the training split, pooled over error models; patterns that "
        "occur only in the oracle replay are left out. The groups differ in their mix "
        "of models, which is why greedy, which learns nothing, differs between them "
        "too; @tbl-seen-unseen removes that confound.",
        "tbl-seen-unseen-pooled", "rlrrrr"), SRC)

    rows = []
    for cond, e, label in cells:
        for agent, name in [("min_sum_expert", "Min-Sum"), ("gallager_b", "Gallager-B"),
                            ("greedy", "Greedy")]:
            if f"{cond}/{e}/{agent}" not in det["contingency"]:
                continue
            c = det["contingency"][f"{cond}/{e}/{agent}"]["pooled"]
            rows.append([label, name, f"{c['both']:,}", f"{c['rl_only']:,}",
                         f"{c['baseline_only']:,}", f"{c['neither']:,}"])
    P.write_table("contingency", P.table(
        ["DQN agent", "Baseline", "Both succeed", "Agent only", "Baseline only", "Neither"], rows,
        "Word-by-word outcomes on the same held-out words, pooled over five seeds "
        "(525,000 words).", "tbl-contingency", "llrrrr"), SRC)


def sensitivity_table(res):
    """The principal summaries with and without seed 42, the seed the
    configuration was tuned on. This checks dependence on that seed; it is not
    an independent test of a recipe chosen after the campaign."""
    S = P.seeds(res)
    rest = [s for s in S if s != "42"]

    def series(a, b, e):
        def get(cond, s):
            if e == "1":
                return P.cell(res, cond, s)["success"]
            return P.cell(res, P.three_epoch(res, cond), s, epoch=e)["success"]
        if e != "1" and (P.three_epoch(res, a) is None or (b and P.three_epoch(res, b) is None)):
            return None
        return {s: get(a, s) - (get(b, s) if b else 0.0) for s in S}

    rows = []
    for aid, held, a, b in CONTRASTS:
        for e in ("1", "3"):
            v = series(a, b, e)
            if v is None:
                continue
            rows.append([f"{aid}, {held} held fixed, epoch {e}", P.ci(P.stats([v[s] for s in S]), 2),
                         P.ci(P.stats([v[s] for s in rest]), 2)])
    v = series(HEAD_BASE, None, EPOCH)
    rows.append([f"{P.LABEL[HEAD_BASE]}, epoch {EPOCH} (final policy)",
                 P.ci(P.stats([v[s] for s in S]), 2), P.ci(P.stats([v[s] for s in rest]), 2)])
    P.write_table("sensitivity", P.table(
        ["Quantity", "Seeds 42 to 46", "Seeds 43 to 46"], rows,
        "The principal results with and without seed 42, on whose dataset the "
        "configuration was tuned: contrasts in points and the final policy's task "
        "success in percent, with the 95% interval ($t$ with four and three degrees "
        "of freedom). This measures dependence on the tuning seed; the final recipe was "
        "chosen after the five-seed experiments, so it is not an independent test.",
        "tbl-sensitivity", "lrr"), SRC)


def development_table(chain):
    if chain is None:
        P.write_table("development", "*Table pending: run scripts/summarize_development_chain.py.*\n", SRC)
        return
    # The run IDs stay in the chain record; they name unpublished result folders.
    rows = [[r["change"].replace("epsilon", "$\\epsilon$"), f"{r['success']:.2f}",
             "" if r["gain"] is None else f"{r['gain']:+.2f}", f"{r['mean_steps']:.2f}"]
            for r in chain["steps"]]
    P.write_table("development", P.table(
        ["Step", "Task success (%)", "Gain (points)", "Mean steps"], rows,
        "The last stretch of development: each run changes one setting of the "
        "previous one, with curriculum and replay seeding, one run per step, scored on "
        "its held-out split. These runs used an earlier seed-42 dataset that, unlike the "
        "datasets of this study, still contained some seven-error words, so their numbers "
        "are not comparable with this study's and are not replicated.",
        "tbl-development", "lrrr", cap=48), SRC)


def fallback_table():
    """What recovers the words hard-input Min-Sum fails on at 20 iterations."""
    casc, ctl = P.cascade_record("cascade"), P.cascade_record("controls")
    if casc is None or ctl is None:
        P.write_table("fallback", "*Table pending: run scripts/cascade_minsum_agent.py "
                      "and scripts/cascade_controls.py.*\n", SRC)
        return
    cp, kp = ctl["pooled"], str(casc["k_primary"])
    f1 = cp["min_sum_failures"]
    rows = [[f"Min-Sum, {it} iterations", f"{c['recovered']}", f"{c['wrong_codeword']}"]
            for it, c in cp["min_sum_iterations"].items()]
    rows += [["DQN agent", f"{cp['stage2']['rl']}", "0"],
             [f"DQN agent; on its failures, its first {kp} flips then Min-Sum",
              f"{cp['stage2']['rl'] + casc['pooled']['stage3'][kp]['recovered']}",
              f"{casc['pooled']['stage3'][kp]['wrong_codeword']}"],
             ["Gallager-B", f"{cp['stage2']['gallager_b']}", "0"],
             ["Greedy bit flipping", f"{cp['stage2']['greedy']}", "0"]]
    for src, name in [("agent", f"The agent's first {kp} flips only"),
                      ("greedy", f"Greedy's first {kp} flips"),
                      ("random", f"{kp} random flips")]:
        c = cp["perturb_then_min_sum"]["min_sum_failures"][src][kp]
        rows.append([f"{name}, then Min-Sum", f"{c['recovered']:.0f}", f"{c['wrong_codeword']:g}"])
    P.write_table("fallback", P.table(
        ["Applied to Min-Sum's failures", f"Recovered (of {f1})", "Wrong codeword"], rows,
        f"The {f1} held-out words that hard-input Min-Sum does not recover within 20 "
        "iterations, pooled over the five seeds of the final recipe, and what recovers them. "
        "Every one is a failure to converge. Decoders start from the received word; "
        "random flips are the mean over 20 draws per word.", "tbl-fallback", "lrr"), SRC)


def gallager_selection_table():
    path = P.ROOT / "docs/ablation/GALLAGER_B_SELECTION.json"
    if not path.exists():
        print("GALLAGER_B_SELECTION.json missing: skipping")
        return
    sel = json.loads(path.read_text())
    rows = []
    for r in sorted(sel["rows"], key=lambda r: -r["recovered_20"]):
        t = r["thresholds"]
        tags = [x for x, on in (("majority", r["majority"]), ("Gallager A", r["gallager_a"])) if on]
        name = ", ".join(f"$b_{{{d}}} = {b}$" for d, b in t.items())
        if tags:
            name += f" ({', '.join(tags)})"
        if t == sel["selected"]:
            name = f"**{name}**"
        rows.append([name, f"{r['recovered_20']:.1f}", f"{r['recovered_100']:.1f}",
                     f"{r['mean_iterations_20']:.1f}"])
    P.write_table("gallager-selection", P.table(
        ["Thresholds by variable degree", "Recovered, 20 iterations (%)", "Recovered, 100 iterations (%)",
         "Mean iterations"], rows,
        f"Gallager-B thresholds on {sel['sample']:,} words of the seed-42 training split; "
        "the held-out words play no part. Selected in bold, at 20 iterations.",
        "tbl-gallager-selection", "lrrr"), SRC)


def gallager_ops_per_edge() -> int:
    """Gallager-B's per-edge operation count, read from the benchmark that used it.
    The benchmark record stores only the Min-Sum and network constants."""
    src = (P.ROOT / "scripts/benchmark_decoders.py").read_text()
    return int(re.search(r"^GB_OPS_PER_EDGE = (\d+)", src, re.M).group(1))


def cost_table(bench):
    s = bench["summary"]
    rows = []
    for key, name in [("rl@cpu", "DQN agent"), ("min_sum_expert@cpu", "Hard-input Min-Sum"),
                      ("gallager_b@cpu", "Gallager-B"), ("greedy@cpu", "Greedy bit flipping")]:
        if key not in s:
            continue
        r = s[key]
        rows.append([name, f"{r['success_pct']:.2f}", f"{r['decisions_per_frame']:.2f}",
                     f"{r['parameters']:,}" if r["parameters"] else "0",
                     f"{r['ops_per_frame'] / 1e6:.2f} M" if r["ops_per_frame"] >= 1e5
                     else f"{r['ops_per_frame'] / 1e3:.1f} k",
                     f"{1e3 * r['latency_median_s']:.3f}", f"{1e3 * r['latency_p99_s']:.3f}",
                     f"{r['frames_per_s']:,.0f}"])
    P.write_table("cost", P.table(
        ["Decoder", "Task success (%)", "Decisions per frame", "Parameters", "Operations per frame",
         "Latency p50 (ms)", "Latency p99 (ms)", "Frames/s"], rows,
        f"Per-frame cost on one CPU thread ({P.cpu_name(bench['machine']['cpu'])}), one frame at a "
        f"time, pooled over the five seeds of the final recipe, 5,000 held-out frames each. "
        "Latency p50 and p99 are the median and 99th percentile of the per-frame latency "
        "over the timed frames, and frames/s is the inverse of the mean latency. "
        f"Min-Sum ran {s['min_sum_expert@cpu']['iterations_per_frame']:.2f} iterations per "
        "frame on average. Decisions are environment steps, flips and STOP; Min-Sum and "
        "Gallager-B decode first and then apply their result one flip per step. Task "
        "success is on this sample, not the full held-out set. "
        "Operations count each add, multiply, compare, absolute value and select as "
        "one: for the agent, two per multiply-accumulate of the network plus the "
        "per-unit work of normalization and activation, per decision; for Min-Sum, "
        f"{bench['constants']['ops_per_edge']} per edge per iteration; for Gallager-B, "
        f"{gallager_ops_per_edge()} per edge per iteration plus 2 per bit for the decision; "
        "for greedy bit flipping, 2 per bit per decision. The syndrome and "
        "penalty updates of the environment, which every decoder pays alike, are not "
        "counted.", "tbl-cost"), SRC)


def cost_gpu_table(bench):
    s = bench["summary"]
    gpu = bench["machine"]["gpu_before"]
    rows = []
    for key, name in [("rl@cuda", "DQN agent, GPU"), ("rl@cpu", "DQN agent, one CPU thread")]:
        if key not in s:
            continue
        r = s[key]
        rows.append([name, f"{1e3 * r['latency_median_s']:.3f}", f"{1e3 * r['latency_p99_s']:.3f}",
                     f"{1e3 * r['decision_median_s']:.3f}", f"{r['frames_per_s']:,.0f}"])
    P.write_table("cost-gpu", P.table(
        ["Decoder and device", "Latency p50 (ms)", "Latency p99 (ms)", "Decision p50 (ms)",
         "Frames/s"], rows,
        "The agent's per-frame cost on the GPU against one CPU thread, one frame at a "
        f"time, same runs and frames as @tbl-cost. The GPU ({gpu['name']}) also drives "
        f"the display and was {gpu['utilisation_pct']}% busy before the run, so its "
        "timings are indicative.", "tbl-cost-gpu"), SRC)


def main():
    P.configure("Write the paper's tables.")
    res = P.results()
    bench = P.benchmark()
    det = P.details()
    # Main text (contrasts-seeds and terminal included).
    main_tables(res)
    performance_tables(res, bench)
    code_table()
    error_models_table()
    curriculum_table()
    hyperparameter_table()
    ablation_tables(res)
    # Supplement.
    epoch_table(res)
    by_weight_table(res)
    detail_tables(det)
    sensitivity_table(res)
    development_table(P.development_chain())
    fallback_table()
    gallager_selection_table()
    cost_table(bench)
    cost_gpu_table(bench)


if __name__ == "__main__":
    main()
