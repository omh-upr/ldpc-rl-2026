#!/usr/bin/env python3
"""Write the paper's figures to <out>/figures/ (see paperlib for <out>).

Each figure is written as PDF (print) and PNG at 300 dpi (HTML). Run from the
project root:

    PYTHONPATH=. python scripts/paper/make_figures.py [--out DIR]

Colour does one job throughout: blue marks runs with replay seeding and
orange runs without (slots 1 and 2 of the reference categorical palette, which
are validated as a pair for colour-vision deficiency). Line style carries the
ordering of the data: solid for the curriculum, dashed for the flat ordering.
Epochs of one decoder use that decoder's hue from light to dark (ordinal ramps,
validated with the dataviz validator's --ordinal check). Baselines are grey. Nothing is identified by colour alone:
every series is also named in a legend or a label.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paperlib as P  # noqa: E402

SEEDED, UNSEEDED = "#2a78d6", "#eb6834"
EPOCH_RAMP = {"1": "#86b6ef", "2": "#2a78d6", "3": "#104281"}   # ordinal steps 250/450/650
EPOCH_RAMP_UNSEEDED = {"1": "#f0a07a", "2": "#eb6834", "3": "#9a3a14"}
SEEDED_CONDITIONS = {"OnlyE", "Full", "Full3ep", "OnlyE3ep"}
CURRICULUM_CONDITIONS = {"OnlyC", "Full", "OnlyC3ep", "Full3ep"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BASELINE = "#8a8984"
HEAD, _ = P.HEADLINE

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})
COLUMN = 3.4   # inches, one journal column
DOUBLE = 7.0


def save(fig, name):
    P.FIGURES.mkdir(parents=True, exist_ok=True)
    # No creation date in the PDF and no software stamp in the PNG, so an
    # unchanged figure regenerates byte for byte and shows no diff.
    fig.savefig(P.FIGURES / f"{name}.pdf", dpi=300, metadata={"CreationDate": None})
    fig.savefig(P.FIGURES / f"{name}.png", dpi=300, metadata={"Software": None})
    plt.close(fig)
    print(f"wrote {P.shown(P.FIGURES / name)}.pdf and .png")


def seeds_figure(res):
    S = P.seeds(res)
    fig, ax = plt.subplots(figsize=(COLUMN, 2.4))
    labels = {"NoC_NoE": "Flat", "OnlyC": "Curriculum", "OnlyE": "Flat", "Full": "Curriculum"}
    rng = np.random.default_rng(0)
    for x, c in enumerate(P.CONDITIONS):
        seeded = c in ("OnlyE", "Full")
        colour = SEEDED if seeded else UNSEEDED
        vals = np.array([P.cell(res, c, s)["success"] for s in S])
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(x + jitter, vals, s=22, color=colour, edgecolor="white", linewidth=0.8,
                   zorder=3)
        st = P.stats(list(vals))
        ax.errorbar(x + 0.3, st["mean"], yerr=st["half"], fmt="_", color=INK, ms=9,
                    elinewidth=1.0, capsize=0, zorder=2)
    ax.set_xticks(range(4), [labels[c] for c in P.CONDITIONS])
    ax.set_xlim(-0.5, 3.7)
    ax.set_ylabel("Held-out task success (%)")
    ax.set_ylim(72, 97)
    # Group labels under the tick labels name the seeding factor in text.
    for x0, text, colour in [(0.5, "no seeding", UNSEEDED), (2.5, "replay seeding", SEEDED)]:
        ax.annotate(text, xy=(x0, 0), xycoords=("data", "axes fraction"), xytext=(0, -22),
                    textcoords="offset points", ha="center", va="top", color=INK, fontsize=7.5)
        ax.plot([x0 - 0.9, x0 + 0.9], [-0.155, -0.155], transform=ax.get_xaxis_transform(),
                color=colour, lw=2, clip_on=False, solid_capstyle="round")
    ax.text(3.72, P.stats([P.cell(res, 'Full', s)['success'] for s in S])["mean"],
            " mean ± 95%", fontsize=6.5, color=INK2, va="center")
    save(fig, "seeds")


def epochs_figure(res):
    S = P.seeds(res)
    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 2.6), gridspec_kw={"width_ratios": [1, 1.4]})
    E = ["1", "2", "3"]
    # The inset enlarges epochs 2 and 3, where the means lie within a point.
    zoom = a.inset_axes([0.47, 0.1, 0.5, 0.42])
    for cond in P.epoch_conditions(res):
        colour = SEEDED if cond in SEEDED_CONDITIONS else UNSEEDED
        style = "-" if cond in CURRICULUM_CONDITIONS else "--"
        marker = "o" if cond in CURRICULUM_CONDITIONS else "s"
        # Flat ordering gets hollow markers, so a mean hidden under another
        # shows as a ring around it; error bars are thinner than the lines so
        # they read as uncertainty, not as data.
        face = colour if cond in CURRICULUM_CONDITIONS else "none"
        for s in S:
            a.plot([1, 2, 3], [P.cell(res, cond, s, epoch=e)["success"] for e in E], color=colour,
                   alpha=0.35, lw=0.8, ls=style)
        means = [P.over_seeds(res, lambda s: P.cell(res, cond, s, epoch=e)["success"]) for e in E]
        a.errorbar([1, 2, 3], [m["mean"] for m in means], yerr=[m["half"] for m in means],
                   color=colour, lw=1.2, ls=style, marker=marker, ms=3.5, mfc=face, mew=0.9,
                   elinewidth=0.8, capsize=0, zorder=3, label=P.LABEL[cond])
        zoom.errorbar([2, 3], [m["mean"] for m in means[1:]], yerr=[m["half"] for m in means[1:]],
                      color=colour, lw=1.0, ls=style, marker=marker, ms=3, mfc=face, mew=0.8,
                      elinewidth=0.8, capsize=0)
    a.set_xticks([1, 2, 3])
    a.set_xlim(0.8, 3.2)
    a.set_ylim(73, 99)
    a.set_xlabel("Epoch")
    a.set_ylabel("Held-out task success (%)")
    # Seeded before unseeded, curriculum before flat, in two columns clear of
    # the axis label.
    handles, labels = a.get_legend_handles_labels()
    rank = {P.LABEL[c]: i for i, c in enumerate(["Full", "OnlyE", "OnlyC", "NoC_NoE"])}
    order = sorted(range(len(labels)), key=lambda i: rank[labels[i]])
    a.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center",
             bbox_to_anchor=(0.5, -0.32), ncols=2, fontsize=6.5, columnspacing=1.0)
    zoom.set_xlim(1.8, 3.2)
    zoom.set_ylim(94.9, 97.0)
    zoom.set_xticks([2, 3], ["", ""])
    zoom.set_yticks([95, 96, 97])
    zoom.tick_params(labelsize=6)
    zoom.set_facecolor("white")
    for side in ("top", "right"):
        zoom.spines[side].set_visible(True)

    W = [str(w) for w in range(2, 7)]
    ramp = EPOCH_RAMP if HEAD in SEEDED_CONDITIONS else EPOCH_RAMP_UNSEEDED
    for e in E:
        m = [P.over_seeds(res, lambda s: P.cell(res, HEAD, s, epoch=e)["by_weight"][w][0])["mean"] for w in W]
        b.plot([int(w) for w in W], m, color=ramp[e], lw=1.2, marker="o", ms=3.5,
               label=f"DQN agent, epoch {e}")
    baselines = [("min_sum_expert", "--", "Min-Sum")]
    if P.has_baseline(res, "gallager_b"):
        baselines.append(("gallager_b", "-.", "Gallager-B"))
    baselines.append(("greedy", ":", "Greedy"))
    for agent, style, name in baselines:
        m = [P.over_seeds(res, lambda s: P.cell(res, P.HEADLINE_BASE, s, agent)["by_weight"][w][0])["mean"]
             for w in W]
        b.plot([int(w) for w in W], m, color=BASELINE, lw=1.0, ls=style, label=name)
    b.set_ylim(75, 101)
    b.set_xticks([int(w) for w in W])
    b.set_xlabel("Error weight")
    b.set_ylabel("Held-out task success (%)")
    b.set_title(f"Final recipe ({P.LABEL[HEAD].lower()}) by error weight", loc="left", color=INK)
    b.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout(w_pad=2)
    save(fig, "epochs")


def learning_figure():
    path = P.ROOT / "docs/ablation/learning_curves.csv"
    if not path.exists():
        print("learning_curves.csv missing: skipping the learning figure")
        return
    curves = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["epoch"] == "1":
                curves[(r["condition"], r["seed"])].append(
                    (int(r["bin"]), int(r["episodes"]), float(r["training_success_pct"]),
                     int(r["phase"])))
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.5), sharey=True)
    for ax, (unseeded, seeded, title) in zip(axes, [("NoC_NoE", "OnlyE", "Flat ordering"),
                                                     ("OnlyC", "Full", "Curriculum ordering")]):
        for cond, colour, name in [(unseeded, UNSEEDED, "no seeding"), (seeded, SEEDED, "replay seeding")]:
            first = True
            for (c, s), rows in sorted(curves.items()):
                if c != cond:
                    continue
                rows.sort()
                x = np.cumsum([r[1] for r in rows]) / 1000
                y = [r[2] for r in rows]
                ax.plot(x, y, color=colour, lw=1.0, alpha=0.8, label=name if first else None)
                first = False
        rows = sorted(next(v for (c, _), v in curves.items() if c == seeded))
        x = np.cumsum([r[1] for r in rows]) / 1000
        phases = [r[3] for r in rows]
        for i in range(1, len(rows)):
            if phases[i] != phases[i - 1]:
                ax.axvline(x[i - 1], color=GRID, lw=0.8, zorder=0)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("Training episodes (thousands)")
    axes[0].set_ylabel("Training task success (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=2, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(w_pad=2)
    save(fig, "learning")


# Toy code for the Tanner-graph panel: the cycle code of the complete graph on
# four vertices, six bits on four checks, every bit on two checks and no two bits
# on the same pair, so one error leaves a unique bit with the most unsatisfied
# checks. Illustration only; nothing in it is the (128,64) code.
TOY_CHECKS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]   # checks of each bit
TOY_ERROR = 0


def episode_figure():
    rec = P.example_episode()
    if rec is None:
        print("example_episode.json missing: skipping the episode figure")
        return
    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 2.6), gridspec_kw={"width_ratios": [1.15, 1.6]})

    # (a) Tanner graph of the toy code with one error.
    n, m = len(TOY_CHECKS), 4
    y = [1 if i == TOY_ERROR else 0 for i in range(n)]
    z = [sum(y[i] for i in range(n) if j in TOY_CHECKS[i]) % 2 for j in range(m)]
    p = [sum(z[j] for j in TOY_CHECKS[i]) for i in range(n)]
    vx = np.arange(n) * 1.0
    cx = np.linspace(0.75, 4.25, m)
    for i, checks in enumerate(TOY_CHECKS):
        for j in checks:
            unsat = z[j] == 1
            a.plot([vx[i], cx[j]], [1, 0], color=INK if unsat else GRID, lw=1.1 if unsat else 0.8,
                   zorder=1)
    for i in range(n):
        a.scatter(vx[i], 1, s=150, marker="o", facecolor="white", edgecolor=INK,
                  linewidth=2.2 if i == TOY_ERROR else 0.8, zorder=2)
        a.text(vx[i], 1, str(y[i]), ha="center", va="center", fontsize=7.5, color=INK, zorder=3)
        a.text(vx[i], 1.33, str(p[i]), ha="center", va="center", fontsize=7.5, color=INK2)
    for j in range(m):
        unsat = z[j] == 1
        a.scatter(cx[j], 0, s=140, marker="s", facecolor=INK if unsat else "white", edgecolor=INK,
                  linewidth=0.8, zorder=2)
        a.text(cx[j], 0, str(z[j]), ha="center", va="center", fontsize=7.5,
               color="white" if unsat else INK, zorder=3)
    for yy, label in [(1.33, "unsatisfied\nchecks $p_i$"), (1, "bits $y_i$"), (0, "checks $z_j$")]:
        a.text(-0.55, yy, label, ha="right", va="center", fontsize=7, color=INK2, linespacing=1.0)
    a.set_xlim(-2.6, 5.4)
    a.set_ylim(-0.3, 1.55)
    a.axis("off")
    a.set_title("(a) Tanner graph, one error (bold)", loc="left", color=INK)

    # (b) One held-out episode of the final policy, and greedy on the same word.
    errors = set(rec["error_positions"])
    stop = rec["stop_action"]
    agent = rec["agent"]
    colour = (EPOCH_RAMP if HEAD in SEEDED_CONDITIONS else EPOCH_RAMP_UNSEEDED)["3"]
    g = rec["greedy"]
    b.plot(range(len(g["syndrome_weight"])), g["syndrome_weight"], color=BASELINE, lw=1.2, ls="--",
           marker="o", ms=3, zorder=2, label="Greedy bit flipping")
    # Greedy's last steps alternate between two bits; say so once instead of
    # labelling every step.
    tail = g["actions"][-4:]
    if len(set(tail)) == 2:
        last = len(g["syndrome_weight"]) - 1
        b.annotate(f"greedy alternates\nbits {min(tail)} and {max(tail)}", (last - 2, g["syndrome_weight"][-3]),
                   xytext=(0, 12), textcoords="offset points", ha="center", va="bottom", fontsize=6.5,
                   color=INK2, arrowprops={"arrowstyle": "-", "color": BASELINE, "lw": 0.6})
    sw = agent["syndrome_weight"]
    b.plot(range(len(sw)), sw, color=colour, lw=1.6, zorder=3, label="DQN agent (final policy)")
    b.scatter([0], [sw[0]], s=18, color=INK2, zorder=4)
    # A marker is filled when the flip removes an error from the current word:
    # a flip of a bit outside the pattern adds one, and flipping it back later
    # removes it again, so the state is tracked step by step.
    wrong = set(errors)
    gw = g["syndrome_weight"]
    for t, bit in enumerate(agent["actions"], start=1):
        if bit == stop:
            b.scatter(t, sw[t], s=34, marker="s", color=colour, zorder=4)
            b.annotate("STOP", (t, sw[t]), xytext=(6, 0), textcoords="offset points", ha="left",
                       va="center", fontsize=6.5, color=colour)
            continue
        hit = bit in wrong
        wrong ^= {bit}
        b.scatter(t, sw[t], s=26, marker="o", facecolor=colour if hit else "white", edgecolor=colour,
                  linewidth=1.2, zorder=4)
        # Labels sit clear of the line: below a local minimum, to the right on a
        # descent (to the lower left when greedy's line runs through that spot),
        # above otherwise.
        before, after = sw[t - 1], sw[t + 1] if t + 1 < len(sw) else sw[t]
        if before > sw[t] < after:
            offset, ha, va = (0, -6), "center", "top"
        elif before > sw[t] > after:
            g_here = gw[t] + 0.35 * (gw[t + 1] - gw[t]) if t + 1 < len(gw) else None
            crowded = g_here is not None and sw[t] <= g_here <= sw[t] + 2
            offset, ha, va = ((-5, -3), "right", "top") if crowded else ((5, 3), "left", "bottom")
        else:
            offset, ha, va = (0, 5), "center", "bottom"
        b.annotate(str(bit), (t, sw[t]), xytext=offset, textcoords="offset points", ha=ha, va=va,
                   fontsize=6.5, color=colour)
    budget = len(g["syndrome_weight"]) - 1
    b.axvline(budget, color=GRID, lw=1.0, zorder=0)
    b.text(budget - 0.15, 15.5, "step budget", ha="right", va="top",
           fontsize=6.5, color=INK2)
    b.scatter([], [], s=26, marker="o", facecolor=colour, edgecolor=colour, label="flip removes an error")
    b.scatter([], [], s=26, marker="o", facecolor="white", edgecolor=colour,
              label="flip adds an error")
    b.set_xlim(-0.4, budget + 0.4)
    b.set_ylim(-1.5, 16)
    b.set_yticks(range(0, 17, 4))
    b.set_xticks(range(0, budget + 1, 2))
    b.set_xlabel("Step")
    b.set_ylabel(r"Syndrome weight $\|\mathbf{z}_t\|_1$")
    b.set_title(f"(b) A held-out word with {rec['weight']} errors", loc="left", color=INK)
    b.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout(w_pad=1.5)
    save(fig, "episode")


def main():
    P.configure("Write the paper's figures.")
    res = P.results()
    seeds_figure(res)
    epochs_figure(res)
    learning_figure()
    episode_figure()


if __name__ == "__main__":
    main()
