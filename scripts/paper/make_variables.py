#!/usr/bin/env python3
"""Write <out>/_variables.yml, every number quoted in prose (see paperlib for <out>).

The prose cites these as {{< var key.sub >}}. Values are strings, formatted as
they should read. Run from the project root:

    PYTHONPATH=. python scripts/paper/make_variables.py [--out DIR]
"""

import math
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paperlib as P  # noqa: E402

HEAD, EPOCH = P.HEADLINE


# A seeded curriculum cell carries every setting the paper describes.
CONFIG_CELL = P.ROOT / "config/ablation_config/Full3ep_s42"



def in_words(n: int) -> str:
    """A small count as prose spells it ("six strata"); larger ones stay digits."""
    names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
             "nine", "ten", "eleven", "twelve"]
    return names[n] if 0 <= n < len(names) else f"{n:,}"


def one_digit(x: float) -> str:
    """x to one significant digit, in plain decimals: 0.00419 -> "0.004"."""
    if x == 0:
        return "0"
    d = max(0, -math.floor(math.log10(abs(x))))
    return f"{round(x, d):.{d}f}"


def fixed_facts(bench: dict) -> dict:
    """The code, network, budgets and data sizes, read from the matrix, the
    configuration and the instantiated network rather than typed."""
    import json
    from make_tables import girth
    from src.core.ldpc_utils import get_ldpc, load_matrices
    from src.core.networks import LDPCOptimizedMLP

    H = np.asarray(load_matrices("64x128")[0]) % 2
    m, n = H.shape
    d = get_ldpc("64x128")["d_min"]
    exp = yaml.safe_load((CONFIG_CELL / "experiment_config.yml").read_text())
    ds = json.loads((CONFIG_CELL / "main_dataset_config.json").read_text())
    ee = yaml.safe_load((CONFIG_CELL / "expert_experience_config.yml").read_text())
    ed = json.loads((CONFIG_CELL / "expert_dataset_config.json").read_text())
    total, test_size = ds["total_samples"], exp["dataset_creation"]["test_size"]
    test = round(total * test_size)

    def params(n_, m_):
        net = LDPCOptimizedMLP(state_dim=3 * n_ + m_ + 3, action_dim=n_ + 1)
        return sum(p.numel() for p in net.parameters())
    p0, p_n, p_m, p_code = params(0, 0), params(1, 0), params(0, 1), params(n, m)
    # @eq-params is typed in the Method; fail loudly if the network stops matching it.
    if (p_n - p0, p_m - p0, p0) != (2191, 640, 584_641) or p_code != 2191 * n + 640 * m + 584_641:
        raise SystemExit("the network no longer matches @eq-params in sections/03-method.qmd")
    if p_code != bench["network"]["parameters"]:
        raise SystemExit("the benchmark record counts a different network")
    col = H.sum(axis=0).astype(int)
    return {
        "code": {"n": n, "k": n - m, "m": m, "dmin": d, "tmax": (d - 1) // 2,
                 "edges": int(H.sum()), "girth": girth(H),
                 "check_degree": int(H.sum(axis=1).max()),
                 "var_degrees": " and ".join(str(x) for x in sorted(set(col.tolist()))),
                 "state_dim": 3 * n + m + 3, "actions": n + 1},
        "net": {"params": f"{p_code:,}", "widths": "640, 480, 360, 270",
                "hidden_share": f"{100 * p0 / p_code:.0f}"},
        "env": {"max_steps": exp["environment"]["max_steps"],
                "max_flips": exp["environment"]["max_flips_vector"]},
        "data": {"total": f"{total:,}", "train": f"{total - test:,}", "test": f"{test:,}",
                 "expert_transitions": f"{ee['num_experiences']:,}", "seeds": "42 to 46",
                 "buffer": f"{exp['training']['replay_buffer_size']:,}",
                 "expert_dataset": f"{ed['total_samples']:,}"},
    }


def wall_clock() -> dict:
    """Minutes per cell (docs/ablation/WALL_CLOCK.json) as ranges over seeds,
    seeded and unseeded apart. Cells whose span includes a system suspend are
    left out of the times, not of any result."""
    cells = P.wall_clock()["cells"]
    groups = {"seeded_1ep": ["OnlyE", "Full"], "unseeded_1ep": ["NoC_NoE", "OnlyC"],
              "seeded_3ep": ["Full3ep", "OnlyE3ep"], "unseeded_3ep": ["OnlyC3ep", "NoC_NoE3ep"]}
    out = {}
    for key, conds in groups.items():
        # The record holds finished cells only.
        vals = [cells[cell]["minutes"] for c in conds for s in range(42, 47)
                if (cell := f"{c}_s{s}") in cells and not cells[cell]["suspended"]]
        out[key] = f"{min(vals):.0f} to {max(vals):.0f}"
    return out


def pm(st, d=1):
    return P.ci(st, d)


def main():
    P.configure("Write every number the paper quotes in prose.")
    res, det, bench = P.results(), P.details(), P.benchmark()
    S = P.seeds(res)
    v = {}

    v["archive"] = {"concept_doi": "10.5281/zenodo.20482543"}
    v.update(fixed_facts(bench))
    v["data"]["nseeds"] = len(S)
    v["compute"] = wall_clock()

    # Ablation.
    ab = {}
    for c in P.CONDITIONS:
        st = P.over_seeds(res, lambda s: P.cell(res, c, s)["success"])
        ab[c] = {"success": pm(st), "mean": f"{st['mean']:.1f}", "sd": f"{st['sd']:.2f}",
                 "sd_whole": f"{st['sd']:.0f}",  # for "about 6"
                 "min": f"{st['min']:.1f}", "max": f"{st['max']:.1f}",
                 "range": f"{st['max'] - st['min']:.1f}"}
    unseeded = [P.cell(res, c, s)["success"] for c in ("NoC_NoE", "OnlyC") for s in S]
    seeded = [P.cell(res, c, s)["success"] for c in ("OnlyE", "Full") for s in S]
    ab["unseeded_min"], ab["unseeded_max"] = f"{min(unseeded):.1f}", f"{max(unseeded):.1f}"
    ab["seeded_min"], ab["seeded_max"] = f"{min(seeded):.1f}", f"{max(seeded):.1f}"
    ab["unseeded_span"] = f"{max(unseeded) - min(unseeded):.1f}"
    ab["seeded_span"] = f"{max(seeded) - min(seeded):.1f}"
    for key, a, b in [("seeding_with_curriculum", "Full", "OnlyC"),
                      ("seeding_without_curriculum", "OnlyE", "NoC_NoE"),
                      ("curriculum_with_seeding", "Full", "OnlyE"),
                      ("curriculum_without_seeding", "OnlyC", "NoC_NoE"),
                      ("both", "Full", "NoC_NoE")]:
        ab[key] = pm(P.stats([P.cell(res, a, s)["success"] - P.cell(res, b, s)["success"] for s in S]))
    for c in P.CONDITIONS:
        tot = sum(P.cell(res, c, s)["n"] for s in S)
        ab[c]["max_steps_pct"] = f"{100 * sum(P.cell(res, c, s)['reasons'].get('MAX_STEPS', 0) for s in S) / tot:.1f}"
    v["ablation"] = ab

    # Training budget: every multi-epoch condition, keyed by lower-case name.
    budget = {}
    for cond in P.epoch_conditions(res):
        ep = {}
        for e in ["1", "2", "3"]:
            st = P.over_seeds(res, lambda s: P.cell(res, cond, s, epoch=e)["success"])
            ep[f"e{e}"] = pm(st, 2)
            ep[f"e{e}_mean"] = f"{st['mean']:.1f}"
            ep[f"e{e}_sd"] = f"{st['sd']:.2f}"
            ep[f"e{e}_min"], ep[f"e{e}_max"] = f"{st['min']:.2f}", f"{st['max']:.2f}"
            ep[f"e{e}_steps"] = f"{P.over_seeds(res, lambda s: P.cell(res, cond, s, epoch=e)['steps'])['mean']:.2f}"
        for key, a, b in [("inc12", "2", "1"), ("inc23", "3", "2"), ("total", "3", "1")]:
            ep[key] = pm(P.stats([P.cell(res, cond, s, epoch=a)["success"]
                                  - P.cell(res, cond, s, epoch=b)["success"] for s in S]), 2)
        for w in ["4", "5", "6"]:
            ep[f"gain_w{w}"] = pm(P.stats([P.cell(res, cond, s, epoch="3")["by_weight"][w][0]
                                           - P.cell(res, cond, s, epoch="1")["by_weight"][w][0] for s in S]))
        ep["gain_cluster"] = pm(P.stats([P.cell(res, cond, s, epoch="3")["by_model"]["cluster"][0]
                                         - P.cell(res, cond, s, epoch="1")["by_model"]["cluster"][0] for s in S]))
        for e in ["1", "3"]:
            tot = sum(P.cell(res, cond, s, epoch=e)["n"] for s in S)
            ep[f"e{e}_max_steps_pct"] = f"{100 * sum(P.cell(res, cond, s, epoch=e)['reasons'].get('MAX_STEPS', 0) for s in S) / tot:.2f}"
            ep[f"e{e}_max_flips_pct"] = f"{100 * sum(P.cell(res, cond, s, epoch=e)['reasons'].get('MAX_FLIPS', 0) for s in S) / tot:.2f}"
        budget[cond.lower()] = ep
    # How the headline's held-out episodes end, in percent.
    tot = sum(P.cell(res, HEAD, s, epoch=EPOCH)["n"] for s in S)
    budget["head_endings"] = {
        r.lower(): f"{100 * sum(P.cell(res, HEAD, s, epoch=EPOCH)['reasons'].get(r, 0) for s in S) / tot:.2f}"
        for r in ("MAX_STEPS", "MAX_FLIPS", "FAILED_STOP")}
    # Wrong stops round to 0.00 at two decimals, so the text gives their count
    # and the percentage to one significant digit.
    wrong = sum(P.cell(res, HEAD, s, epoch=EPOCH)["reasons"].get("FAILED_STOP", 0) for s in S)
    budget["head_endings"].update(failed_stop_n=P.num(wrong, 0), episodes=P.num(tot, 0),
                                  failed_stop_1sd=one_digit(100 * wrong / tot))
    # Contrasts at each epoch: seeding and curriculum with the other factor at
    # either level, for the pairs whose cells have all run.
    for key, a, b in [("seeding", "Full3ep", "OnlyC3ep"), ("curriculum", "OnlyC3ep", "NoC_NoE3ep"),
                      ("seeding_flat", "OnlyE3ep", "NoC_NoE3ep"),
                      ("curriculum_seeded", "Full3ep", "OnlyE3ep")]:
        if not {a, b} <= set(P.epoch_conditions(res)):
            continue
        for e in ["1", "2", "3"]:
            diffs = [P.cell(res, a, s, epoch=e)["success"] - P.cell(res, b, s, epoch=e)["success"] for s in S]
            budget[f"{key}_e{e}"] = pm(P.stats(diffs), 2)
            budget[f"{key}_e{e}_seeds_positive"] = sum(d > 0 for d in diffs)
    ctl = res["epoch_controls"].get("Full3epFloor", {})
    if "42" in ctl:
        ref = [P.cell(res, "Full3ep", "42", epoch=e)["success"] for e in ("1", "3")]
        c = [ctl["42"][e]["success"] for e in ("1", "3")]
        budget["floor_share"] = f"{100 * (c[1] - c[0]) / (ref[1] - ref[0]):.0f}"
    v["budget"] = budget

    # Final policy against the baselines.
    perf = {}
    base = P.HEADLINE_BASE   # baselines on the headline's held-out words
    perf["headline"] = f"`{HEAD}`"
    decoders = [("rl", lambda s: P.cell(res, HEAD, s, epoch=EPOCH)),
                ("greedy", lambda s: P.cell(res, base, s, "greedy")),
                ("minsum", lambda s: P.cell(res, base, s, "min_sum_expert"))]
    if P.has_baseline(res, "gallager_b"):
        decoders.append(("gallagerb", lambda s: P.cell(res, base, s, "gallager_b")))
    for key, get in decoders:
        perf[key] = {"success": pm(P.over_seeds(res, lambda s: get(s)["success"]), 2),
                     "success_mean": f"{P.over_seeds(res, lambda s: get(s)['success'])['mean']:.1f}",
                     "fer": pm(P.over_seeds(res, lambda s: get(s)["fer"]), 2),
                     "recovery": pm(P.over_seeds(res, lambda s: 100.0 - get(s)["fer"]), 2),
                     "recovery_mean": f"{P.over_seeds(res, lambda s: 100.0 - get(s)['fer'])['mean']:.1f}"}
        for w in ["5", "6"]:
            perf[key][f"w{w}"] = f"{P.over_seeds(res, lambda s: get(s)['by_weight'][w][0])['mean']:.1f}"
    # Task success and exact recovery of the final policy differ by the episodes
    # that reach the transmitted word without choosing STOP.
    head = [(P.cell(res, HEAD, s, epoch=EPOCH)["success"], 100.0 - P.cell(res, HEAD, s, epoch=EPOCH)["fer"])
            for s in S]
    perf["rl"]["stop_gap"] = f"{np.mean([r - t for t, r in head]):.2f}"
    v["perf"] = perf

    # Sensitivity to the tuning seed: the same summaries over seeds 43 to 46.
    rest = [s for s in S if s != "42"]
    v["sens"] = {
        "headline_rest": pm(P.stats([P.cell(res, HEAD, s, epoch=EPOCH)["success"] for s in rest]), 2),
        "seeding_e1_rest": pm(P.stats([P.cell(res, "Full", s)["success"] - P.cell(res, "OnlyC", s)["success"]
                                       for s in rest])),
        "seeding_e3_rest": pm(P.stats([P.cell(res, "Full3ep", s, epoch="3")["success"]
                                       - P.cell(res, "OnlyC3ep", s, epoch="3")["success"] for s in rest]), 2),
        "curriculum_e3_rest": pm(P.stats([P.cell(res, "OnlyC3ep", s, epoch="3")["success"]
                                          - P.cell(res, "NoC_NoE3ep", s, epoch="3")["success"] for s in rest]), 2),
    }

    # Cost.
    s = bench["summary"]
    rl, ms, gr = s["rl@cpu"], s["min_sum_expert@cpu"], s["greedy@cpu"]
    sel_path = P.ROOT / "docs/ablation/GALLAGER_B_SELECTION.json"
    if sel_path.exists():
        import json
        sel = json.loads(sel_path.read_text())
        chosen = next(r for r in sel["rows"] if r["thresholds"] == sel["selected"])
        majority = next(r for r in sel["rows"] if r["majority"])
        v["gallager"] = {"b3": sel["selected"]["3"], "b5": sel["selected"]["5"],
                         "train_selected": f"{chosen['recovered_20']:.1f}",
                         "train_majority": f"{majority['recovered_20']:.1f}",
                         "train_sample": f"{sel['sample']:,}"}
    v["cost"] = {
        "rl_ops": f"{rl['ops_per_frame'] / 1e6:.1f} million",
        "ms_ops": f"{ms['ops_per_frame'] / 1e3:.1f} thousand",
        "ops_ratio": f"{rl['ops_per_frame'] / ms['ops_per_frame']:.0f}",
        "rl_p50": f"{1e3 * rl['latency_median_s']:.2f}",
        "ms_p50": f"{1e3 * ms['latency_median_s']:.2f}",
        "greedy_p50": f"{1e3 * gr['latency_median_s']:.2f}",
        "latency_ratio": f"{rl['latency_median_s'] / ms['latency_median_s']:.1f}",
        "ms_iterations": f"{ms['iterations_per_frame']:.1f}",
        "cpu": P.cpu_name(bench["machine"]["cpu"]),
    }
    if "gallager_b@cpu" in s:
        gb = s["gallager_b@cpu"]
        v["cost"].update({"gb_ops": f"{gb['ops_per_frame'] / 1e3:.1f} thousand",
                          "gb_p50": f"{1e3 * gb['latency_median_s']:.2f}",
                          "gb_iterations": f"{gb['iterations_per_frame']:.1f}"})

    # Details, when available.
    if det is not None:
        d = {}
        rec = det["exact_recovery"]
        d["success_wrong_word"] = sum(x["success_not_recovered"] for x in rec.values())
        r3 = rec[f"{HEAD}/3/rl"]
        d["recovered_no_stop_pct"] = f"{100 * r3['recovered_without_success'] / r3['episodes']:.2f}"
        c = det["contingency"][f"{HEAD}/3/min_sum_expert"]["pooled"]
        d["rl_only"], d["ms_only"], d["neither"] = c["rl_only"], f"{c['baseline_only']:,}", c["neither"]
        g = det["greedy_agreement"][f"{HEAD}/3"]
        d["max_penalty_pct"] = f"{100 * g['max_penalty'] / g['flips']:.0f}"
        d["greedy_action_pct"] = f"{100 * g['greedy_action'] / g['flips']:.0f}"
        gap = [x for x in det["seen_unseen"][f"{HEAD}/3/gap_w4_6"]]
        d["unseen_gap"] = pm(P.stats(gap))
        # Greedy learns nothing, so its unseen-minus-seen difference at weights
        # 4 to 6 measures only the mix of patterns in the two groups.
        def greedy_mean(w, g):
            vals = [v for v in det["seen_unseen"][f"{P.HEADLINE_BASE}/1/greedy/w{w}/{g}"] if v == v]
            return float(np.mean(vals))
        greedy_gap = [greedy_mean(w, 2) - greedy_mean(w, 0) for w in ("4", "5", "6")]
        d["greedy_unseen_gap_min"] = f"{min(greedy_gap):.0f}"
        d["greedy_unseen_gap_max"] = f"{max(greedy_gap):.0f}"
        used = det["expert_episodes_in_replay"].values()
        d["expert_words_used"] = f"{round(min(used), -3):,.0f}"
        # Within one error model and weight on the headline's split: the largest
        # mean gap in absolute value, over the cells that qualify.
        within = det["seen_unseen_within_model_flat"]
        means = [abs(float(np.mean(v))) for k, v in within.items() if k.endswith(f"/{HEAD}/{EPOCH}")]
        d["within_model_max_abs"] = f"{max(means):.1f}"
        d["within_model_cells"] = in_words(len(means))
        dis = det["disagreements"][f"{HEAD}/{EPOCH}"]
        rl_only, ms_only = dis["RL only"], dis["Min-sum only"]
        d["disagree"] = {
            "rl_only_w6": rl_only["weight"].get("6", 0),
            "rl_only_burst": rl_only["model"].get("burst", 0),
            "rl_only_ms_nonconverged": rl_only["ending"].get("FAILED_STOP", 0),
            "ms_only_w56_pct": f"{100 * (ms_only['weight'].get('5', 0) + ms_only['weight'].get('6', 0)) / ms_only['n']:.0f}",
            "ms_only_max_steps_pct": f"{100 * ms_only['ending'].get('MAX_STEPS', 0) / ms_only['n']:.0f}",
            "persist_kept": dis["rl_only_persisting_from_epoch1"][0],
            "persist_first": dis["rl_only_persisting_from_epoch1"][1],
        }
        if "training" in det:
            # The replay figures describe the seeded three-epoch cells.
            left = [det["training"][f"Full3ep/{s}"]["3"]["expert_transitions_left_at_end"] for s in S]
            trans = [det["training"][f"Full3ep/{s}"]["1"]["transitions"] for s in S]
            d["expert_left_epoch3"] = f"{np.mean(left):,.0f}"
            d["transitions_per_epoch"] = f"{np.mean(trans) / 1e6:.2f} million"
            total = [sum(det["training"][f"Full3ep/{s}"][e]["transitions"] for e in ("1", "2", "3"))
                     for s in S]
            d["transitions_three_epochs"] = f"{np.mean(total) / 1e6:.1f} million"
            # Share of the buffer the 500,000 oracle transitions hold at the end.
            d["expert_share_end_pct"] = f"{100 * 500_000 / (500_000 + np.mean(total)):.0f}"
        v["details"] = d

    # Min-Sum's failures and what recovers them (scripts/cascade_*.py).
    casc, ctl = P.cascade_record("cascade"), P.cascade_record("controls")
    if casc is not None and ctl is not None:
        cp = ctl["pooled"]
        v["fallback"] = {
            "ms_failures": cp["min_sum_failures"],
            "agent": cp["stage2"]["rl"],
            "k": casc["k_primary"],
            "ms_100": cp["min_sum_iterations"]["100"]["recovered"],
            "ms_200": cp["min_sum_iterations"]["200"]["recovered"],
        }

    chain = P.development_chain()
    if chain is not None:
        st = chain["steps"]
        v["development"] = {"start": f"{st[0]['success']:.2f}", "end": f"{st[-1]['success']:.2f}"}
        for i, r in enumerate(st[1:], 1):
            v["development"][f"gain{i}"] = f"{r['gain']:+.2f}"

    # Scaling projections (@tbl-scaling): the (128,64) code against the CCSDS
    # (256,128) code, whose minimum distance 24 is stated in CCSDS 231.1-O-1.
    from math import comb, floor, log10
    t_small, t_large = (14 - 1) // 2, (24 - 1) // 2
    v["scaling"] = {"t_small": t_small, "t_large": t_large,
                    "steps_small": 2 * t_small, "steps_large": 2 * t_large,
                    "orders": floor(log10(comb(256, t_large) / comb(128, t_small)))}

    out = P.OUT / "_variables.yml"
    header = ("# GENERATED by scripts/paper/make_variables.py from docs/ablation/RESULTS.json,\n"
              "# DETAILS.json and the benchmark record. Do not edit; rerun the script.\n\n")
    out.write_text(header + yaml.safe_dump(P.minus(v), sort_keys=False, allow_unicode=True, width=100))
    print(f"wrote {P.shown(out)}")


if __name__ == "__main__":
    main()
