#!/usr/bin/env python3
"""
Measure what each decoder costs per frame: latency, arithmetic and throughput.

Scores the RL agent, greedy bit flipping and the min-sum expert on held-out
frames of trained ablation cells, on one machine, and writes a JSON record and
a Markdown summary. It answers the manuscript's missing complexity accounting
(review audit item R1.2): the paper reports decoding steps, and a step of the
RL agent is a forward pass of a 906K-parameter network, not a comparable unit.

Usage:
    PYTHONPATH=. python scripts/benchmark_decoders.py
        [--cells Full_s42 Full_s43 ...] [--frames 5000] [--repeats 3]
        [--devices cuda cpu] [--cpu-threads 1] [--no-verify]

What is timed. Each frame runs the loop of EpisodeRunner.run_episode with the
metric collectors removed: reset the agent and the environment, then ask for an
action, apply it, until the episode ends. The same loop drives every decoder,
so the environment's bookkeeping (incremental syndrome and penalty updates, the
reward) is charged to all three alike. The time spent inside select_action is
also recorded on its own, as the decision cost. Min-sum plans the whole frame on
its first call, so its decision time is the message passing.

Three figures per decoder:
    latency       wall clock per frame, one frame at a time, median and 99th
                  percentile over the timed repeats
    arithmetic    operations per frame from the network's layer shapes, the
                  measured decisions per frame and the measured min-sum
                  iterations; an estimate by construction, see OPS_PER_EDGE
    throughput    RL forward passes batched on the recorded states, for the
                  batch sizes asked; excludes the environment, and has no
                  min-sum counterpart because the min-sum decoder here decodes
                  one frame at a time

Checks. An untimed reference pass runs every sampled frame once per decoder
and device. Its outcome (terminal reason and action sequence) must match the
cell's recorded evaluation file frame for frame, which proves the benchmark
scores the decoder the paper reports; every timed repeat must then reproduce
the reference pass. Mismatches are reported, not hidden.

The timings are only meaningful on an otherwise idle machine. The GPU's
utilisation is read before the run and a busy GPU is reported in the output.
"""

import argparse
import ast
import copy
import datetime as dt
import gzip
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from src.core.environment import LDPCEnvironment
from src.core.ldpc_code import LDPC_Code
from src.evaluation.agent_registry import AgentRegistry, AgentSpec
from src.utils.path_utils import resolve_project_path

DEFAULT_CELLS = [f"Full_s{s}" for s in (42, 43, 44, 45, 46)]
BASELINES = ["greedy", "min_sum_expert"]
# Baselines added after the campaign, scored by evaluation-only configs
# (scripts/make_baseline_eval_configs.py). Their parameters and their recorded
# evaluation come from config/baseline_eval/<agent>_<split>_s<seed>/, not from
# the cell, and they are benchmarked whenever that config exists.
ADDED_BASELINES = ["gallager_b"]
ADDED_ROOT = "config/baseline_eval"
AGENT_LABEL = {"rl": "RL (DQN)", "greedy": "Greedy bit flipping",
               "min_sum_expert": "Min-sum (hard input)",
               "gallager_b": "Gallager-B (hard input)"}

# Arithmetic per Tanner-graph edge per min-sum iteration, counting each add,
# multiply, compare, abs and select as one operation, as MinSumDecoder does it:
#   check side     abs 1, sign 1, two-smallest search 2, sign product 1,
#                  own-sign removal 1, other-magnitude select 1,
#                  damping 3 (two multiplies, one add), clip 2        -> 12
#   variable side  incoming sum 1, own-message removal 1, channel add 1,
#                  clip 2                                             ->  5
#   stopping test  hard decision and parity accumulation              ->  2
OPS_PER_EDGE = 19
# Arithmetic per edge per Gallager-B iteration, counted the same way:
#   check side     parity accumulation 1, own-message removal 1           -> 2
#   variable side  compare with the received bit 1, disagreement count 1,
#                  own-message removal 1, threshold compare 1, select 1   -> 5
#   stopping test  parity accumulation of the decision                    -> 1
# plus a majority decision of about 2 operations per variable per iteration.
GB_OPS_PER_EDGE = 8
# Per hidden unit of LDPCOptimizedMLP beyond its matrix product: bias 1,
# LayerNorm about 8 (mean, variance, normalise, scale, shift), LeakyReLU 1.
OPS_PER_HIDDEN_UNIT = 10


# --------------------------------------------------------------------------
# Inputs: cell configuration, held-out frames, recorded outcomes
# --------------------------------------------------------------------------

def load_cell_config(config_root: Path, cell: str) -> dict:
    path = config_root / cell / "experiment_config.yml"
    if not path.exists():
        sys.exit(f"{cell}: no experiment config at {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def load_test_frames(dataset_path: Path) -> tuple:
    """Held-out frames in the order the evaluation iterated them.

    Returns (original, noisy) arrays indexed by position in the test split, so
    row k is episode k+1 of every evaluation file written from this dataset.
    """
    with h5py.File(dataset_path, "r") as h:
        phases = sorted((k for k in h if k.startswith("phase_")),
                        key=lambda k: int(k.split("_")[1]))
        original = np.concatenate([h[f"{p}/original_codewords"][:] for p in phases])
        noisy = np.concatenate([h[f"{p}/noisy_codewords"][:] for p in phases])
        test = h["splits/test_indices"][:]
    return original[test], noisy[test]


def find_model(models_dir: Path) -> Path:
    """The cell's final weights, ignoring any per-epoch checkpoints."""
    found = [p for p in models_dir.glob("ldpc_rl_agent_*.pth") if "_epoch" not in p.name]
    if len(found) != 1:
        sys.exit(f"{models_dir}: expected one final model, found {len(found)}")
    return found[0]


def load_recorded_outcomes(results_dir: Path, agent: str, positions: np.ndarray,
                           noisy: np.ndarray) -> dict:
    """Terminal reason and action sequence of the sampled episodes, as recorded
    by the cell's evaluation. Also confirms each recorded episode was run on the
    frame this benchmark will give the decoder."""
    files = sorted(results_dir.glob(f"*_evaluation_agent_{agent}.json.gz"))
    if len(files) != 1:
        sys.exit(f"{results_dir}: expected one evaluation file for {agent}, found {len(files)}")
    with gzip.open(files[0]) as f:
        episodes = json.load(f)["episodes"]
    if len(episodes) != len(noisy):
        sys.exit(f"{files[0].name}: {len(episodes)} episodes against "
                 f"{len(noisy)} held-out frames")
    # Episode ids run on from one agent's evaluation to the next, so a file's
    # ids are consecutive from its first, not from one.
    first_id = episodes[0]["episode_id"]
    outcomes = {}
    for k in positions.tolist():
        ep = episodes[k]
        if ep["episode_id"] != first_id + k:
            sys.exit(f"{files[0].name}: episode at position {k} has id {ep['episode_id']}")
        recorded_word = ep["noisy_codeword"]
        if isinstance(recorded_word, str):
            recorded_word = json.loads(recorded_word)
        if not np.array_equal(np.asarray(recorded_word), noisy[k]):
            sys.exit(f"{files[0].name}: episode {k + 1} was not run on held-out frame {k}")
        steps = ep["steps"]
        if isinstance(steps, str):
            steps = ast.literal_eval(steps)
        outcomes[k] = (ep["terminal_reason"], [int(s["action"]) for s in steps])
    return outcomes


# --------------------------------------------------------------------------
# Decoding one frame
# --------------------------------------------------------------------------

def decode_frame(agent, env, original: np.ndarray, noisy: np.ndarray) -> tuple:
    """EpisodeRunner.run_episode without the metric collectors.

    Returns (terminal reason, actions, seconds spent inside select_action).
    """
    if hasattr(agent, "reset_with_oracle"):
        agent.reset_with_oracle(original)
    elif hasattr(agent, "reset"):
        agent.reset()
    state = env.reset(noisy, original)
    actions = []
    decision = 0.0
    done = False
    while not done:
        legal_actions = env.get_legal_actions()
        t0 = time.perf_counter()
        action = agent.select_action(state, training=False,
                                     legal_actions=legal_actions,
                                     action_mask=env.get_action_mask())
        decision += time.perf_counter() - t0
        state, _, done, info = env.step(action)
        actions.append(int(action))
    return info["terminal_reason"], actions, decision


class IterationCounter:
    """Counts min-sum check-node half-iterations, one per iteration, by
    wrapping the decoder's update for the untimed reference pass only."""

    def __init__(self, decoder):
        self.decoder = decoder
        self.original = decoder._update_check_to_var
        self.count = 0

    def __enter__(self):
        def counted(*args, **kwargs):
            self.count += 1
            return self.original(*args, **kwargs)
        self.decoder._update_check_to_var = counted
        return self

    def __exit__(self, *exc):
        del self.decoder._update_check_to_var
        return False


def reference_pass(agent, env, originals, noisies, positions, collect_states: bool) -> dict:
    """Run every sampled frame once, untimed, and keep what the timed passes
    are checked against. Doubles as the warm-up."""
    outcomes, states = {}, []
    iterations = None
    reports_iterations = hasattr(agent, "last_iterations")
    counted = 0
    counter = (IterationCounter(agent.decoder)
               if hasattr(agent, "decoder") and not reports_iterations else None)
    if counter:
        counter.__enter__()
    try:
        for k in positions.tolist():
            if collect_states:
                # Each frame's opening state, for the batched throughput
                # measurement; a forward pass costs the same whatever the values.
                states.append(env.reset(noisies[k], originals[k]))
            reason, actions, _ = decode_frame(agent, env, originals[k], noisies[k])
            outcomes[k] = (reason, actions)
            if reports_iterations:
                counted += agent.last_iterations
    finally:
        if counter:
            counter.__exit__()
            iterations = counter.count
    if reports_iterations:
        iterations = counted
    return {"outcomes": outcomes, "states": states, "iterations": iterations}


def timed_pass(agent, env, originals, noisies, positions, sync) -> tuple:
    """Per-frame wall clock and decision time, plus the outcomes for the
    determinism check."""
    frame_times, decision_times, outcomes = [], [], {}
    for k in positions.tolist():
        sync()
        t0 = time.perf_counter()
        reason, actions, decision = decode_frame(agent, env, originals[k], noisies[k])
        sync()
        frame_times.append(time.perf_counter() - t0)
        decision_times.append(decision)
        outcomes[k] = (reason, actions)
    return frame_times, decision_times, outcomes


# --------------------------------------------------------------------------
# Arithmetic
# --------------------------------------------------------------------------

def network_costs(network: torch.nn.Module) -> dict:
    linears = [m for m in network.modules() if isinstance(m, torch.nn.Linear)]
    macs = sum(m.in_features * m.out_features for m in linears)
    hidden_units = sum(m.out_features for m in linears[:-1])
    action_dim = linears[-1].out_features
    # Multiply and add per MAC, the per-unit work of the hidden layers, the
    # output bias, and the masked argmax over the actions.
    ops = 2 * macs + OPS_PER_HIDDEN_UNIT * hidden_units + 3 * action_dim
    return {
        "parameters": sum(p.numel() for p in network.parameters()),
        "macs_per_forward": macs,
        "ops_per_forward": ops,
        "layers": [[m.in_features, m.out_features] for m in linears],
    }


def state_update_ops_per_flip(code: LDPC_Code) -> float:
    """Mean additions the environment makes to the syndrome and penalty vector
    per flip, which every decoder pays equally: each check of the flipped bit
    toggles, and each toggled check moves the penalty of every bit it covers."""
    H = code.H.astype(int)
    row_weight = H.sum(axis=1)
    return float(np.mean([1 + row_weight[np.flatnonzero(H[:, j])].sum()
                          for j in range(code.n)]))


def decision_ops(agent_name: str, net: dict, code: LDPC_Code, decisions_per_frame: float,
                 iterations_per_frame: float) -> float:
    if agent_name == "rl":
        return net["ops_per_forward"] * decisions_per_frame
    if agent_name == "greedy":
        # Penalty sum and argmax over the n bits, per decision.
        return 2 * code.n * decisions_per_frame
    if agent_name == "min_sum_expert":
        edges = int(code.H.sum())
        # Message passing, plus the channel mapping and the final decision.
        return OPS_PER_EDGE * edges * iterations_per_frame + 2 * code.n
    if agent_name == "gallager_b":
        edges = int(code.H.sum())
        # Message passing and the per-iteration decision, plus the initial syndrome.
        return (GB_OPS_PER_EDGE * edges + 2 * code.n) * iterations_per_frame + edges
    raise ValueError(agent_name)


# --------------------------------------------------------------------------
# Batched throughput of the network
# --------------------------------------------------------------------------

def batched_throughput(network: torch.nn.Module, states: list, device: torch.device,
                       batch_sizes: list, min_seconds: float) -> list:
    pool = np.stack(states).astype(np.float32)
    rows = []
    network.eval()
    for batch in batch_sizes:
        reps = int(np.ceil(batch / len(pool)))
        x = torch.as_tensor(np.tile(pool, (reps, 1))[:batch], device=device)
        sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
        with torch.no_grad():
            for _ in range(10):
                network(x).argmax(dim=1)
            sync()
            calls, t0 = 0, time.perf_counter()
            while True:
                network(x).argmax(dim=1)
                calls += 1
                if calls % 10 == 0:
                    sync()
                    if time.perf_counter() - t0 >= min_seconds:
                        break
            elapsed = time.perf_counter() - t0
        rows.append({"batch": batch, "calls": calls,
                     "us_per_call": 1e6 * elapsed / calls,
                     "states_per_s": batch * calls / elapsed})
    return rows


# --------------------------------------------------------------------------
# Machine description
# --------------------------------------------------------------------------

def gpu_utilisation() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        name, util, mem = [s.strip() for s in out.splitlines()[0].split(",")]
        return {"name": name, "utilisation_pct": int(util), "memory_used_mib": int(mem)}
    except Exception as e:  # no GPU, no driver, or no nvidia-smi
        return {"error": str(e)}


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def percentile(values: list, q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def fmt_ms(seconds: float) -> str:
    return f"{1e3 * seconds:.3f}"


def fmt_count(x: float) -> str:
    if x >= 1e6:
        return f"{x / 1e6:.2f}M"
    if x >= 1e3:
        return f"{x / 1e3:.1f}K"
    return f"{x:.0f}"


def table(header: list, rows: list) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def build_document(record: dict) -> str:
    m = record["machine"]
    a = record["arguments"]
    out = [
        "# Decoder Cost Benchmark",
        "",
        f"Generated by `scripts/benchmark_decoders.py` on {record['generated']}. "
        "Regenerate rather than edit.",
        "",
        "## Setup",
        "",
        f"- Cells: {', '.join(a['cells'])}, each scored on {a['frames']} held-out frames "
        f"drawn uniformly from its 105,000 (sample seed {a['sample_seed']}), "
        f"{a['repeats']} timed repeats after an untimed reference pass.",
        f"- CPU: {m['cpu']}; torch CPU threads: {a['cpu_threads']}.",
        f"- GPU: {m['gpu_before'].get('name', 'none')}; utilisation before the run "
        f"{m['gpu_before'].get('utilisation_pct', 'n/a')}%.",
        f"- Python {m['python']}, torch {m['torch']}, numpy {m['numpy']}.",
        "- Latency is wall clock per frame, one frame at a time, through the "
        "evaluation loop without metric collection. Decision time is the part spent "
        "choosing actions; the remainder is the environment's bookkeeping, "
        "which every decoder pays alike.",
        "- Operations are estimates from the network's layer shapes and the measured "
        f"decisions and min-sum iterations per frame ({OPS_PER_EDGE} operations per "
        f"edge per iteration; see the constants at the top of the script). The "
        f"environment adds about {record['state_update_ops_per_flip']:.0f} operations "
        "per flip for every decoder and is not included.",
        "",
    ]

    agg = record["summary"]
    out += ["## Per-frame cost", "",
            "Pooled over all cells and repeats. Success is on the sampled frames.", ""]
    rows = []
    for key, s in agg.items():
        rows.append([AGENT_LABEL[s["agent"]], s["device"],
                     f"{s['success_pct']:.2f}", f"{s['decisions_per_frame']:.2f}",
                     fmt_count(s["parameters"]) if s["parameters"] else "0",
                     fmt_count(s["ops_per_frame"]),
                     fmt_ms(s["latency_median_s"]), fmt_ms(s["latency_p99_s"]),
                     fmt_ms(s["decision_median_s"]),
                     f"{s['frames_per_s']:.0f}"])
    out += [table(["Decoder", "Device", "Success (%)", "Decisions/frame", "Parameters",
                   "Ops/frame", "Latency p50 (ms)", "Latency p99 (ms)",
                   "Decision p50 (ms)", "Frames/s, serial"], rows), ""]

    ms = [s for s in agg.values() if s["agent"] == "min_sum_expert"]
    if ms:
        out += [f"Min-sum ran {ms[0]['iterations_per_frame']:.2f} iterations per frame "
                "on average, against a limit of 20.", ""]
    gb = [s for s in agg.values() if s["agent"] == "gallager_b"]
    if gb:
        out += [f"Gallager-B ran {gb[0]['iterations_per_frame']:.2f} iterations per frame "
                f"on average, against a limit of 20 ({GB_OPS_PER_EDGE} operations per edge "
                "per iteration).", ""]
    rl = [s for s in agg.values() if s["agent"] == "rl"]
    if rl and ms:
        ratio = rl[0]["ops_per_frame"] / ms[0]["ops_per_frame"]
        out += [f"The RL agent spends about {ratio:.0f} times the arithmetic of min-sum "
                "per frame.", ""]

    out += ["## RL latency by cell", "",
            "The five cells share the architecture, so they should time alike.", ""]
    rows = []
    for cell, per in record["cells"].items():
        for key, r in per["runs"].items():
            if r["agent"] == "rl":
                rows.append([cell, r["device"], f"{r['success_pct']:.2f}",
                             f"{r['decisions_per_frame']:.2f}",
                             fmt_ms(r["latency_median_s"]), fmt_ms(r["latency_p99_s"])])
    out += [table(["Cell", "Device", "Success (%)", "Decisions/frame",
                   "Latency p50 (ms)", "Latency p99 (ms)"], rows), ""]

    if record["throughput"]:
        out += ["## Batched RL throughput", "",
                "Forward pass and argmax only, on the opening states of the first "
                "cell's sampled frames; the environment is not included. Frames per "
                "second divides by the measured decisions per frame. No min-sum row: "
                "the decoder here decodes one frame at a time, so a batched figure "
                "would compare an optimised batch path against an unbatched one.", ""]
        rows = []
        for dev, entries in record["throughput"].items():
            for r in entries["rows"]:
                rows.append([dev, r["batch"], f"{r['us_per_call']:.1f}",
                             f"{r['states_per_s']:,.0f}",
                             f"{r['states_per_s'] / entries['decisions_per_frame']:,.0f}"])
        out += [table(["Device", "Batch", "us per call", "States/s", "Frames/s"], rows), ""]

    out += ["## Checks", ""]
    rows = []
    for cell, per in record["cells"].items():
        for key, r in per["runs"].items():
            v = r["verification"]
            rows.append([cell, AGENT_LABEL[r["agent"]], r["device"],
                         "not run" if v is None else f"{v} of {a['frames']}",
                         f"{r['nondeterministic_frames']} of {a['frames'] * a['repeats']}"])
    out += ["Mismatches against the recorded evaluation, and timed frames whose outcome "
            "differed from the reference pass. Both should be zero; the evaluation "
            "ran on the GPU, so a CPU mismatch can be a floating-point tie.", "",
            table(["Cell", "Decoder", "Device", "Differs from evaluation file",
                   "Differs from reference pass"], rows), ""]
    if m["gpu_before"].get("utilisation_pct", 0) > 10:
        out += ["**Warning:** the GPU was busy when the run started; the GPU timings "
                "are not representative.", ""]
    return "\n".join(out)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def summarise_run(agent_name, device, frames, repeats_times, repeats_decisions,
                  reference, net, code) -> dict:
    latencies = [t for rep in repeats_times for t in rep]
    decisions = [t for rep in repeats_decisions for t in rep]
    outcomes = reference["outcomes"]
    steps = [len(a) for _, a in outcomes.values()]
    success = sum(r.startswith("SUCCESS") for r, _ in outcomes.values())
    iterations = (reference["iterations"] / frames) if reference["iterations"] is not None else None
    per_frame_decisions = float(np.mean(steps))
    return {
        "agent": agent_name, "device": device,
        "success_pct": 100.0 * success / frames,
        "decisions_per_frame": per_frame_decisions,
        "iterations_per_frame": iterations,
        "parameters": net["parameters"] if agent_name == "rl" else 0,
        "ops_per_frame": decision_ops(agent_name, net, code, per_frame_decisions,
                                      iterations or 0.0),
        "latency_median_s": statistics.median(latencies),
        "latency_mean_s": float(np.mean(latencies)),
        "latency_p99_s": percentile(latencies, 99),
        "decision_median_s": statistics.median(decisions),
        "decision_mean_s": float(np.mean(decisions)),
        "frames_per_s": 1.0 / float(np.mean(latencies)),
        "repeat_mean_latency_s": [float(np.mean(rep)) for rep in repeats_times],
        "_latencies": latencies, "_decisions": decisions, "_steps": steps,
        "_success": success,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cells", nargs="+", default=DEFAULT_CELLS)
    parser.add_argument("--config-root", default="config/ablation_config")
    parser.add_argument("--frames", type=int, default=5000,
                        help="held-out frames sampled per cell")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--devices", nargs="+", default=["cuda", "cpu"],
                        help="devices for the RL agent; baselines run on the CPU")
    parser.add_argument("--cpu-threads", type=int, default=1,
                        help="torch intra-op threads; 1 compares a single core "
                             "against the single-threaded baselines")
    parser.add_argument("--batch-sizes", type=int, nargs="+",
                        default=[1, 16, 256, 1024, 4096])
    parser.add_argument("--min-seconds", type=float, default=1.0,
                        help="minimum timed duration per batch size")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip the comparison with the recorded evaluation files")
    parser.add_argument("--output-dir", default="docs/ablation/benchmark")
    args = parser.parse_args()

    torch.set_num_threads(args.cpu_threads)
    devices = []
    for d in args.devices:
        if d.startswith("cuda") and not torch.cuda.is_available():
            sys.exit(f"device {d} requested but CUDA is unavailable")
        devices.append(d)

    gpu_before = gpu_utilisation()
    if gpu_before.get("utilisation_pct", 0) > 10:
        print(f"WARNING: GPU at {gpu_before['utilisation_pct']}% before the run; "
              "GPU timings will not be representative.", flush=True)

    config_root = resolve_project_path(args.config_root)
    rng = np.random.default_rng(args.sample_seed)
    record = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "arguments": {**vars(args), "cells": args.cells},
        "machine": {"cpu": cpu_name(), "gpu_before": gpu_before,
                    "python": platform.python_version(), "torch": torch.__version__,
                    "numpy": np.__version__, "platform": platform.platform()},
        "constants": {"ops_per_edge": OPS_PER_EDGE,
                      "ops_per_hidden_unit": OPS_PER_HIDDEN_UNIT},
        "cells": {}, "throughput": {},
    }
    pooled = {}
    net = code = None
    started = time.perf_counter()

    for cell in args.cells:
        config = load_cell_config(config_root, cell)
        code = LDPC_Code.from_default(code_id=config["code_id"])
        env = LDPCEnvironment(code=code, **config["environment"])
        dataset = resolve_project_path(config["dataset_creation"]["output_path"])
        results_dir = resolve_project_path(config["output_paths"]["results_dir"])
        models_dir = resolve_project_path(config["output_paths"]["models_dir"])
        model_path = find_model(models_dir)

        print(f"{cell}: reading {dataset.name} ...", flush=True)
        originals, noisies = load_test_frames(dataset)
        positions = np.sort(rng.choice(len(noisies), size=args.frames, replace=False))

        enabled = [b["name"] for b in config["evaluation"]["baseline_agents"]
                   if b.get("enabled") and b["name"] in BASELINES]
        split = Path(config["dataset_creation"]["output_path"]).stem
        added = {}
        for name in ADDED_BASELINES:
            added_config = resolve_project_path(f"{ADDED_ROOT}/{name}_{split}_s{config['seed']}")
            if (added_config / "experiment_config.yml").exists():
                cfg_added = load_cell_config(added_config.parent, added_config.name)
                if cfg_added["dataset_creation"]["output_path"] != config["dataset_creation"]["output_path"]:
                    sys.exit(f"{added_config}: scores a different dataset than {cell}")
                added[name] = {
                    "results_dir": resolve_project_path(cfg_added["output_paths"]["results_dir"]),
                    "baseline": next(b for b in cfg_added["evaluation"]["baseline_agents"]
                                     if b["name"] == name)}
        runs = [("rl", d) for d in devices] + [(b, "cpu") for b in enabled + list(added)]
        cell_record = {"model": str(model_path.relative_to(resolve_project_path("."))),
                       "dataset": str(dataset.relative_to(resolve_project_path("."))),
                       "runs": {}}

        recorded = {}
        if not args.no_verify:
            for agent_name in sorted({r[0] for r in runs}):
                print(f"{cell}: reading the recorded {agent_name} evaluation ...", flush=True)
                source = added[agent_name]["results_dir"] if agent_name in added else results_dir
                recorded[agent_name] = load_recorded_outcomes(source, agent_name,
                                                              positions, noisies)

        for agent_name, device in runs:
            cfg = copy.deepcopy(config)
            cfg["device"] = device
            registry = AgentRegistry(cfg, env=env, models_dir=str(models_dir))
            if agent_name == "rl":
                spec = AgentSpec(name="rl", display_name="RL Agent", enabled=True,
                                 agent_class=None, requires_model=True,
                                 model_path=str(model_path), config=cfg["training"])
            else:
                baseline = (added[agent_name]["baseline"] if agent_name in added else
                            next(b for b in cfg["evaluation"]["baseline_agents"]
                                 if b["name"] == agent_name))
                spec = AgentSpec(name=agent_name, display_name=AGENT_LABEL[agent_name],
                                 enabled=True, agent_class=registry.agent_classes[agent_name],
                                 config=baseline)
            agent = registry.create_agent(spec)
            if agent_name == "rl":
                net = network_costs(agent.network)
                sync = torch.cuda.synchronize if device.startswith("cuda") else (lambda: None)
            else:
                sync = lambda: None

            print(f"{cell}: {agent_name} on {device}: reference pass ...", flush=True)
            reference = reference_pass(agent, env, originals, noisies, positions,
                                       collect_states=(agent_name == "rl"))
            verification = None
            if agent_name in recorded:
                verification = sum(reference["outcomes"][k] != recorded[agent_name][k]
                                   for k in positions.tolist())

            times, decisions, drift = [], [], 0
            for rep in range(args.repeats):
                t, d, outcomes = timed_pass(agent, env, originals, noisies, positions, sync)
                times.append(t)
                decisions.append(d)
                drift += sum(outcomes[k] != reference["outcomes"][k] for k in positions.tolist())
                print(f"{cell}: {agent_name} on {device}: repeat {rep + 1}, "
                      f"{fmt_ms(float(np.mean(t)))} ms per frame", flush=True)

            summary = summarise_run(agent_name, device, args.frames, times, decisions,
                                    reference, net, code)
            summary["verification"] = verification
            summary["nondeterministic_frames"] = drift
            key = f"{agent_name}@{device}"
            pooled.setdefault(key, []).append(summary)
            cell_record["runs"][key] = {k: v for k, v in summary.items() if not k.startswith("_")}

            if agent_name == "rl" and not record["throughput"].get(device):
                print(f"{cell}: batched throughput on {device} ...", flush=True)
                rows = batched_throughput(agent.network, reference["states"],
                                          agent.device, args.batch_sizes, args.min_seconds)
                record["throughput"][device] = {
                    "cell": cell, "decisions_per_frame": summary["decisions_per_frame"],
                    "rows": rows}
            del agent
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        record["cells"][cell] = cell_record

    # Pool each decoder and device over the cells.
    summary = {}
    for key, runs in pooled.items():
        latencies = [t for r in runs for t in r["_latencies"]]
        decisions = [t for r in runs for t in r["_decisions"]]
        steps = [s for r in runs for s in r["_steps"]]
        frames = len(steps)
        iterations = [r["iterations_per_frame"] for r in runs if r["iterations_per_frame"] is not None]
        agent_name = runs[0]["agent"]
        decisions_per_frame = float(np.mean(steps))
        iterations_per_frame = float(np.mean(iterations)) if iterations else None
        summary[key] = {
            "agent": agent_name, "device": runs[0]["device"],
            "success_pct": 100.0 * sum(r["_success"] for r in runs) / frames,
            "decisions_per_frame": decisions_per_frame,
            "iterations_per_frame": iterations_per_frame,
            "parameters": runs[0]["parameters"],
            "ops_per_frame": decision_ops(agent_name, net, code, decisions_per_frame,
                                          iterations_per_frame or 0.0),
            "latency_median_s": statistics.median(latencies),
            "latency_p99_s": percentile(latencies, 99),
            "decision_median_s": statistics.median(decisions),
            "frames_per_s": 1.0 / float(np.mean(latencies)),
        }
    record["summary"] = summary
    record["network"] = net
    record["state_update_ops_per_flip"] = state_update_ops_per_flip(code)
    record["machine"]["gpu_after"] = gpu_utilisation()
    record["elapsed_s"] = time.perf_counter() - started

    out_dir = resolve_project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"decoder_cost_{stamp}.json"
    md_path = out_dir / f"decoder_cost_{stamp}.md"
    json_path.write_text(json.dumps(record, indent=2))
    md_path.write_text(build_document(record))
    print(f"wrote {json_path}\nwrote {md_path}\n"
          f"elapsed {record['elapsed_s'] / 60:.1f} min", flush=True)

    bad = [(c, k) for c, per in record["cells"].items() for k, r in per["runs"].items()
           if r["verification"] or r["nondeterministic_frames"]]
    if bad:
        print(f"WARNING: outcome mismatches in {bad}; see the Checks table.", flush=True)


if __name__ == "__main__":
    main()
