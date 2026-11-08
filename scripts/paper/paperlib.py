"""Shared loaders and formatting for the paper's generator scripts.

The generators (make_tables.py, make_variables.py, make_figures.py) write every
table, figure and number the paper quotes. Everything here reads generated
records, never raw result files:
    docs/ablation/RESULTS.json         scripts/aggregate_ablation.py
    docs/ablation/DETAILS.json         scripts/analyze_ablation_details.py
    docs/ablation/learning_curves.csv  scripts/analyze_ablation_details.py
    docs/ablation/example_episode.json scripts/extract_example_episode.py
    docs/ablation/WALL_CLOCK.json      scripts/record_wall_clock.py
    docs/ablation/DEVELOPMENT_CHAIN.json      scripts/summarize_development_chain.py
    docs/ablation/benchmark/decoder_cost_*.json  scripts/benchmark_decoders.py
    docs/ablation/cascade/cascade_*.json         scripts/cascade_minsum_agent.py
    docs/ablation/cascade/controls_*.json        scripts/cascade_controls.py

Run the generators from the project root with PYTHONPATH=. so that `src` and
`scripts` import:

    PYTHONPATH=. python scripts/paper/make_tables.py [--out DIR]

They write to DIR when given; otherwise into the manuscript's source folder when
it exists (a development checkout), and otherwise to results/paper/, where the
output can be compared with the published paper.
"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from scripts.aggregate_ablation import stats  # one definition of the intervals
from src.utils.path_utils import resolve_project_path

ROOT = resolve_project_path(".")
# The manuscript's source folder: present in development checkouts only.
MANUSCRIPT = ROOT / "docs" / "article" / "manuscript_v2"
OUT = MANUSCRIPT if MANUSCRIPT.is_dir() else ROOT / "results" / "paper"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"


def configure(description: str) -> Path:
    """Parse a generator's command line and set the output folder; returns it."""
    global OUT, TABLES, FIGURES
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--out", type=Path, default=OUT,
                        help=f"output folder (default: {OUT.relative_to(ROOT)})")
    OUT = parser.parse_args().out.resolve()
    TABLES, FIGURES = OUT / "tables", OUT / "figures"
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT


def shown(path: Path) -> str:
    """A path as printed: relative to the project root when inside it."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)

CONDITIONS = ["NoC_NoE", "OnlyC", "OnlyE", "Full"]
# Reader-facing names: the ordering of the data, then replay seeding. The paper
# uses only these; docs/ablation/ and config/ablation_config/ use the IDs.
LABEL = {"NoC_NoE": "Flat, unseeded", "OnlyC": "Curriculum, unseeded",
         "OnlyE": "Flat, seeded", "Full": "Curriculum, seeded"}
SEEDED = {"OnlyE", "Full"}
CURRICULUM = {"OnlyC", "Full"}
# Multi-epoch conditions, each with the one-epoch condition equal to its epoch 1.
EPOCH_CONDITIONS = {"Full3ep": "Full", "OnlyC3ep": "OnlyC", "NoC_NoE3ep": "NoC_NoE",
                    "OnlyE3ep": "OnlyE"}
LABEL.update({c: LABEL[b] for c, b in EPOCH_CONDITIONS.items()})
HEADLINE = ("NoC_NoE3ep", "3")     # decision D1 (revised 2026-09-25): the paper's decoder
# The one-epoch condition whose held-out words, and so whose baselines, the
# headline shares: the flat split.
HEADLINE_BASE = EPOCH_CONDITIONS[HEADLINE[0]]
BENCHMARK_CELLS = [f"{HEADLINE[0]}_s{s}" for s in (42, 43, 44, 45, 46)]


def results() -> dict:
    return json.loads((ROOT / "docs/ablation/RESULTS.json").read_text())


def details() -> dict:
    path = ROOT / "docs/ablation/DETAILS.json"
    return json.loads(path.read_text()) if path.exists() else None


def benchmark() -> dict:
    """The newest benchmark record that scored the headline cells."""
    for path in sorted((ROOT / "docs/ablation/benchmark").glob("decoder_cost_*.json"), reverse=True):
        rec = json.loads(path.read_text())
        if sorted(rec["cells"]) == BENCHMARK_CELLS:
            rec["_path"] = str(path.relative_to(ROOT))
            return rec
    raise SystemExit("no benchmark record for the headline cells")


def cascade_record(kind: str) -> dict:
    """The newest cascade or controls record ("cascade" or "controls") on the
    headline condition over all five seeds, or None."""
    for path in sorted((ROOT / "docs/ablation/cascade").glob(f"{kind}_*.json"), reverse=True):
        rec = json.loads(path.read_text())
        if rec["condition"] == HEADLINE[0] and sorted(rec["seeds"]) == ["42", "43", "44", "45", "46"]:
            rec["_path"] = str(path.relative_to(ROOT))
            return rec
    return None


def development_chain() -> dict:
    path = ROOT / "docs/ablation/DEVELOPMENT_CHAIN.json"
    return json.loads(path.read_text()) if path.exists() else None


def wall_clock() -> dict:
    return json.loads((ROOT / "docs/ablation/WALL_CLOCK.json").read_text())


def example_episode() -> dict:
    path = ROOT / "docs/ablation/example_episode.json"
    return json.loads(path.read_text()) if path.exists() else None


def seeds(res: dict) -> list:
    return [str(s) for s in res["seeds"]]


def cell(res: dict, cond: str, seed: str, agent: str = "rl", epoch: str = None) -> dict:
    """Summary for a one-epoch condition, or for an epoch of a multi-epoch one."""
    if epoch is None:
        return res["conditions"][cond][seed][agent]
    if agent != "rl":
        raise ValueError("multi-epoch cells carry the RL agent only")
    return res["epoch_conditions"][cond][seed][epoch]


def epoch_conditions(res: dict) -> dict:
    """The multi-epoch conditions whose cells have all run, in a fixed order."""
    return {c: b for c, b in EPOCH_CONDITIONS.items() if c in res.get("epoch_conditions", {})}


def three_epoch(res: dict, base: str) -> str:
    """The multi-epoch condition that continues a one-epoch one, or None."""
    return next((c for c, b in epoch_conditions(res).items() if b == base), None)


def has_baseline(res: dict, agent: str) -> bool:
    return agent in res["conditions"]["Full"][seeds(res)[0]]


def over_seeds(res: dict, fn) -> dict:
    return stats([fn(s) for s in seeds(res)])


def signed(x: float, d: int = 1) -> str:
    """A difference with its sign; an exact zero at this precision prints unsigned."""
    return f"{0.0:.{d}f}" if round(x, d) == 0 else f"{x:+.{d}f}"


def ci(st: dict, d: int = 1, sign: bool = False) -> str:
    m = st["mean"]
    mean = signed(m, d) if sign else f"{0.0 if round(m, d) == 0 else m:.{d}f}"
    return f"{mean} ± {st['half']:.{d}f}"


def num(x: float, d: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:,.{d}f}"


_SPANS = re.compile(r"(\$[^$]*\$|`[^`]*`)")          # math and code, left alone
_SIGN = re.compile(r"(?<![\w.\\}])-(?=\d)")            # a hyphen used as a minus sign


def minus(text):
    """Print negative numbers with a true minus sign (U+2212), not a hyphen.

    Python formats -0.58 with an ASCII hyphen, which typesets as a short dash.
    Only a hyphen that starts a number is changed; ranges, compounds, dates and
    anything inside $...$ or `...` are not. Non-strings pass through, so the
    function can be applied to every value of a nested structure."""
    if isinstance(text, dict):
        return {k: minus(v) for k, v in text.items()}
    if isinstance(text, list):
        return [minus(v) for v in text]
    if not isinstance(text, str):
        return text
    parts = _SPANS.split(text)
    return "".join(p if i % 2 else _SIGN.sub("−", p) for i, p in enumerate(parts))


def cpu_name(raw: str) -> str:
    """The processor as a reader names it: "12th Gen Intel(R) Core(TM) i7-12700"
    becomes "Intel Core i7-12700"."""
    name = re.sub(r"\((R|TM)\)", "", raw)
    name = re.sub(r"^\s*\d+(st|nd|rd|th) Gen\s+", "", name)
    return " ".join(name.split())


def table(header: list, rows: list, caption: str, label: str, align: str = None,
          cap: int = 26, whole_headers: bool = False) -> str:
    align = align or "l" + "r" * (len(header) - 1)
    # Pandoc sizes a pipe table's columns by the dashes in its separator line
    # once a line is wider than 72 characters, so every table gets a separator
    # that long, with dashes in proportion to what each column must hold: its
    # longest cell, or its longest header word, since headers wrap at spaces.
    # Text columns are capped (at `cap` characters, 26 unless a table holds one
    # long text column) and wrap; numeric columns are not, so a number never
    # breaks. `whole_headers` sizes each column for its full header instead, for
    # a table narrow enough that no header needs to wrap. The PDF sets tables in
    # \\small with narrow padding (_quarto.yml).
    def weight(i):
        cells = [len(str(r[i])) for r in rows]
        word = (len(str(header[i])) if whole_headers
                else max(len(w) for w in str(header[i]).split()))
        need = max(cells + [word, 3])
        return (min(need, cap) if align[i] in "lc" else need) + 2
    weights = [weight(i) for i in range(len(header))]
    scale = max(1.0, 100 / sum(weights))
    marks = {"l": ":{}", "r": "{}:", "c": ":{}:"}
    sep = [marks[a].format("-" * max(1, round(w * scale) - marks[a].count(":")))
           for w, a in zip(weights, align)]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(sep) + "|"]
    lines += ["| " + " | ".join(minus(str(c)) for c in r) + " |" for r in rows]
    return "\n".join(lines) + f"\n\n: {minus(caption)} {{#{label}}}\n"


def write_table(name: str, text: str, source: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    header = f"<!-- GENERATED by {source}; do not edit. -->\n\n"
    (TABLES / f"{name}.md").write_text(header + text)
    print(f"wrote {shown(TABLES / f'{name}.md')}")
