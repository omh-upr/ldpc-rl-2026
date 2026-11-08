# LDPC RL Scripts

This directory contains the primary CLI entrypoints for dataset creation, training/evaluation, expert experience generation, analysis, and reporting.

Run all commands from project root with `PYTHONPATH=.`.

## Scripts

1. `run_experiment.py`
2. `run_expert_experience.py`
3. `run_multi_matrix_experiments.py`
4. `run_analysis.py`
5. `run_experiment_report.py`

Auxiliary: `make_ablation_configs.py` writes the cells of the five-seed
ablation under `config/ablation_config/` from the base folder named in the
script and the design in `docs/ablation/PROTOCOL.md`: the four one-epoch
conditions, the same four at three epochs, and the one-seed `Full3epFloor`
control, 41 cells in all. Cells are regenerated,
not edited; a cell whose results folder already holds files is refused unless
`--force` is passed. Run the cells with
`run_multi_matrix_experiments.py --config-folder config/ablation_config --all-folders`.
`aggregate_ablation.py` then reads the results folders and the
per-seed datasets and writes `docs/ablation/RESULTS.md` in full (means,
confidence intervals and within-seed contrasts as specified in the
protocol's section 7); it takes about four minutes and the document is
never edited by hand. It also reports the multi-epoch conditions of the
training-budget study (section 9) once all their cells exist, and writes a
JSON sidecar, `RESULTS.json`, with every per-cell summary behind the tables.
`analyze_ablation_details.py` writes `docs/ablation/DETAILS.md` (with
`DETAILS.json` and `learning_curves.csv`): exact recovery, seen and unseen
error patterns, frame-aligned agreement with the baselines, agreement with the
greedy rule, and the training traces. It caches parsed result files under
`results/ablation_cache/`; the first run parses every training file and takes
about an hour, later runs a few minutes.

Auxiliary: Gallager-B, the hard-input bit-flipping message-passing baseline added
after the campaign, is set up in two steps. `select_gallager_b_thresholds.py`
chooses its thresholds on training words of the seed-42 split and writes
`docs/ablation/GALLAGER_B_SELECTION.md`; `make_baseline_eval_configs.py` writes
ten evaluation-only configs under `config/baseline_eval/` that score it with those
thresholds on each seed's curriculum and flat held-out splits, run one by one
with `run_experiment.py` (about a minute each). `aggregate_ablation.py`,
`analyze_ablation_details.py` and `benchmark_decoders.py` include it once all ten
results exist. `tests/verify_gallager_b.py` checks the vectorized decoder against
a node-by-node reference and the agent against the environment.

Auxiliary: `verify_min_distance.py` certifies a code's minimum distance by
enumerating every low-weight message on two disjoint information sets
(`--code-id 64x128 --max-info-weight 6` certifies $d_{\min} = 14$ in about a
second).

Auxiliary: `benchmark_decoders.py` measures what the RL agent, greedy and
min-sum cost per frame on trained ablation cells (`--cells`; the default is
the five `Full` cells, the paper's record uses `NoC_NoE3ep_s42` to `_s46`, the
only cells whose weights are in the repository): wall-clock latency on GPU and
CPU, estimated operations per frame, and batched network throughput. Every sampled frame's outcome is checked against the cell's
recorded evaluation first. It writes a JSON record and a Markdown summary to
`docs/ablation/benchmark/`. Run it on an idle machine; the defaults take a few
minutes. `--no-verify` skips the comparison with the recorded evaluation, for a
checkout that does not hold the evaluation files.

Auxiliary: `cascade_minsum_agent.py` and `cascade_controls.py` test the trained
agent as a stage after Min-Sum on the final policy's cells (`NoC_NoE3ep`, five
seeds): Min-Sum first, the agent on the words Min-Sum does not recover, then
Min-Sum again after the agent's first flips; the controls replace the agent with
Gallager-B, greedy flips, random flips or more Min-Sum iterations. Both check
their first stages against the recorded evaluations of the agent and the
baselines, so those evaluation files must exist. They write JSON and Markdown
records to `docs/ablation/cascade/`.

Auxiliary: `make_final_eval_configs.py` writes evaluation-only configs under
`config/final_eval/<cell>/` that re-score a cell from its saved final weights,
by default the final policy's five cells, whose weights are in the repository.
Each creates or reuses the cell's dataset, trains nothing, and writes the
agent's and the baselines' evaluation files to `results/ablation/final_eval/<cell>/`.
On a checkout without the original results, run them (and the five
`config/baseline_eval/gallager_b_flat_s<seed>` configs for Gallager-B), then
pass `--config-root config/final_eval` to `benchmark_decoders.py` and the two
cascade scripts.

Auxiliary: `paper/` holds the generators of the paper's tables
(`make_tables.py`), quoted numbers (`make_variables.py`) and figures
(`make_figures.py`). They read only the records under `docs/ablation/`, the
ablation configurations and the code's matrices, never raw training output, so
they run on a fresh checkout; they write to `results/paper/` unless given
`--out DIR`. `record_wall_clock.py` saves each ablation cell's wall-clock time
to `docs/ablation/WALL_CLOCK.json`, since file timestamps do not survive a copy.
`paper/README.md` lists the records and maps the repository's condition IDs to
the names the paper uses.

## Quick Start

```bash
# 1) Run one experiment
PYTHONPATH=. python scripts/run_experiment.py \
  --config config/experiments_config/64x128_Full/experiment_config.yml

# 2) List analyses
PYTHONPATH=. python scripts/run_analysis.py --list

# 3) Run one analysis for one experiment id
PYTHONPATH=. python scripts/run_analysis.py \
  --experiment_id <timestamp> \
  --analysis_id overall_performance_summary

# 4) Generate markdown report
PYTHONPATH=. python scripts/run_experiment_report.py \
  --experiment-id <timestamp>
```

## 1) `run_experiment.py`

Runs a single LDPC RL experiment pipeline based on YAML config.

### CLI

```bash
PYTHONPATH=. python scripts/run_experiment.py --config <path> [--output-parent-dir <dir>]
```

### Arguments

- `--config` (required): Experiment config YAML.
- `--output-parent-dir` (optional): Prepended to `output_paths.results_dir` and `output_paths.models_dir`.

### Typical config

- `config/experiments_config/24x48/experiment_config.yml`
- `config/experiments_config/56x80/experiment_config.yml`
- `config/experiments_config/64x128_Full/experiment_config.yml`

### Notes

- Pipeline switches are controlled by config keys:
  - `run_dataset_creation`
  - `run_training`
  - `run_evaluation`
- Paths are resolved through project path utilities.
- With `run_dataset_creation: true` and `force_recreate: false`, an existing
  dataset is reused only if it matches the curriculum config, the code and the
  requested split; otherwise the run stops and names the mismatch.
- `training.save_epoch_checkpoints: true` writes
  `ldpc_rl_agent_<timestamp>_epoch<N>.pth` at every epoch boundary of a
  multi-epoch run. Evaluation runs once at the end, so this is the only way to
  score epochs separately: point `load_pretrained_model` at one of those files
  with `run_training: false`. Inert for a single epoch.

## 2) `run_expert_experience.py`

Generates expert experiences using BaseAgent-compatible expert agents.

### CLI

```bash
PYTHONPATH=. python scripts/run_expert_experience.py --config <path> [--agent-type <type>]
```

### Arguments

- `--config` (required): Expert generation YAML.
- `--agent-type` (optional): Override `agent_type` from config.
  The output filename is not rewritten automatically; if you override the agent type,
  verify that `output_file` still matches the generated content.

### Typical config

- `config/experiments_config/64x128_Full/expert_experience_config.yml`
- `config/experiments_config/24x48/expert_experience_config.yml`

### Supported agent types

- `omniscient`
- `greedy`
- `min_sum`
- `min_sum_expert`
- `gallager_b`
- `random`

## 3) `run_multi_matrix_experiments.py`

Batch orchestrator for 4-stage multi-folder experiment runs.

### CLI

```bash
PYTHONPATH=. python scripts/run_multi_matrix_experiments.py \
  --config-folder config/experiments_config \
  (--all-matrices | --all-folders | --matrices <ids...> | --folders <names...>) \
  [--dry-run] [--expert <type>] [--output-dir <dir>] \
  [--skip-stages <1|2|3|4 ...>] [--log-dir <dir>]
```

### Arguments

- `--config-folder` (required): Root folder with experiment subfolders.
- One required selection mode:
  - `--all-matrices`: only strict code-id-style folders (currently `24x48` and `56x80` in this repo).
  - `--all-folders`: any valid config folder (includes variants like `64x128_Full` and `64x128_NoC_NoE`).
  - `--matrices <ids...>`: explicit strict matrix folders (for example `24x48 56x80`).
  - `--folders <names...>`: explicit folder names.
- `--dry-run`: validate only.
- `--expert`: override expert type for stage 3.
- `--output-dir`: override output parent dir for stage 4 only.
  It is passed to `run_experiment.py` and prepended to `output_paths.results_dir`
  and `output_paths.models_dir`; it does not relocate stage 1/2 datasets or the
  stage 3 expert experience output file.
- `--skip-stages`: skip selected stages.
- `--log-dir`: log directory (default `experiments/`).

### Stages

1. Main dataset creation
2. Expert dataset creation
3. Expert experience generation
4. Experiment run (training + evaluation)

Dataset paths and the train/test split come from the configurations that own
them: stage 4's `dataset_creation` block and stage 3's `dataset` key. Stages 1
and 2 reuse an existing file only if it matches the config that asks for it;
a mismatch fails the stage rather than training on the wrong data.

## 4) `run_analysis.py`

Runs SQL-backed analyses using DuckDB built from experiment result files.

### CLI

```bash
# List available analyses
PYTHONPATH=. python scripts/run_analysis.py --list

# Run one analysis
PYTHONPATH=. python scripts/run_analysis.py \
  --experiment_id <timestamp> \
  --analysis_id <analysis_id> \
  [--results-dir <dir>] [--database-path <db>] [--rebuild-database] \
  [--data-only] [--to-markdown]
```

### Useful flags

- `--results-dir`: directory used for file discovery and default DB path.
- `--database-path`: explicit DB path. Only ever reused for the experiment it
  already holds; one belonging to another run is refused rather than read or
  overwritten. The default is one database per experiment, which cannot collide.
- `--rebuild-database`: force reload from json.gz files.
- `--data-only`: print tabular output only.
- `--to-markdown`: markdown table output (requires `--data-only`).

Exits nonzero, with the diagnostic on stderr, whenever no analysis ran: a refused
database, an experiment with no result files, a missing definitions file. A query
that runs and returns no rows is a success. `run_experiment_report.py` runs this
command as a subprocess and judges it by return code, so a failure that exited 0
became a section of the report.

## 5) `run_experiment_report.py`

Generates a markdown report by reading placeholders from a template and running only referenced analyses.

### CLI

```bash
PYTHONPATH=. python scripts/run_experiment_report.py \
  --experiment-id <timestamp> \
  [--template <path>] [--no-metadata] [--results-dir <dir>]
```

### Arguments

- `--experiment-id` (required): experiment timestamp id.
- `--template` (optional): template markdown (default `config/report_template.md`).
- `--no-metadata` (optional): omit analysis title/description headers.
- `--results-dir` (optional): directory for report output and DB.

### Output

- Report: `<results-dir>/<experiment-id>_report.md`
- DB: `<results-dir>/<experiment-id>.db` (created/reused)

## Config Layout

Standard folder structure under `config/experiments_config/<folder>/`:

- `experiment_config.yml`
- `main_dataset_config.json`
- `expert_dataset_config.json`
- `expert_experience_config.yml`

### Supported LDPC codes

Experiment configs use `code_id` to select the LDPC matrices:

- `24x48` (`d_min=7`)
- `56x80` (`d_min=12`)
- `64x128` (`d_min=14`)
- `128x256` (`d_min=24`) - matrices only; no experiment configuration ships for this code.

### Network types

Training configs support these `training.network.type` values:

- `SimpleMLP`
- `LDPCOptimizedMLP`

### Expert generation targets

Expert generation configs may specify `num_experiences` or `num_episodes`.
If both are present, `num_experiences` takes priority.

For `omniscient` demonstrations, the agent is reset with the original codeword
before each episode, allowing it to generate oracle-guided randomized teaching
trajectories.

## Common Workflows

### Single experiment + targeted analysis

```bash
PYTHONPATH=. python scripts/run_experiment.py \
  --config config/experiments_config/64x128_Full/experiment_config.yml

PYTHONPATH=. python scripts/run_analysis.py --list

PYTHONPATH=. python scripts/run_analysis.py \
  --experiment_id <timestamp> \
  --analysis_id overall_performance_summary
```

### Evaluation report generation

```bash
PYTHONPATH=. python scripts/run_experiment_report.py \
  --experiment-id <timestamp>
```

### Full batch orchestration

```bash
PYTHONPATH=. python scripts/run_multi_matrix_experiments.py \
  --config-folder config/experiments_config \
  --all-folders
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

Run from project root and set `PYTHONPATH=.`.

### `Could not find any result files for experiment <id>`

- Verify experiment completed and generated files.
- Verify `--results-dir` matches where files were saved.
- Either training or evaluation files are enough. A run with
  `run_training: false` writes only evaluation files, and loads from those
  alone; the analyses that read training episodes then return no rows.

### `Dataset not found at ...`

- For `run_experiment.py`, enable dataset creation or verify `dataset_creation.output_path` exists.
- For `run_expert_experience.py`, verify the `dataset` path in `expert_experience_config.yml` exists.
- Keep paths project-root-relative in config.

### `No agents configured for evaluation`

- Ensure RL is enabled via training or pretrained model configuration.
- Or enable at least one baseline agent in `evaluation.baseline_agents`.
