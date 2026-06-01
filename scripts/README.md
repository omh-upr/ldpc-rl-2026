# LDPC RL Scripts

This directory contains the primary CLI entrypoints for dataset creation, training/evaluation, expert experience generation, analysis, and reporting.

Run all commands from project root with `PYTHONPATH=.`.

## Scripts

1. `run_experiment.py`
2. `run_expert_experience.py`
3. `run_multi_matrix_experiments.py`
4. `run_analysis.py`
5. `run_experiment_report.py`

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
- `config/experiments_config/128x256/experiment_config.yml`

### Notes

- Pipeline switches are controlled by config keys:
  - `run_dataset_creation`
  - `run_training`
  - `run_evaluation`
- Paths are resolved through project path utilities.

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
  - `--all-matrices`: only strict code-id-style folders (currently `24x48`, `56x80`, `128x256` in this repo).
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
- `--database-path`: explicit DB path.
- `--rebuild-database`: force reload from json.gz files.
- `--data-only`: print tabular output only.
- `--to-markdown`: markdown table output (requires `--data-only`).

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
- `128x256` (`d_min=24`)

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

### `Could not find training files for experiment <id>`

- Verify experiment completed and generated files.
- Verify `--results-dir` matches where files were saved.

### `Dataset not found at ...`

- For `run_experiment.py`, enable dataset creation or verify `dataset_creation.output_path` exists.
- For `run_expert_experience.py`, verify the `dataset` path in `expert_experience_config.yml` exists.
- Keep paths project-root-relative in config.

### `No agents configured for evaluation`

- Ensure RL is enabled via training or pretrained model configuration.
- Or enable at least one baseline agent in `evaluation.baseline_agents`.
