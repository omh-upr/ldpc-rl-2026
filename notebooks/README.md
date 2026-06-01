# Notebooks

Companion notebooks for the LDPC-RL codebase. Each notebook is interactive scaffolding around functionality that already lives in `src/` and `scripts/`; nothing here is required for reproducibility, but everything here is grounded in the same code paths used by the command-line entry points.

Open them from the project root (e.g. `jupyter lab notebooks/`). Every notebook bootstraps `sys.path` to the project root on its first cell, so absolute imports like `from src.analysis.analysis import LDPCAnalysis` resolve without setting `PYTHONPATH`.

Notebook outputs are stripped before commit. Re-run cells locally to populate them.

## Index

| Notebook | Purpose | Corresponding script |
|----------|---------|----------------------|
| `complete_analysis.ipynb` | Build or open a per-experiment DuckDB, summarise its configuration, and run any subset of the analyses defined in `config/analysis_definitions.json`. Optional text-file export of tabular results. | `scripts/run_analysis.py`, `scripts/run_experiment_report.py` |
| `sql_tester.ipynb` | Read-only DuckDB console for the same per-experiment databases. Useful for ad-hoc queries before promoting them to entries in `config/analysis_definitions.json`. | — |
| `expert_experiences_analysis.ipynb` | Loads a `.pkl` replay buffer produced by `scripts/run_expert_experience.py`, segments it into episodes, and runs sanity checks (`STOP` termination, no repeated bit-flips, length within `environment.max_steps`, positive final reward). | `scripts/run_expert_experience.py` |
| `analysis_definitions_editor.ipynb` | `ipywidgets` editor for `config/analysis_definitions.json`. Validates entries, writes a timestamped backup under `config/.backups/` before saving, and populates its dropdowns from the file's current vocabulary. | — |
| `epsilon_estimations.ipynb` | Computes per-phase `epsilon_decay` constants (linear or exponential) for `training.epsilon_decay` in `experiment_config.yml`. Inputs are illustrative; edit for your dataset and `update_frequency`. | — |

## Typical workflow

1. **Run an experiment** with `PYTHONPATH=. python scripts/run_experiment.py --config config/experiments_config/<folder>/experiment_config.yml`. This drops `*_training_phase*.json.gz` and `*_evaluation_agent_*.json.gz` files into the configured `results/` directory.
2. **Explore the results** in `complete_analysis.ipynb` (set `EXPERIMENT_ID` to the timestamp from step 1, then run all cells).
3. **Iterate on SQL** in `sql_tester.ipynb` while drafting a new analysis.
4. **Promote it** to `config/analysis_definitions.json` via `analysis_definitions_editor.ipynb` (or by editing the JSON directly).
5. **Validate expert data** in `expert_experiences_analysis.ipynb` before re-seeding the replay buffer with a freshly generated `.pkl`.

## Conventions

- Hard-coded experiment IDs, paths, and dataset sizes belong in the configuration cell at the top of each notebook — never spread across multiple cells.
