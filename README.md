# Reinforcement Learning for LDPC Decoding

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20482543.svg)](https://doi.org/10.5281/zenodo.20482543)

## 1. Project Overview

This project implements decoding of Low-Density Parity-Check (LDPC) codes using Deep Reinforcement Learning. A DQN agent learns a bit-flipping policy that sequentially corrects errors in noisy codewords under a hard-decision observation model. The decoding problem is formulated as a Markov Decision Process where the agent chooses one bit to flip (or to stop) at each step, guided by the current syndrome and flip history.

The system features curriculum learning across six progressive difficulty phases, expert replay seeding from an omniscient oracle, and a five-layer neural architecture optimized for LDPC structure.

For the formal MDP specification (state, action, reward, success definitions), see [docs/MASTER_DESIGN.md](docs/MASTER_DESIGN.md). For the full experiment workflow and flagship configuration, see [docs/MASTER_EXPERIMENT_DESIGN.md](docs/MASTER_EXPERIMENT_DESIGN.md).

## 2. System Architecture

### 2.1. Package Structure

```
src/
├── core/                     # Foundation components
│   ├── ldpc_code.py         # Central LDPC_Code class, single source of truth for code properties
│   ├── environment.py       # RL environment (MDP formulation) with dynamic state/action spaces
│   ├── base_agent.py        # Abstract base class for all agents
│   ├── learning_agent.py    # DQN-based learning agent
│   ├── baseline_agents.py   # Classical decoders and experts (Min-Sum, Gallager-B, Greedy, Random, Omniscient)
│   ├── networks.py          # Neural network architectures (SimpleMLP, LDPCOptimizedMLP)
│   ├── network_factory.py   # Factory for creating networks from configuration
│   ├── reward.py            # Reward functions
│   └── ldpc_utils.py        # Utilities for loading matrices and code metadata
├── training/                # Training pipeline
│   ├── epsilon_schedule.py  # Per-phase and per-epoch epsilon schedules, including auto_episodes
│   ├── replay_buffer.py     # Experience replay with persistence
│   └── training.py          # Curriculum learning and DQN training
├── evaluation/              # Evaluation framework
│   ├── evaluation.py        # Performance metrics and comparison
│   ├── episode_runner.py    # Episode execution engine
│   ├── episode_context.py   # Context management for per-agent evaluation
│   ├── metrics_callbacks.py # Modular metrics collection
│   ├── metrics_manager.py   # Results saving and reporting
│   ├── progressive_saver.py # Progressive saving for memory management
│   └── agent_registry.py    # Centralized agent instantiation and model loading
├── experiment/              # Main orchestration
│   └── experiment.py        # LDPCExperiment pipeline controller
├── data/                    # Data generation and management
│   ├── robust_data_generator.py # Curriculum-based dataset creation
│   ├── create_dataset.py    # Dataset creation utilities
│   ├── simple_agent_registry.py # Lightweight agent registry for expert generation
│   └── experience_generator.py # Core expert experience collection logic
├── advanced/                # Advanced algorithms
│   ├── min_sum_core.py      # Core implementation of the Min-Sum algorithm
│   └── gallager_b.py        # Gallager-B hard-decision message passing
├── analysis/                # Analysis and visualization
│   ├── analysis.py          # DuckDB SQL-based results analysis
│   └── experiment_loader.py # One loader shared by the CLI and the notebook
└── utils/                   # General-purpose utilities
    ├── path_utils.py        # Project-wide path resolution
    ├── file_patterns.py     # Consistent file naming and discovery
    └── gf2_operations.py    # GF(2) operations for LDPC codes
```

### 2.2. Configuration-Driven Design

The system uses YAML and JSON configuration files for complete experiment control:

- **`config/experiments_config/<folder>/experiment_config.yml`**: Main experiment configuration.
- **`config/experiments_config/<folder>/main_dataset_config.json`**: Curriculum and dataset generation settings.
- **`config/experiments_config/<folder>/expert_dataset_config.json`**: Expert dataset generation settings.
- **`config/experiments_config/<folder>/expert_experience_config.yml`**: Expert experience generation settings.
- **`config/analysis_definitions.json`**: SQL queries and metadata for the analysis framework.

#### A note on `update_frequency`

`update_frequency` (under `training:` in `experiment_config.yml`) sets how many environment steps elapse between gradient updates: `1` performs one update per step, `4` one update every four steps. Raising it reduces wall-clock time roughly in proportion to the updates skipped, at the cost of **sample efficiency** — the agent learns less per environment step. The effect on *final* decoding performance is not guaranteed in either direction, since the update-to-data ratio is itself a hyperparameter and very high ratios can destabilise Q-learning.

**When comparing configurations, keep `update_frequency` identical across all of them.** Every experiment config uses the code default of `1`; only the two smoke tests, `64x128_QuickStart` and `64x128_QuickStart_3ep`, set `4`, deliberately, because there speed matters and convergence does not.

#### Reproducibility

Each `experiment_config.yml` and `expert_experience_config.yml` carries a `seed`, which is applied to Python's `random`, NumPy, and PyTorch (including CUDA) before any component is constructed. Given the same seed and configuration, dataset generation, expert experience generation, and training are reproducible.

Reproduction is bit for bit on the same GPU model with the locked package versions: a rerun of an ablation cell reproduced its network weights to the last bit and its evaluation episode by episode (`docs/ablation/PROTOCOL.md`). On other hardware, floating-point differences can shift the results slightly.

Reproducibility holds **within** a release series. Results are not comparable bit-for-bit **across** series:

- **1.0.x to 1.1.x.** v1.1.0 changed the replay buffer's index sampler to a constant-time algorithm, which draws a different random stream. The sampling distribution is unchanged (uniform, without replacement); only the draw order differs.
- **1.1.x to 1.2.x.** v1.2.0 keeps every `64x128` error pattern within the correction radius (section 4.3), so datasets generated from earlier configurations no longer match and are refused on reuse. Result files now carry the epoch in their names, and earlier results are not discovered.

Regenerate results rather than mixing series; see `CHANGELOG.md`.

Experiments that request `device: "cuda"` fail fast if CUDA is unavailable, rather than silently completing on CPU with results that look identical but take several times longer. Set `strict_device: false` at the top level of an experiment config to allow the fallback with a warning instead.

### 2.3. Path Management

All file paths are managed through a centralized `path_utils` system that detects the project root automatically and resolves paths consistently regardless of working directory.

### 2.4. Regression Checks

Seven standalone scripts under `tests/` each build a small synthetic fixture, exercise one subsystem and exit nonzero on the first failed check: training updates and target synchronisation, the progressive saver, action masking, Min-Sum equivalence against the reference decoder, the Gallager-B decoder against a node-by-node reference, analysis loading, and dataset validation. They run on CPU in a few minutes:

```bash
for t in tests/verify_*.py; do PYTHONPATH=. python "$t" || break; done
```

## 3. The Reinforcement Learning Problem

### 3.1. State Representation

The agent observes a flat state vector whose dimensions adapt to the selected LDPC code:

- **Codeword** (`n` dims): Current bit estimate.
- **Syndrome** (`m` dims): Parity check violations.
- **Flip History** (`n` dims): Which positions have been flipped.
- **Penalties** (`n` dims): Number of unsatisfied checks per bit (integer, not mod 2).
- **Metadata** (3 dims): Syndrome weight, penalty sum, step count.

Total state dimension: `3n + m + 3`. For example:
- **24x48**: 171 dimensions, 49 actions.
- **56x80**: 299 dimensions, 81 actions.
- **64x128** (flagship): 451 dimensions, 129 actions.

### 3.2. Action Space

The discrete action space has `n + 1` actions: `n` bit-flip actions (one per codeword position) plus an explicit STOP action. The agent must learn both which bits to flip and when to terminate.

The environment setting `forbid_immediate_reflip` optionally removes the bit flipped on the previous step from the legal set, since flipping it back restores the codeword exactly and can never change the syndrome. It applies to every agent, learned and classical alike, and it changes the MDP, so it is off by default and runs that enable it are compared only with each other.

### 3.3. Reward Function

The `ImprovedRewardFunction` is intentionally sparse, avoiding dense per-action shaping:

- **Terminal success bonus**: Large reward for achieving zero syndrome at episode end.
- **STOP training**: Bonus for correct termination, penalty for premature stopping.
- **Step penalty**: Small per-step cost to encourage efficiency.
- **Exponential flip penalty**: Escalates sharply when flips exceed the code's correction capability.

See [docs/MASTER_DESIGN.md](docs/MASTER_DESIGN.md) Section 3.6 for the full reward specification with formulas and a worked example.

## 4. Key Technical Features

### 4.1. Centralized Code Management

The `LDPC_Code` class encapsulates all properties of a given code (H and G matrices, dimensions, correction capability). Changing the `code_id` parameter automatically adjusts state/action dimensions, network layers, and dataset generation.

### 4.2. Network Architectures

Two architectures are available via the `network_factory`:
- **`SimpleMLP`**: Standard 2-layer MLP baseline.
- **`LDPCOptimizedMLP`**: 5-layer MLP (640-480-360-270) with LayerNorm, LeakyReLU, and minimal dropout. Approximately 906K parameters for 64x128. This is the flagship architecture.

### 4.3. Curriculum Learning

The `RobustDataGenerator` creates static HDF5 datasets with a six-phase curriculum progressing from single-error patterns to complex multi-error scenarios. Error models include uniform count, uniform BER, burst, cluster, and systematic patterns. All generation is seed-controlled for full reproducibility. Every `64x128` dataset config keeps each error pattern within the code's correction radius of six flips: the systematic patterns are chosen so no offset places a seventh error, and the BER models carry `max_errors: 6`, which redraws the rare heavier pattern.

Training can repeat the curriculum for several epochs. Each epoch restarts at phase 1, and the epsilon fields accept either one value per phase, reused by every epoch, or one such row per epoch. `epsilon_decay: "auto_episodes"` derives the schedule from an intent instead of per-update decrements: reach `epsilon_min` after `epsilon_anneal_fraction` of each phase, counted in episodes. With `save_epoch_checkpoints: true` the weights at each epoch boundary are kept, so the epochs can be scored separately afterwards.

### 4.4. Expert-Guided Learning

The replay buffer can be warm-started with pregenerated expert demonstrations. Available expert agents:

- **OmniscientAgent**: Oracle-based expert with ground truth access (primary expert in the flagship experiment).
- **MinSumExpertAgent**: Episode-level planning using Min-Sum algorithm convergence.
- **GreedyAgent**: Step-by-step penalty-based heuristic selection.
- **RandomAgent**: Baseline random action generation.

Expert experiences are generated offline via `scripts/run_expert_experience.py` and stored as pickle files for reuse across training runs.

### 4.5. Evaluation and Analysis

- **Progressive saving**: Results are saved per `(epoch, phase)` training segment and per evaluation agent to manage memory.
- **Compressed storage**: All results are GZIP-compressed JSON files (70-80% size reduction).
- **Database-backed analysis**: Results are loaded into a DuckDB database for SQL-based querying.
- **Automated reports**: Template-driven Markdown report generation from analysis results.
- **Guarded reuse**: an existing dataset is reused only when it matches the configuration asking for it, and an analysis database only for the run it holds. The analysis and report commands exit nonzero whenever an analysis did not run, so a failure cannot pass as a result.

## 5. LDPC Code Specifications

The system supports multiple LDPC codes, selectable via `code_id`:

```yaml
code_id: "64x128"  # Options: "24x48", "56x80", "64x128", "128x256"
```

| Code | Dimensions (m x n) | d_min | Rate | Correction Capability |
|------|-------------------|-------|------|-----------------------|
| `24x48` | 24 x 48 | 7 | 0.50 | 3 |
| `56x80` | 56 x 80 | 12 | 0.30 | 5 |
| `64x128` | 64 x 128 | 14 | 0.50 | 6 |
| `128x256` | 128 x 256 | 24 | 0.50 | 11 |

All LDPC matrices (H and G) are stored as NumPy arrays in the `data/` directory. The flagship experiment and the ablation use `64x128`, the rate-1/2 (128,64) code of the CCSDS telecommand standard; `scripts/verify_min_distance.py` certifies its minimum distance of 14. `128x256` has matrices and a registry entry but no experiment configuration yet.

## 6. How to Run

### 6.1. Environment Setup

```bash
# Create and activate virtual environment
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv sync
```

`uv sync` installs the exact, pinned versions from the committed `uv.lock`, so the
environment is reproducible across machines. Use `uv sync --frozen` to fail rather than
re-resolve if `uv.lock` and `pyproject.toml` ever drift apart.

The committed `.python-version` pins CPython 3.12; `uv` downloads a matching interpreter
automatically if the host does not have one. The pinned environment installs a CUDA-enabled
PyTorch build — verify the GPU is visible before a long run:

```bash
PYTHONPATH=. python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 6.2. Smoke Test (Quick Start)

Before launching the flagship run, verify the toolchain end-to-end with the `64x128_QuickStart` config — 8 000 training samples across three curriculum phases, 5 000 expert experiences, on the order of a few minutes on a single GPU. Produces the same artefact layout as the full run (per-phase training files, an intra-phase flush file, per-agent evaluation files, a model checkpoint, and a per-experiment DuckDB once you build it):

```bash
PYTHONPATH=. python scripts/run_multi_matrix_experiments.py \
  --config-folder config/experiments_config --folders 64x128_QuickStart
```

Note: this overwrites `data/ldpc_64x128_{main,expert}_dataset.h5` if they exist (regenerated on demand by `force_recreate: true`). The slow-to-produce 500 000-experience expert pickle used by `64x128_Full` is kept distinct (`data/quickstart_64x128_omniscient_5000.pkl`) and is not touched.

`64x128_QuickStart_3ep` is the same smoke test run for three epochs with per-epoch checkpoints, and it writes its own dataset instead of overwriting the shared one.

### 6.3. Reproducing the Main Experiment

The full end-to-end workflow for the flagship `64x128_Full` experiment. It trains over 245,000 episodes across six curriculum phases and takes roughly **1.5 hours** on a single RTX 3060 Ti (8 GB), including dataset generation and evaluation against both baselines. Peak host memory is about 8.5 GB, most of it the replay buffer.

Four further configurations form a comparison chain on the same seed-42 dataset and the same expert replay, each differing from the previous one by a single setting: `64x128_Full_AutoEp` puts `64x128_Full` on a derived `auto_episodes` epsilon schedule, `64x128_Full_AutoEp_Uniform` gives every phase the same start and floor instead of a taper, `64x128_Full_Uniform_NoReflip` adds `forbid_immediate_reflip`, and `64x128_Full_Uniform_3ep` adds three epochs with a halving epsilon ladder. That chain, run on one seed, led to the ablation's training configuration (section 6.6), which supersedes it as the reference for results. See [docs/MASTER_EXPERIMENT_DESIGN.md](docs/MASTER_EXPERIMENT_DESIGN.md) section 5.6.

```bash
# Step 1: Run the complete experiment (dataset creation, training, evaluation)
PYTHONPATH=. python scripts/run_multi_matrix_experiments.py \
  --config-folder config/experiments_config --folders 64x128_Full

# Step 2: Build the metrics database and run an initial analysis
#         (replace <experiment_id> with the timestamp from Step 1, e.g., 20260316_160734)
PYTHONPATH=. python scripts/run_analysis.py \
  --experiment_id <experiment_id> \
  --analysis_id overall_performance_summary \
  --rebuild-database

# Step 3: Generate the comprehensive report
PYTHONPATH=. python scripts/run_experiment_report.py \
  --experiment-id <experiment_id>
```

### 6.4. Running Individual Experiments

```bash
# Run a single experiment with a specific configuration
PYTHONPATH=. python scripts/run_experiment.py \
  --config config/experiments_config/64x128_Full/experiment_config.yml

# Run with a different LDPC code
PYTHONPATH=. python scripts/run_experiment.py \
  --config config/experiments_config/24x48/experiment_config.yml

# Run all experiment configurations
PYTHONPATH=. python scripts/run_multi_matrix_experiments.py \
  --config-folder config/experiments_config --all-folders
```

### 6.5. Analyzing Results

```bash
# List all available analyses
PYTHONPATH=. python scripts/run_analysis.py --list

# Run a specific analysis
PYTHONPATH=. python scripts/run_analysis.py \
  --experiment_id <experiment_id> --analysis_id <analysis_name>
```

### 6.6. The Five-Seed Ablation and the Paper

A five-seed study crosses the curriculum with replay seeding, at one and at three training epochs, and adds a one-seed control that holds epsilon at its floor after the first epoch: 41 cells under `config/ablation_config/`, generated by `scripts/make_ablation_configs.py` from the design frozen in [docs/ablation/PROTOCOL.md](docs/ablation/PROTOCOL.md). Each cell trains in one to three hours on an RTX 3060 Ti; all 41 take about 80 GPU-hours.

```bash
# Train and evaluate every cell (or pass --folders <cell> ... for some)
PYTHONPATH=. python scripts/run_multi_matrix_experiments.py \
  --config-folder config/ablation_config --all-folders

# Summarize them: RESULTS.md/.json, then DETAILS.md/.json
PYTHONPATH=. python scripts/aggregate_ablation.py
PYTHONPATH=. python scripts/analyze_ablation_details.py
```

`config/epoch_eval/` scores the earlier epoch checkpoints of the three-epoch cells, and `config/baseline_eval/` scores the Gallager-B baseline on every seed's held-out words. The decoder cost benchmark and the test of the agent combined with Min-Sum are described in [scripts/README.md](scripts/README.md).

The records these steps produce are kept under `docs/ablation/`, and every table, figure and number of the accompanying paper regenerates from them in seconds, without a GPU; see [scripts/paper/README.md](scripts/paper/README.md), which also maps the cell IDs to the names the paper uses. The trained weights of the paper's final policy (`NoC_NoE3ep`, five seeds) are included under `models/ablation/`, so its evaluation, cost benchmark and Min-Sum combination rerun without training.

## 7. Documentation

| Document | Purpose |
|----------|---------|
| [docs/MASTER_DESIGN.md](docs/MASTER_DESIGN.md) | Formal MDP specification: state, action, reward, success definitions, Q-learning formulation, training algorithm, neural architecture |
| [docs/MASTER_EXPERIMENT_DESIGN.md](docs/MASTER_EXPERIMENT_DESIGN.md) | Experiment workflow: configuration roles, dataset pipeline, flagship parameters, artifact naming, analysis and report generation |
| [docs/ablation/PROTOCOL.md](docs/ablation/PROTOCOL.md) | The five-seed ablation's design, frozen before the runs, and the record of how it was run |
| [docs/ablation/RESULTS.md](docs/ablation/RESULTS.md), [DETAILS.md](docs/ablation/DETAILS.md) | The ablation's results and detailed analyses, generated from the runs |
| [scripts/README.md](scripts/README.md) | Script-level documentation for all entry points |
| [scripts/paper/README.md](scripts/paper/README.md) | Regenerating the paper's tables, figures and numbers; condition names |

## 8. Acknowledgements

This material is based upon work supported by the National Aeronautics and Space Administration (NASA) under EPSCoR Rapid Response Research award No. 80NSSC23M0156, "Improvements on low-density parity-check (LDPC) codes and high-performance neuromorphic engineering for communication systems." Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of NASA.
