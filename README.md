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
│   ├── baseline_agents.py   # Classical decoders (Min-Sum, Greedy, Random, Omniscient)
│   ├── networks.py          # Neural network architectures (SimpleMLP, LDPCOptimizedMLP)
│   ├── network_factory.py   # Factory for creating networks from configuration
│   ├── reward.py            # Reward functions
│   └── ldpc_utils.py        # Utilities for loading matrices and code metadata
├── training/                # Training pipeline
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
│   └── min_sum_core.py      # Core implementation of the Min-Sum algorithm
├── analysis/                # Analysis and visualization
│   └── analysis.py          # DuckDB SQL-based results analysis
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

### 2.3. Path Management

All file paths are managed through a centralized `path_utils` system that detects the project root automatically and resolves paths consistently regardless of working directory.

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

The `RobustDataGenerator` creates static HDF5 datasets with a six-phase curriculum progressing from single-error patterns to complex multi-error scenarios. Error models include uniform count, uniform BER, burst, cluster, and systematic patterns. All generation is seed-controlled for full reproducibility.

### 4.4. Expert-Guided Learning

The replay buffer can be warm-started with pregenerated expert demonstrations. Available expert agents:

- **OmniscientAgent**: Oracle-based expert with ground truth access (primary expert in the flagship experiment).
- **MinSumExpertAgent**: Episode-level planning using Min-Sum algorithm convergence.
- **GreedyAgent**: Step-by-step penalty-based heuristic selection.
- **RandomAgent**: Baseline random action generation.

Expert experiences are generated offline via `scripts/run_expert_experience.py` and stored as pickle files for reuse across training runs.

### 4.5. Evaluation and Analysis

- **Progressive saving**: Results are saved per training phase and per evaluation agent to manage memory.
- **Compressed storage**: All results are GZIP-compressed JSON files (70-80% size reduction).
- **Database-backed analysis**: Results are loaded into a DuckDB database for SQL-based querying.
- **Automated reports**: Template-driven Markdown report generation from analysis results.

## 5. LDPC Code Specifications

The system supports multiple LDPC codes, selectable via `code_id`:

```yaml
code_id: "64x128"  # Options: "24x48", "56x80", "64x128"
```

| Code | Dimensions (m x n) | d_min | Rate | Correction Capability |
|------|-------------------|-------|------|-----------------------|
| `24x48` | 24 x 48 | 7 | 0.50 | 3 |
| `56x80` | 56 x 80 | 12 | 0.30 | 5 |
| `64x128` | 64 x 128 | 14 | 0.50 | 6 |


All LDPC matrices (H and G) are stored as NumPy arrays in the `data/` directory. The flagship experiment uses `64x128`.

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
environment is reproducible across machines.

### 6.2. Smoke Test (Quick Start)

Before launching the flagship run, verify the toolchain end-to-end with the `64x128_QuickStart` config — 8 000 training samples across three curriculum phases, 5 000 expert experiences, on the order of a few minutes on a single GPU. Produces the same artefact layout as the full run (per-phase training files, an intra-phase flush file, per-agent evaluation files, a model checkpoint, and a per-experiment DuckDB once you build it):

```bash
PYTHONPATH=. python scripts/run_multi_matrix_experiments.py \
  --config-folder config/experiments_config --folders 64x128_QuickStart
```

Note: this overwrites `data/ldpc_64x128_{main,expert}_dataset.h5` if they exist (regenerated on demand by `force_recreate: true`). The slow-to-produce 500 000-experience expert pickle used by `64x128_Full` is kept distinct (`data/quickstart_64x128_omniscient_5000.pkl`) and is not touched.

### 6.3. Reproducing the Main Experiment

The full end-to-end workflow for the flagship `64x128_Full` experiment:

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

## 7. Documentation

| Document | Purpose |
|----------|---------|
| [docs/MASTER_DESIGN.md](docs/MASTER_DESIGN.md) | Formal MDP specification: state, action, reward, success definitions, Q-learning formulation, training algorithm, neural architecture |
| [docs/MASTER_EXPERIMENT_DESIGN.md](docs/MASTER_EXPERIMENT_DESIGN.md) | Experiment workflow: configuration roles, dataset pipeline, flagship parameters, artifact naming, analysis and report generation |
| [scripts/README.md](scripts/README.md) | Script-level documentation for all entry points |

## 8. Acknowledgements

This material is based upon work supported by the National Aeronautics and Space Administration (NASA) under EPSCoR Rapid Response Research award No. 80NSSC23M0156, "Improvements on low-density parity-check (LDPC) codes and high-performance neuromorphic engineering for communication systems." Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of NASA.
