# Master Experiment Design

## 1. Purpose and Boundary

This document describes how experiments are organized and executed in the repository, with emphasis on the main `64x128_Full` experiment family. It complements [MASTER_DESIGN.md](MASTER_DESIGN.md):

- [MASTER_DESIGN.md](MASTER_DESIGN.md) defines the decoding problem and MDP.
- This document defines the workflow, configuration roles, artifact flow, and the concrete parameter choices used for the flagship experiment.

To reduce redundancy, this document assumes the MDP formulation is already understood and does not repeat the full state-action-reward specification.

## 2. Execution Modes

The codebase supports two practical execution modes.

### 2.1 Single Experiment Mode

The primary entry point is:

```bash
PYTHONPATH=. python scripts/run_experiment.py \
  --config config/experiments_config/64x128_Full/experiment_config.yml
```

This mode executes the experiment configuration directly. Depending on the YAML flags, it may perform:

- dataset creation
- DQN training
- evaluation

An optional `--output-parent-dir` argument rewrites only the configured results and model directories. It does not relocate datasets or pregenerated expert replay files.

### 2.2 Multi-Configuration Orchestration Mode

Batch execution is handled by:

```bash
PYTHONPATH=. python scripts/run_multi_matrix_experiments.py \
  --config-folder config/experiments_config --all-folders
```

This orchestrator discovers experiment folders and runs a four-stage pipeline:

1. Main dataset creation
2. Expert dataset creation
3. Expert experience generation
4. Full experiment execution

It supports both:

- matrix folders such as `64x128`
- variant folders such as `64x128_Full`

## 3. Configuration Set for `64x128_Full`

The flagship experiment folder is:

`config/experiments_config/64x128_Full/`

It contains four coordinating configuration files:

- `main_dataset_config.json`
- `expert_dataset_config.json`
- `expert_experience_config.yml`
- `experiment_config.yml`

These files serve distinct purposes and should be documented together, not merged into a single monolithic narrative.

## 4. Workflow Design

### 4.1 Stage 1: Main Dataset Creation

The main training/evaluation dataset is generated from:

- `config/experiments_config/64x128_Full/main_dataset_config.json`

and written to:

- `data/ldpc_64x128_main_dataset.h5`

The generator is deterministic given the config seed and uses a six-phase curriculum. Importantly, when `total_samples` is present, the code rescales the per-phase `num_samples` values proportionally. For `64x128_Full`:

- configured `total_samples`: `350000`
- configured phase weights: `5000, 15000, 15000, 20000, 30000, 15000`
- effective phase sizes after proportional rescaling:
  - Phase 1: `17500`
  - Phase 2: `52500`
  - Phase 3: `52500`
  - Phase 4: `70000`
  - Phase 5: `105000`
  - Phase 6: `52500`

After generation, the dataset is split with `test_size=0.3`, using a stratified per-phase split. Training iterates over the `train` split; evaluation uses the `test` split.

### 4.2 Stage 2: Expert Dataset Creation

The expert demonstration dataset is generated from:

- `config/experiments_config/64x128_Full/expert_dataset_config.json`

and written to:

- `data/ldpc_64x128_expert_dataset.h5`

This config uses the same six-phase curriculum shape but a smaller total sample count:

- configured `total_samples`: `115000`
- effective phase sizes after rescaling:
  - Phase 1: `5750`
  - Phase 2: `17250`
  - Phase 3: `17250`
  - Phase 4: `23000`
  - Phase 5: `34500`
  - Phase 6: `17250`

The separation between main and expert datasets is deliberate:

- the main dataset drives RL training and held-out evaluation
- the expert dataset drives generation of offline demonstration transitions

### 4.3 Stage 3: Expert Experience Generation

Pregenerated expert replay is produced by:

```bash
PYTHONPATH=. python scripts/run_expert_experience.py \
  --config config/experiments_config/64x128_Full/expert_experience_config.yml
```

For the flagship setup:

- expert dataset: `data/ldpc_64x128_expert_dataset.h5`
- agent type: `omniscient`
- environment: `max_steps=12`, `max_flips_vector=7`
- target experience count: `500000`
- output file: `data/64x128_omniscient_500000.pkl`

The experience generator iterates over the expert dataset using the `all` split, not just the training subset. The result is a replay buffer file that can be loaded directly at RL training start.

### 4.4 Stage 4: RL Training and Evaluation

The final stage is driven by:

- `config/experiments_config/64x128_Full/experiment_config.yml`

This configuration currently enables all three pipeline flags:

- `run_dataset_creation: true`
- `run_training: true`
- `run_evaluation: true`

As implemented, that means `run_experiment.py` will recreate the main dataset internally even if Stage 1 already created it. Because `force_recreate: true` is also set in `dataset_creation`, the orchestrated four-stage workflow contains a duplicated behavior: Stage 1 creates the main dataset, then Stage 4 regenerates it. Since data generation is fully seed-controlled (`seed: 42` in the dataset config, `seed=42` for the train/test split), the regenerated dataset is identical. The only cost is the extra generation time.

This design is intentional: keeping `force_recreate: true` in the experiment config allows testing different flag combinations (e.g., running Stage 4 standalone with dataset creation enabled) without requiring manual file cleanup between runs.

## 5. Flagship `64x128_Full` Training Design

### 5.1 Environment and Task Limits

The main experiment uses:

- code: `64x128`
- maximum episode steps: `12`
- maximum flip-vector weight: `7`

These limits define the actual success regime described in [MASTER_DESIGN.md](MASTER_DESIGN.md).

### 5.2 Replay and Optimization

The training configuration specifies:

- epochs: `1`
- replay buffer capacity: `5000000`
- batch size: `32`
- target network update frequency: `1000`
- optimizer learning rate: `1e-4`
- weight decay: `1e-4`
- Q-network update frequency: every `4` environment steps

The experiment also enables expert replay seeding through:

- `use_pregenerated_file: true`
- `pregenerated_file: data/64x128_omniscient_500000.pkl`

This is a central part of the flagship design, not an optional footnote.

### 5.3 Epsilon Schedule

Exploration is phase-dependent and reset across curriculum transitions:

- decay type: `linear`
- `epsilon_phase_reset: true`
- `epsilon_start = [1.0, 0.8, 0.6, 0.6, 0.6, 0.2]`
- `epsilon_decay = [0.0000092105, 0.0000024291, 0.0000020243, 0.0000012995, 0.0000007628, 0.0000006061]`
- `epsilon_min = [0.3, 0.2, 0.1, 0.1, 0.1, 0.01]`

This schedule is part of the experiment design because it interacts directly with the six-phase curriculum.

### 5.4 Network Choice

The flagship network configuration is:

- type: `LDPCOptimizedMLP`
- `hidden_dim1: 640`
- `dropout_rate: 0.05`
- `leaky_slope: 0.01`

Only `hidden_dim1` is specified in the YAML. The remaining widths come from the network defaults:

- `hidden_dim2 = 480`
- `hidden_dim3 = 360`
- `hidden_dim4 = 270`

### 5.5 Evaluation Scope

The available baseline agents are:

- `random` -- random action selection
- `greedy` -- heuristic bit selection based on penalty values
- `simple_classical` -- simple classical decoder
- `min_sum` -- classical Min-Sum step-by-step decoding
- `min_sum_expert` -- episode-level planning using Min-Sum convergence

The flagship `64x128_Full` experiment enables `greedy` and `min_sum_expert` alongside the RL agent, providing a direct comparison between the learned policy and two classical baselines of different sophistication. The remaining agents (`random`, `simple_classical`, `min_sum`) are disabled by default.

## 6. Curriculum Content

Both dataset configs use the same six semantic phases:

1. Foundation
2. Intermediate
3. Advanced
4. Expert
5. Master
6. Monster

Across these phases, the error models include:

- `uniform_count`
- `uniform_ber`
- `burst`
- `cluster`
- `systematic`

The curriculum moves from very low-weight errors toward harder mixtures that include:

- fixed-count errors up to 6 flips
- BER-based random corruption
- contiguous bursts
- localized clustered errors
- structured periodic error patterns

This phase structure is a key experiment design decision and should be documented here rather than repeated in the master problem formulation.

## 7. Artifact Layout and Naming

### 7.1 Datasets and Replay Artifacts

For `64x128_Full`, the main persistent inputs are:

- main dataset: `data/ldpc_64x128_main_dataset.h5`
- expert dataset: `data/ldpc_64x128_expert_dataset.h5`
- pregenerated expert replay: `data/64x128_omniscient_500000.pkl`

### 7.2 Model Checkpoints

The default saved checkpoint pattern is:

- `models/ldpc_rl_agent_{timestamp}.pth`

The code does not include the code identifier in the checkpoint filename by default.

### 7.3 Results Files

With progressive saving enabled, the main result files are not a single monolithic training/evaluation pair. The active patterns are:

- training phase files: `{experiment_name}_{timestamp}_training_phase{N}.json.gz`
- training flush files when memory limits are hit: `{experiment_name}_{timestamp}_training_phase{N}_flush{K}.json.gz`
- evaluation files per agent: `{experiment_name}_{timestamp}_evaluation_agent_{name}.json.gz`

Legacy aggregate files such as:

- `{experiment_name}_{timestamp}_training.json.gz`
- `{experiment_name}_{timestamp}_evaluation.json.gz`

are not the primary artifacts when progressive saving is enabled.

### 7.4 Logs

The current implementation writes a global experiment log with pattern:

- `experiment_log_{timestamp}.txt`

The workflow does not currently materialize a richer per-stage summary artifact such as `experiment_summary.json`.

## 8. Analysis and Report Generation

After an experiment completes, two scripts process the compressed result files into structured analyses.

### 8.1 Database-Backed Analysis

```bash
PYTHONPATH=. python scripts/run_analysis.py --experiment_id <timestamp> --analysis_id <analysis_name>
```

The analysis module (`src/analysis/analysis.py`) loads the compressed JSON result files into a normalized DuckDB database. This database is created on first use and persisted as `{experiment_id}.db` in the results directory. Available analyses are defined as SQL query templates in `config/analysis_definitions.json`, organized by category (e.g., `performance_analysis`, `training_dynamics`, `behavioral_analysis`) and filterable by data focus and complexity.

To list all available analyses:

```bash
PYTHONPATH=. python scripts/run_analysis.py --list
```

### 8.2 Automated Report Generation

```bash
PYTHONPATH=. python scripts/run_experiment_report.py --experiment-id <timestamp>
```

The report generator (`scripts/run_experiment_report.py`) is template-driven: it discovers analysis placeholders (e.g., `{{analysis_id}}`) in a Markdown template, executes only those analyses, and populates the template with the results. The output is a self-contained Markdown report. Analysis metadata (titles and descriptions) is included by default and can be suppressed with `--no-metadata`.

Both tools reuse the same DuckDB database and analysis definitions, so the report generator does not duplicate the database creation step if the analysis module has already been run for the same experiment.

## 9. Current Implementation Caveats

The following points are important for accurate documentation.

### 9.1 Dataset Regeneration in Stage 4

The orchestrator treats Stage 1 and Stage 4 as separate steps, but the flagship Stage 4 config has dataset creation enabled with `force_recreate: true`. This means Stage 4 regenerates the main dataset that Stage 1 already created. As noted in Section 4.4, this is a deliberate convenience for testing flexibility, not a correctness issue: deterministic seeding guarantees identical output.

### 9.2 Output Override Scope

The multi-run script exposes `--output-dir`, but that override reaches Stage 4 only through `run_experiment.py --output-parent-dir`. In practice it redirects:

- results directory
- models directory

It does not redirect:

- generated dataset paths
- expert replay file paths

### 9.3 Validation Scope

The orchestration script does validate the presence of required config files and several required fields. However, its validation is still limited relative to what a final publication-facing workflow description might imply. It is better to describe current validation as basic structural validation, not as full semantic verification of all paths, dimensions, and runtime dependencies.

## 10. Recommended Role of This Document

Researchers should use this document to answer:

- Which configs define the flagship experiment?
- How are the datasets, expert replay, training, and evaluation connected?
- What concrete parameters are used for `64x128_Full`?
- What artifacts are produced, and how are they named?
- What current workflow caveats matter when reproducing results?
