# Changelog

All notable changes to this project are documented here.

This project is archived on Zenodo. The concept DOI
[10.5281/zenodo.20482543](https://doi.org/10.5281/zenodo.20482543) always resolves to the
latest version; each release additionally receives its own version DOI.

## [1.2.0] — 2026-11-08

The release behind the five-seed ablation study and its paper. It adds
multi-epoch training, an epsilon schedule derived from episode counts, the
re-flip restriction and a Gallager-B baseline; it carries the ablation's
configurations, generated records and analysis scripts, and the weights of the
paper's final policy, so every table, figure and number of the paper
regenerates from the repository. The new mechanisms are opt-in, but the
`64x128` curricula changed (see Changed), so results are not comparable bit for
bit with the 1.1.x series: regenerate rather than mix.

### Added

- **Five-seed ablation.** The design is frozen in `docs/ablation/PROTOCOL.md`,
  with the record of how it was run. `scripts/make_ablation_configs.py` writes
  its 41 cells under `config/ablation_config/`: curriculum and replay seeding
  crossed, at one and three epochs, on five seeds, plus a one-seed control
  that holds epsilon at its floor after the first epoch.
  `scripts/aggregate_ablation.py` writes `docs/ablation/RESULTS.md` and
  `RESULTS.json`; `scripts/analyze_ablation_details.py` writes `DETAILS.md`,
  `DETAILS.json` and `learning_curves.csv`. `config/ablation_checks/` holds the
  replicate that reproduced a suspended cell bit for bit.
- **Evaluation-only configurations.** The pipeline now evaluates a config with
  no model when it enables baselines explicitly, scoring those alone. Three
  generated trees use it: `config/epoch_eval/` scores the earlier epoch
  checkpoints of the three-epoch cells (`scripts/make_epoch_eval_configs.py`),
  `config/baseline_eval/` scores Gallager-B on every seed's held-out splits
  (`scripts/make_baseline_eval_configs.py`), and `config/final_eval/`
  re-scores the final policy from its saved weights
  (`scripts/make_final_eval_configs.py`).
- **Gallager-B baseline.** Hard-input message passing in
  `src/advanced/gallager_b.py`, exposed as the `gallager_b` agent. Its flip
  thresholds are chosen on training words by
  `scripts/select_gallager_b_thresholds.py`, recorded in
  `docs/ablation/GALLAGER_B_SELECTION.md`.
- **Decoder cost benchmark.** `scripts/benchmark_decoders.py` measures latency
  on GPU and CPU, operations per frame and batched throughput for the agent and
  the baselines, after checking each sampled frame against the recorded
  evaluation; records go to `docs/ablation/benchmark/`.
- **Agent and Min-Sum combined.** `scripts/cascade_minsum_agent.py` runs
  Min-Sum, the agent on Min-Sum's failures and Min-Sum again;
  `scripts/cascade_controls.py` replaces the agent with cheaper alternatives.
  Records go to `docs/ablation/cascade/`.
- **Minimum-distance certificate.** `scripts/verify_min_distance.py` proves
  $d_{\min} = 14$ for `64x128` by exhaustive enumeration on two information
  sets, in about a second. The same check identifies the matrix as the CCSDS
  (128,64) telecommand code.
- **The paper's generators.** `scripts/paper/` writes every table, figure and
  quoted number of the paper from the records under `docs/ablation/`, in
  seconds and without a GPU; its README maps the condition IDs to the paper's
  names. `scripts/summarize_development_chain.py`,
  `scripts/record_wall_clock.py` and `scripts/extract_example_episode.py` write
  the remaining records it reads.
- **Final-policy weights.** The five trained networks of `NoC_NoE3ep`
  (18 MB) under `models/ablation/`, so the final policy's evaluation, the cost
  benchmark and the Min-Sum combination rerun without training.

- **`max_errors` on `uniform_ber`.** An opt-in cap: the generator redraws any
  pattern heavier than it, so a BER model can be held inside the code's
  correction radius. Every `64x128` dataset config sets it to 6.
- **Epochs as a first-class axis.** Training is partitioned by `(epoch, phase)`
  segment, every training episode carries a run-global `episode_id` and its
  `epoch`, and the training analyses report per epoch. Epsilon fields accept one
  row per epoch. `training.save_epoch_checkpoints` keeps the weights at each
  epoch boundary so epochs can be scored separately.
- **`epsilon_decay: "auto_episodes"`.** Sets epsilon from the fraction of each
  phase completed, counted in episodes, reaching `epsilon_min` after
  `epsilon_anneal_fraction` of the phase. Independent of dataset size and of how
  long episodes turn out to be.
- **`environment.forbid_immediate_reflip`.** Removes the bit flipped on the
  previous step from every agent's legal actions, in training, evaluation and
  expert-experience generation alike, and from the DQN bootstrap maximum. Off
  by default; it changes the MDP.
- **Comparison chain.** `64x128_Full_AutoEp`, `64x128_Full_AutoEp_Uniform`,
  `64x128_Full_Uniform_NoReflip` and `64x128_Full_Uniform_3ep`, each one setting
  apart from the previous, on the shared seed-42 dataset and expert replay.
  `64x128_QuickStart_3ep` is the multi-epoch smoke test.
- **Dataset validation before reuse.** An existing dataset is reused only if it
  matches the requested curriculum config, code width and stratified split, in
  both `run_experiment.py` and the orchestrator.
- **One experiment loader** (`src/analysis/experiment_loader.py`) shared by the
  analysis CLI and the notebook. Reads a run from either half of its output.
- Seven regression scripts under `tests/`, the seventh for Gallager-B.

### Changed

- **`64x128` curricula stay within the correction radius.** The phase 4
  periodic pattern used step 20, whose offsets 0 to 7 placed seven errors on a
  128-bit word: 3.4% of that phase, and 685 held-out episodes, lay beyond
  $t_{\max} = 6$. It now uses step 22, so every offset places exactly six, and
  the BER models are capped at six. Datasets generated from the earlier configs
  no longer match and are refused on reuse; regenerate them.
- **Result file names carry the epoch**:
  `{experiment}_{timestamp}_training_epoch{E}_phase{N}[_flush{F}].json.gz`.
  Discovery reads no earlier form, so results written before this version must
  be regenerated to be analysed.
- The Min-Sum decoder is vectorized over Tanner-graph edges, about 30 to 75
  times faster; the per-node implementation is kept as `ReferenceMinSumDecoder`
  and the two are verified equivalent at every shipped damping.
- Analysis import is one transaction, and a database is reused only for the run
  it holds and only if complete. `run_analysis.py` and
  `run_experiment_report.py` exit nonzero whenever an analysis did not run.
- The target network synchronises on the gradient update that reaches the
  interval rather than at the next episode boundary, and each step's training
  update is recorded on the step that performed it.
- Evaluation writes each agent's episodes through to disk as they are flushed,
  into a temporary file renamed on completion, and reports a failed agent as a
  failed evaluation.
- Progressive saving is one policy per run; a run with it disabled writes the
  single end-of-run training file it always claimed to.
- The orchestrator takes dataset paths and the split from the configurations
  that own them instead of constants.

### Fixed

- `verify_matrix_properties` computes rank over GF(2), and `gf2_rank` scans
  every column rather than the leading square block.
- The `catastrophic_forgetting`, `baseline_superiority` and
  `performance_variance` analyses grouped on `array_length(error_pattern)`, a
  constant, and now group on `num_errors`; the epsilon curves bucket by floor
  division instead of rounding.
- The vectorized Min-Sum decoder handles isolated variables and degree-one
  checks as the reference does, without producing NaN.
- Evaluation scores the checkpoint the run produced rather than a configured
  pretrained model it started from, and baseline agent parameters reach their
  constructors.
- `run_validate_split.py` indexed phase metadata with global offsets;
  `run_qvalue_analysis.py` read `max_steps` from the wrong config block.
- A collector with an episode limit but no flush destination no longer discards
  episodes silently.
- The expert experience generator chooses among the environment's legal
  actions, so an expert config with `forbid_immediate_reflip` honours the
  restriction instead of failing on the first immediate re-flip.

## [1.1.0] — 2026-08-06

A performance release. Decoding behaviour, network architecture, curriculum, reward
function and evaluation pipeline are unchanged.

> **Reproducibility boundary.** Runs remain fully reproducible from the configured `seed`:
> the same seed and configuration produce identical results. But **v1.1.0 does not reproduce
> v1.0.x results bit-for-bit** — the replay buffer's index sampler uses a different algorithm
> and therefore draws a different random stream. The sampling distribution is unchanged
> (uniform, without replacement, 32 distinct indices); only the draw order differs. Anyone
> comparing figures across versions should regenerate them rather than mixing the two.

### Changed

- **Replay sampling no longer costs O(buffer size).** `np.random.choice(n, k, replace=False)`
  draws `k` items by permuting the *entire* range, so its cost scales with the buffer rather
  than the batch. Measured for `k=32`:

  | buffer size | before | after |
  |---:|---:|---:|
  | 50,000 | 449 µs | 3.6 µs |
  | 500,000 | 4,518 µs | 3.6 µs |
  | 2,500,000 | 32,403 µs | 3.6 µs |
  | 5,000,000 | 91,701 µs | 3.6 µs |

  `64x128_Full` configures a 5,000,000-entry buffer, so selecting 32 indices was costing up
  to 91.7 ms per gradient update — far more than the update itself. `Generator.choice` uses
  Floyd's algorithm instead: constant time, identical semantics.

  This also explains why expert-seeded runs were disproportionately slow: seeding starts the
  buffer at 500,000 entries, so sampling begins expensive.

  End-to-end on `64x128_Full` (245,000 episodes, 500,000-experience expert seed):
  **86.9 minutes, against 208–252 minutes previously** — with equivalent results
  (92.1% RL success, matching phase-boundary behaviour).

### Performance (output-identical)

Each of the following was verified to produce byte-identical result files:

- Metrics JSON is encoded once and written once. `json.dump()` streams through one
  `write()` per JSON fragment — 9,022,096 calls in a profiled QuickStart run.
- Dataset phases are read from HDF5 once and cached, rather than four single-element reads
  plus a group lookup per sample (42,006 dataset reads for ~10,500 samples).
- The replay buffer is backed by a ring buffer instead of a `deque`, giving O(1) random
  indexing rather than O(n).
- `DQNAgent.update()` collapses six CUDA synchronisations per update into one.
- Syndrome and penalty vectors update incrementally in `step()` — a bit flip toggles only
  the checks in one column of H — instead of running two sparse mat-vecs per action.

On the smoke test these are worth ~32%; on production runs their effect is small relative to
the sampling fix above.

### Removed

- The `128x256` experiment configuration. It never ran end-to-end, and README section 5 lists
  the supported codes as 24x48, 56x80 and 64x128 — so it shipped both untested and
  undocumented. **The matrices remain**: `data/H_128x256_dmin24.npy`,
  `data/G_128x256_dmin24.npy`, and `code_id: "128x256"` still resolves. A tuned configuration
  may return in a future release.

## [1.0.1] — 2026-08-05

A correctness and reproducibility release. **No algorithmic or architectural changes** — the
decoder, network, curriculum, reward function, and evaluation pipeline are unchanged from
v1.0.0.

### Fixed

- **Expert experience generation is now seeded.** `scripts/run_expert_experience.py` applied no
  seeding at all, so every invocation produced a different expert replay buffer. Because those
  experiences pre-seed the training replay buffer, two runs from an identical configuration and
  seed diverged into materially different training trajectories. All RNGs (`random`, NumPy,
  PyTorch, CUDA) are now seeded from a `seed` key before any component is constructed, and a
  `seed: 42` entry was added to every `expert_experience_config.yml`.

  This is the most consequential fix in the release: under v1.0.0 the published artifact could
  not reproduce its own results, since the expert `.pkl` is not distributed and had to be
  regenerated.

- **`64x128_Full`: restored `update_frequency: 1`.** v1.0.0 shipped `4`. The parameter was
  introduced as an optimisation knob *after* the published experiments were run, which had an
  effective value of `1` (one gradient update per environment step). The released configuration
  therefore did not reproduce the paper's training schedule. It also left the flagship condition
  at a quarter of the gradient updates of the ablations it is compared against, all of which
  omit the key and inherit the default of `1`.

  The published results themselves are unaffected — every condition in the paper ran at an
  effective `1`. This restores the shipped configuration to match them.

- **`64x128_OnlyE`: enabled `use_pregenerated_file: true`.** The "only-expert" ablation was not
  loading expert experiences, contradicting the condition it is named for.

- **Requesting an unavailable CUDA device now fails fast.** Previously a config specifying
  `device: "cuda"` on a machine without CUDA printed a single easily-missed line and completed
  the entire run on CPU, producing output indistinguishable from a GPU run. `DQNAgent` now
  raises a `RuntimeError` with diagnostic detail. Set `strict_device: false` at the top level of
  an experiment config to restore the previous fallback, which now also emits a `RuntimeWarning`.
  Automatic device selection (no `device` key) is unaffected.

### Changed

- `pyproject.toml`: version `1.0.1`; `torch` constraint tightened from `>=2.0.0` to
  `>=2.6,<3.0`. The verified environment is torch 2.10.0+cu128 — reproduce it with
  `uv sync --frozen`.
- Removed an inert `[[tool.uv.index]]` declaration for `download.pytorch.org/whl/cu125`. It was
  never bound to `torch` via `[tool.uv.sources]`, so it had no effect on resolution and implied
  a CUDA pin that did not exist. Replaced with a comment documenting how to pin a specific CUDA
  build deliberately.
- Added `.python-version` pinning CPython 3.12, so `uv` provisions a matching interpreter
  automatically.
- `uv.lock` regenerated (required by the version bump). Same 82 packages resolved, with torch
  unchanged at 2.10.0. Two incidental corrections: the lock now records the full
  `requires-python = ">=3.12, <3.14"` (it previously stored only the lower bound), and wheel
  entries for Python 3.14+ were pruned as unreachable. The lock format advanced to
  `revision = 3`, which requires a reasonably recent `uv` to read.
- `64x128_QuickStart`: corrected a stale comment claiming its `update_frequency: 4` matched
  `64x128_Full`. The value is intentional for a smoke test; only the comment was wrong.

### Documentation

- README §2.2: added notes on the `update_frequency` trade-off (wall-clock against sample
  efficiency, with the requirement that it be held constant across compared configurations) and
  on reproducibility and device handling.
- README §6.1: documented `uv sync --frozen`, the pinned interpreter, and a GPU verification
  command.
- Added this changelog.

## [1.0.0] — 2026-05-31

Initial public release. Archived at
[10.5281/zenodo.20482544](https://doi.org/10.5281/zenodo.20482544).
