# Changelog

All notable changes to this project are documented here.

This project is archived on Zenodo. The concept DOI
[10.5281/zenodo.20482543](https://doi.org/10.5281/zenodo.20482543) always resolves to the
latest version; each release additionally receives its own version DOI.

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
