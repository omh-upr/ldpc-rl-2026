# Ablation Protocol: Curriculum and Expert Seeding on the (128,64) Code

Status: protocol frozen 2026-09-12; merged into `main` 2026-09-13; all 20
cells run and checked as of 2026-09-16; results aggregated in `RESULTS.md`
on 2026-09-17 (see section 10).

## 1. Question

The published ablation (paper Table 6) toggled the six-phase curriculum and the
omniscient expert replay seeding in a 2×2 design, one run per cell, and reported
gaps of 1.6 to 2.7 percentage points from four single-seed point estimates. A
same-code rerun that changed only the random stream moved the full configuration
by 0.9 points, so the ordering of the two single-mechanism cells, and arguably
each mechanism's individual contribution, is not established. This study reruns
the 2×2 with five seeds per cell, on the current code and the current best
one-epoch configuration, and reports mean and 95% confidence interval per cell.

The honest outcome may be that curriculum-only and expert-only do not separate.
That result is reported as such.

## 2. Conditions

| Condition | Curriculum | Expert seeding | Dataset config | `expert_experience.use_pregenerated_file` |
|---|---|---|---|---|
| `NoC_NoE` | no | no | flat | `false` |
| `OnlyC` | yes | no | curriculum | `false` |
| `OnlyE` | no | yes | flat | `true` |
| `Full` | yes | yes | curriculum | `true` |

Nothing else differs between conditions. In particular, unlike the retired
`64x128_OnlyC`, `64x128_OnlyE` and `64x128_NoC_NoE` folders, every cell uses
the same replay buffer size, the same epsilon schedule, the same evaluation
agents, and its own dataset path.

## 3. Fixed configuration

The base is `config/experiments_config/64x128_Full_Uniform_NoReflip`, the best
one-epoch configuration measured to date. Every cell inherits these settings.

| Group | Setting | Value |
|---|---|---|
| Training | epochs | 1 |
| | replay buffer | 5,000,000 |
| | batch size | 32 |
| | learning rate, weight decay | 1e-4, 1e-4 |
| | target update | every 1,000 updates |
| | update frequency | 1 per environment step |
| Exploration | `epsilon_decay` | `"auto_episodes"`, linear |
| | `epsilon_start`, `epsilon_min` | 0.5, 0.01, one value broadcast to every phase |
| | `epsilon_anneal_fraction` | 0.5 |
| Network | type, `hidden_dim1`, dropout, leaky slope | `LDPCOptimizedMLP`, 640, 0.05, 0.01 |
| Environment | `max_steps`, `max_flips_vector` | 12, 7 |
| | `forbid_immediate_reflip` | `true` |
| Evaluation | agents | `rl`, `greedy`, `min_sum_expert` |
| | held-out size | 105,000 episodes, stratified 30% split |

The re-flip restriction is part of the method under test, so the ablation
dissects the method as it will be claimed. Every agent evaluated faces the same
restriction.

Under `"auto_episodes"` the schedule is a function of phase progress only, so
the flat conditions must present the same phase boundaries as the curriculum
conditions for the exploration trajectory to be identical (section 4.2).

## 4. Seeds and data

Five seeds: 42, 43, 44, 45, 46. A seed is a block: it determines every random
input of the four cells that carry it, and the four cells at one seed share
those inputs.

### 4.1 Per-seed files

| File | Path | Used by |
|---|---|---|
| curriculum dataset | `data/ablation/s<seed>/curriculum.h5` | `OnlyC`, `Full` |
| flat dataset | `data/ablation/s<seed>/flat.h5` | `NoC_NoE`, `OnlyE` |
| expert dataset | `data/ablation/s<seed>/expert.h5` | expert replay generation |
| expert replay | `data/ablation/s<seed>/omniscient_500000.pkl` | `OnlyE`, `Full` |

Seed derivation, all from the block seed `s`:

- main dataset config `seed`: `s`
- expert dataset config `seed`: `s + 100` (the reference folder uses 37 against
  a main seed of 42; the offset keeps the two streams distinct)
- expert experience config `seed`: `s`
- experiment config `seed`: `s` (network initialisation, exploration, replay
  sampling)

The train/test split seed is fixed at 42 inside dataset creation and is not
varied. It does not need to be: the samples themselves differ per seed.

### 4.2 Datasets

The curriculum dataset is the reference one: six phases sized 5,000, 15,000,
15,000, 20,000, 30,000 and 15,000 (scaled by the generator to 350,000 in
total), with the phase mixtures of
`config/experiments_config/64x128_Full/main_dataset_config.json`.

The flat dataset keeps those six phase sizes and gives every phase the same
mixture: the 23-source blend of
`config/experiments_config/64x128_NoC_NoE/main_dataset_config.json`, which
reproduces the curriculum's marginal over error sources exactly (checked
2026-09-12, maximum weight difference 0.0). The two datasets therefore have the
same error distribution, the same phase boundaries and the same epsilon
trajectory; they differ only in whether difficulty is ordered.

Both datasets keep every pattern within the correction radius $t_{\max} = 6$:
periodic patterns use steps that never place a seventh error, and BER models
carry `max_errors: 6` (see `docs/MASTER_EXPERIMENT_DESIGN.md`, section 6).

The expert dataset is the reference six-phase expert curriculum (115,000
samples). The expert replay holds 500,000 omniscient transitions generated
from it. Both are shared by all four cells at a seed; the replay file is only
loaded by the two expert-seeded cells. Its transitions are unordered
`(s, a, r, s')` tuples, so their curriculum origin is immaterial to the flat
conditions.

### 4.3 What the design does and does not pair

- Within a seed, `Full` against `OnlyC` and `OnlyE` against `NoC_NoE` are
  scored on identical held-out episodes: the expert effect is a paired
  comparison.
- Within a seed, the curriculum and flat datasets are different draws from the
  same distribution, so the curriculum effect is compared across held-out sets
  of equal marginal but different samples. The within-run binomial standard
  error at 105,000 episodes is about 0.07 points, an order of magnitude below
  the seed-to-seed spread, so this costs little precision.
- The expert side's own sampling variance is included, since the replay file is
  regenerated per seed.

## 5. Configuration folders

All twenty cells live under `config/ablation_config/`, one folder per cell,
named `<Condition>_s<seed>` (for example `OnlyE_s44`), each holding the four
files the orchestrator requires:

- `main_dataset_config.json`: curriculum or flat, with the block seed
- `expert_dataset_config.json`: reference expert curriculum, seed `s + 100`
- `expert_experience_config.yml`: seed `s`, per-seed dataset and output paths
- `experiment_config.yml`: the base configuration with
  `experiment_name: "Ablation_64x128_<Condition>_s<seed>"`, the per-seed
  dataset and replay paths, the condition's `use_pregenerated_file`, and
  `output_paths` of `results/ablation/<Condition>_s<seed>/` and
  `models/ablation/<Condition>_s<seed>/`

The folders are written by `scripts/make_ablation_configs.py` from the base
folder and the condition table above, and are not edited by hand. Regenerating
them is the way to change any of them. The generator refuses to overwrite a
folder that already holds results, so a finished cell cannot be silently
redefined.

The separate root keeps `config/experiments_config/` as the catalogue of
curated runs, and keeps a future `--all-folders` sweep of that catalogue from
picking up twenty ablation cells.

## 6. Execution

Data stages take under a minute per cell; training about 55 minutes and
evaluation about 6 on the reference GPU, so about one hour per cell and about
20 GPU-hours in total, run sequentially.

```bash
PYTHONPATH=. python scripts/run_multi_matrix_experiments.py \
    --config-folder config/ablation_config --all-folders
```

The orchestrator runs the folders in name order, so the four cells of a seed
run together. Stage 1 and 2 reuse a per-seed dataset that already exists and
matches its config, and refuse one that does not. Stage 3 has no such check
and regenerates the replay file for every cell; at 15 seconds and a fixed seed
this is idempotent and is left as is.

Checks after each seed's four cells, before moving on:

- the orchestrator log reports all four stages successful for each cell;
- `greedy` and `min_sum_expert` results are identical between the two cells
  that share a dataset (this proves the held-out set is the same);
- the RL result is in a plausible range. The archived seed-42 run of the base
  configuration scored 94.51%, on a dataset that still contained seven-error
  words; the new datasets do not, so a somewhat higher number is expected.

Interrupted cells are rerun by deleting their `results/ablation/<cell>/`
folder and rerunning the orchestrator with `--folders <cell>`.

## 7. Analysis and reporting

Per cell, from the evaluation files (no database is needed for the headline
numbers; `scripts/run_analysis.py --results-dir results/ablation/<cell>`
rebuilds one on demand for the SQL analyses):

- success rate, frame error rate, final bit error rate and mean steps, for
  `rl`, `greedy` and `min_sum_expert`
- `rl` success rate by error weight 0 to 6
- `rl` success rate by error model

Aggregation, by `scripts/aggregate_ablation.py` (it reads the twenty
evaluation folders and the per-seed datasets and writes
`docs/ablation/RESULTS.md` in full on every run):

- per cell: mean, standard deviation and 95% confidence interval over the five
  seeds, using the t distribution with four degrees of freedom
  ($t_{0.975,4} = 2.776$, so the half-width is $1.24\,s$);
- per contrast: within-seed differences and their confidence intervals for the
  four single-mechanism contrasts (`Full - OnlyC`, `OnlyE - NoC_NoE`,
  `Full - OnlyE`, `OnlyC - NoC_NoE`) and the joint contrast `Full - NoC_NoE`;
- the same per error weight, since the mechanisms may act at different weights;
- the baselines' spread across seeds, which the paper currently lacks;
- for each seed, the fraction of held-out error patterns that also occur in
  that dataset's training split, by weight (the disjointness figure the review
  audit asks for, item M4).

Reported to one decimal in points, with the interval; no cell is bolded. The
table replaces paper Table 6 and its surrounding paragraph.

## 8. Differences from the published ablation

For the record, the new cells differ from the paper's Table 6 runs in:

- epsilon schedule (`auto_episodes`, 0.5 to 0.01 uniformly per phase, instead
  of tapered per-update decrements);
- `forbid_immediate_reflip: true`;
- replay buffer 5,000,000 in every cell (the old single-mechanism folders used
  2,500,000);
- greedy and min-sum baselines evaluated in every cell;
- one dataset per seed and per dataset kind at its own path (the old folders
  rewrote the shared dataset path, so cells were scored on different held-out
  sets without that being a design choice);
- no error pattern above six flips;
- five seeds per cell.

The published numbers are therefore not a sixth seed of this study and are
not pooled with it.

## 9. Out of scope

- The three-epoch schedule (`64x128_Full_Uniform_3ep`) is a separate follow-up
  run on the full condition, not a cell here.
- Other codes. The (48,24) and (80,24) codes are addressed separately.
- Pooling the curriculum phases at training time, which would let all four
  cells share one held-out set per seed, needs a trainer flag and is deferred.

## 10. State and how to resume

Last updated 2026-09-17, after `RESULTS.md` was generated; campaign complete. Update this section
whenever the campaign advances.

**Where the work is.** On `main`, merged from `feat/ablation-study` with
`--no-ff` as `4f8fb58` before the first run. Not pushed. The branch's
commits, in order: the correction-radius fix (`150e8fb`), this protocol
(`d44de99`), the generator and the twenty cells (`217ff9a`), and the first
version of this section (`ad3e0c1`).

**Run schedule.** The cells are run in pairs that share a dataset, so each
sitting is about two and a half hours and ends with a unit the section 6
checks apply to: `OnlyC_s<seed>` with `Full_s<seed>` (curriculum), then
`NoC_NoE_s<seed>` with `OnlyE_s<seed>` (flat), seed 42 first, then 43 to 46.
Ten sittings in all. A cell cannot be paused; only start a pair when the
machine will stay on until it finishes. The desktop suspends after an hour
of inactivity, which put sitting 7 to sleep for five hours mid-cell (the
run survived the suspend and resumed, see below), so each launch holds an
idle inhibitor for its own lifetime. Each pair is launched detached:

```bash
PYTHONPATH=. setsid nohup systemd-inhibit --what=idle \
    --who="LDPC ablation" --why="training" \
    python scripts/run_multi_matrix_experiments.py \
    --config-folder config/ablation_config \
    --folders OnlyC_s43 Full_s43 \
    > experiments/ablation_batch03.out 2>&1 < /dev/null &
```

The inhibitor blocks only idle-triggered suspend; a manual suspend still
works and would pause the run the same way.

**Progress.** Measured cell times on the reference GPU (RTX 3060 Ti) are 60
to 75 minutes, not the hour estimated in section 6.

| Sitting | Cells | Done | Log | RL held-out |
|---|---|---|---|---|
| 1 | OnlyC_s42, Full_s42 | 2026-09-13 | `experiment_log_20260913_164656.txt` | 89.56%, 93.96% |
| 2 | NoC_NoE_s42, OnlyE_s42 | 2026-09-13 | `experiment_log_20260913_194514.txt` | 87.76%, 94.17% |
| 3 | OnlyC_s43, Full_s43 | 2026-09-14 | `experiment_log_20260913_220708.txt` | 89.31%, 93.88% |
| 4 | NoC_NoE_s43, OnlyE_s43 | 2026-09-14 | `experiment_log_20260914_180545.txt` | 90.23%, 94.26% |
| 5 | OnlyC_s44, Full_s44 | 2026-09-14 | `experiment_log_20260914_210754.txt` | 86.10%, 94.18% |
| 6 | NoC_NoE_s44, OnlyE_s44 | 2026-09-15 | `experiment_log_20260914_233109.txt` | 82.24%, 93.83% |
| 7 | OnlyC_s45, Full_s45 | 2026-09-15 | `experiment_log_20260915_153257.txt` | 76.33%, 94.06% |
| 8 | NoC_NoE_s45, OnlyE_s45 | 2026-09-15 | `experiment_log_20260915_153257.txt` | 87.51%, 93.95% |
| 9 | OnlyC_s46, Full_s46 | 2026-09-15 | `experiment_log_20260915_203052.txt` | 89.33%, 94.15% |
| 10 | NoC_NoE_s46, OnlyE_s46 | 2026-09-16 | `experiment_log_20260916_175227.txt` | 75.09%, 93.70% |

Sitting 1 passed the section 6 checks: all four stages successful for both
cells, greedy (71.92%) and min-sum (99.91%) episode-for-episode identical
across the two cells, and the full configuration at 93.96%, within half a
point of the archived 94.51% on the old dataset. The "somewhat higher"
expectation stated in section 6 did not materialise on this seed; whether
that is seed variance is what the remaining seeds will show.

Sitting 2 passed the same checks on the flat dataset: greedy (71.85%) and
min-sum (99.89%) identical across `NoC_NoE_s42` and `OnlyE_s42`, and the
flat file holds 350,000 patterns of weight at most six. Seed 42 is complete.
Its four RL numbers, NoC_NoE 87.76%, OnlyC 89.56%, OnlyE 94.17%, Full 93.96%,
put expert seeding at four to six points and the curriculum at under two
points, with the curriculum's contribution vanishing once the expert replay
is present. One seed; the contrasts in section 7 are what will be reported.

Sitting 3 passed the checks on seed 43's curriculum dataset (baselines
71.84% and 99.88%, identical across the pair; curriculum and expert files
at weight six or below). Its two RL numbers sit within a quarter point of
seed 42's, so the seed-to-seed spread on the curriculum cells is small so
far.

Sitting 4 passed the checks on seed 43's flat dataset (baselines 71.70% and
99.90%, identical across the pair; flat file at weight six or below). Seed
43 is complete: NoC_NoE 90.23%, OnlyC 89.31%, OnlyE 94.26%, Full 93.88%.
The no-expert flat cell moved two and a half points between seeds, the
largest seed effect so far, and on this seed the curriculum alone scores
below no curriculum. Expert seeding is again worth four to five points.

Sitting 5 passed the checks on seed 44's curriculum dataset (baselines
71.93% and 99.90%, identical across the pair; curriculum and expert files
at weight six or below). `OnlyC_s44` at 86.10% is more than three points
below the same cell on seeds 42 and 43, while `Full_s44` at 94.18% stays
within a third of a point of the other two seeds. The pattern is now
consistent: the expert-seeded cells are stable across seeds and the
no-expert cells are not, so the expert contrast will carry a narrow
interval and the curriculum contrast a wide one.

Sitting 6 passed the checks on seed 44's flat dataset (baselines 72.00% and
99.88%, identical across the pair; flat file at weight six or below). Seed
44 is complete: NoC_NoE 82.24%, OnlyC 86.10%, OnlyE 93.83%, Full 94.18%.
`NoC_NoE_s44` is the lowest cell so far, five to eight points under the
same cell on seeds 42 and 43. Its training curve rose normally in every
phase but levelled off lower in the last two, and its held-out deficit is
spread evenly over all error weights with the usual failure modes, so it is
a weak convergence rather than a broken run and stays in the table. Seeds
are not rerun on the strength of their result.

Sittings 6 to 8 were launched as one chain of three orchestrator runs
(`experiments/ablation_chain_6_7_8.out` records the start and end of each),
which stops at the first failed cell.

Sitting 7 was suspended by the desktop's idle timer three minutes into
`OnlyC_s45`, at about 01:46 on 2026-09-15, and resumed at 07:08 when the
machine woke. The training process kept its state in memory and continued
from where it stopped, so the result is unaffected apart from wall-clock
time; an idle inhibitor was then attached to the chain for its remaining
cells.

At 07:20 on 2026-09-15 the chain was stopped on request so the machine
could be shut down, with `OnlyC_s45` in phase 2 of 6 and `Full_s45` not yet
started. The partial output of `OnlyC_s45` was deleted and sittings 7 and 8
were relaunched at 15:32 the same day as one orchestrator run
(`experiments/ablation_batch07_08.out`, log
`experiment_log_20260915_153257.txt`) under the inhibitor.

Sitting 7 passed the checks on seed 45's curriculum dataset (baselines
71.80% and 99.88%, identical across the pair; curriculum file at weight six
or below). `OnlyC_s45` at 76.33% is the lowest cell of the campaign, ten to
thirteen points under the same cell on seeds 42 to 44, while `Full_s45` at
94.06% is again within a third of a point of the other seeds. The
`OnlyC_s45` training curve has the same shape as `NoC_NoE_s44`'s, rising in
every phase but levelling off five to fifteen points lower than the other
`OnlyC` seeds and ending phase 6 at 57% against 69% to 76%; its held-out
deficit is spread over every error weight and its failures are ordinary
step-limit timeouts. A weak convergence, kept in the table under the same
rule as before.

Sitting 8 passed the checks on seed 45's flat dataset (baselines 71.84% and
99.89%, identical across the pair; flat and expert files at weight six or
below). Seed 45 is complete: NoC_NoE 87.51%, OnlyC 76.33%, OnlyE 93.95%,
Full 94.06%. `NoC_NoE_s45` converged normally, its training tail reaching
85% in phase 6, and lands within a quarter point of the same cell on seed
42. On this seed the curriculum alone is eleven points below no curriculum,
the widest curriculum contrast so far and the third no-expert cell to
converge weakly; the two expert-seeded cells are again within a third of a
point of every other seed. Sittings 7 and 8 ran as one orchestrator run of
4 h 51 min. Seed 46 remains.

Sitting 9 passed the checks on seed 46's curriculum dataset (baselines
72.00% and 99.90%, identical across the pair; curriculum and expert files
at weight six or below). Seed 46's three data files were built on this
first use. `OnlyC_s46` at 89.33% converged normally and matches seeds 42
and 43; `Full_s46` at 94.15% keeps the five Full cells within a third of a
point of each other (93.88% to 94.18%).

Sitting 10 was launched at 17:52 on 2026-09-16
(`experiments/ablation_batch10.out`, log
`experiment_log_20260916_175227.txt`) and built seed 46's `flat.h5` on
first use. The machine was suspended by hand at 18:09, two minutes into
phase 3 of `NoC_NoE_s46`, and resumed at 20:26; as with sitting 7, the
training process kept its state in memory and continued, and the phase 3
file that spans the sleep has contiguous episode ids with no gap. The cell
is charged 214 minutes of wall clock, of which 137 were asleep. Sitting 10
passed the checks on seed 46's flat dataset (baselines 72.06% and 99.89%,
identical across the pair; flat and expert files at weight six or below).
`NoC_NoE_s46` at 75.09% is the lowest cell of the campaign, below
`OnlyC_s45`, and shows the same weak-convergence signature: training tail
per phase 14.1, 27.5, 40.5, 48.0, 67.0 and 74.0% against 78.1 and 84.6% in
the last two phases of seed 45, the deficit spread over every error weight
(94.5% at weight one against 96.6 to 98.3% for seeds 44 and 45; 54.2% at
weight six against 61.5 to 70.2%), and the failures ordinary step-limit
timeouts. Under the no-rerun rule it stands. `OnlyE_s46` at 93.70% is the
lowest of the five OnlyE cells but within their 93.70% to 94.26% range.

**Suspend check on `NoC_NoE_s46`.** Because the weakest cell was also the
one paused by the manual suspend, it was replicated on 2026-09-16 from
22:38 to 23:57 without any suspend, as a check and not a replacement. The
replicate is `config/ablation_checks/NoC_NoE_s46_rerun/`, a copy of the
cell differing only in experiment name and output folders; it read the
same `data/ablation/s46/` files (the validator confirmed both matched their
configs), skipped stage 3, and wrote to `results/ablation_checks/` and
`models/ablation_checks/`, outside the campaign's results tree
(`experiments/ablation_check_NoC_NoE_s46.out`, log
`experiment_log_20260916_223856.txt`). The replicate reproduced the
recorded run bit for bit: all 122 training progress lines identical, the
per-phase training tails identical, the saved network weights identical to
the last bit in all 18 tensors, and the RL evaluation identical episode for
episode at 75.09%. The training is deterministic under the cell's seed on
this GPU, and a suspend does not enter into it: the process state is held
in memory and the run has no dependence on wall-clock time. The recorded
cell stands and the replicate is kept only as evidence. The same holds for
`Full_s45`, which the journal shows was suspended for five minutes in its
phase 4 (17:07 to 17:11 on 2026-09-15, not by the idle timer) and which
scored 94.06%.

Determinism also means a rerun of any cell with its seed is a
reproduction, not an independent draw. The between-seed variance in the
table is the only variance the design measures.

All twenty cells are done. Per-seed RL held-out, in the order NoC_NoE,
OnlyC, OnlyE, Full: s42 87.76, 89.56, 94.17, 93.96; s43 90.23, 89.31,
94.26, 93.88; s44 82.24, 86.10, 93.83, 94.18; s45 87.51, 76.33, 93.95,
94.06; s46 75.09, 89.33, 93.70, 94.15. The two expert-seeded columns sit
within 93.70% to 94.26% across all seeds; the two unseeded columns range
from 75.09% to 90.23% and each holds one weakly converged seed. The
aggregation script and `RESULTS.md` followed on 2026-09-17, see "After
the runs" below.

**What exists on disk.**

- `config/ablation_config/`: all twenty cells, tracked in git. The
  orchestrator's `--dry-run` validated all four files of every cell.
- `data/ablation/s42/`: all four files, `curriculum.h5`, `flat.h5`,
  `expert.h5` and `omniscient_500000.pkl`.
- `data/ablation/s43/`: all four files.
- `data/ablation/s44/`: all four files.
- `data/ablation/s45/`: all four files.
- `data/ablation/s46/`: all four files.
- `results/ablation/` and `models/ablation/`: one folder per finished cell,
  listed in the progress table above.
- `config/ablation_checks/NoC_NoE_s46_rerun/` (tracked) with its output
  under `results/ablation_checks/` and `models/ablation_checks/`: the
  suspend check described above; not a campaign cell and not read by the
  aggregation.
- The shared files `data/ldpc_64x128_main_dataset.h5`,
  `data/ldpc_64x128_expert_dataset.h5` and `data/64x128_omniscient_500000.pkl`
  predate the correction-radius fix and are refused by the validator. No
  ablation cell reads them. They only matter for the curated configs under
  `config/experiments_config/` and are rebuilt with:

  ```bash
  rm data/ldpc_64x128_main_dataset.h5 data/ldpc_64x128_expert_dataset.h5 data/64x128_omniscient_500000.pkl
  PYTHONPATH=. python scripts/run_multi_matrix_experiments.py \
      --config-folder config/experiments_config --folders 64x128_Full --skip-stages 4
  ```

**Starting a sitting** (kept for reruns; no undone pair remains). From the
project root on `main`, with the GPU free, launch the pair with the
detached command above, changing the cell names and the batch number. The
orchestrator writes `experiments/experiment_log_<timestamp>.txt`; each
cell's own files land in `results/ablation/<cell>/` and
`models/ablation/<cell>/`. When the pair finishes, apply the section 6
checks and fill in its row. Running everything remaining in one go is
`--all-folders`; cells whose results folder already exists must be excluded
by hand, as the orchestrator does not skip them.

**After an interruption.** A cell that finished has three evaluation files
and six training files in its results folder. Delete the results folder of
any cell that did not finish and rerun it with `--folders <cell>`. The
per-seed data files are reused, not rebuilt, as long as they match their
config.

**After the runs.** The checks of section 6 passed for every seed.
`scripts/aggregate_ablation.py` implements section 7 and wrote
`docs/ablation/RESULTS.md` on 2026-09-17 from the twenty results folders'
evaluation files and the ten dataset files; it reruns in about four minutes
with `PYTHONPATH=. python scripts/aggregate_ablation.py` and regenerates the
document in full, so `RESULTS.md` is never edited by hand. The evaluation
files are gzipped JSON with one `episodes` list; each episode carries
`terminal_reason` (success is any value starting with `SUCCESS`), `steps`,
`error_info` and `phase`. Two conventions the script fixes beyond section
7: the frame error rate counts a final word that differs from the
transmitted one, reconstructed from the flips, and so sits slightly below
100 minus the success rate; and the overlap figure is also given by error
model and for the unstructured draws alone, because burst and systematic
patterns have so few placements that every held-out one also occurs in
training.

Headline, from `RESULTS.md`: expert seeding is worth 7.9 ± 7.1 points with
the curriculum and 9.4 ± 7.2 without it, both intervals clear of zero and
the effect growing with error weight; the curriculum is worth 0.1 ± 0.4
points with expert seeding and 1.6 ± 11.4 without it, neither
distinguishable from zero. The two unseeded conditions vary from 75% to
90% across seeds, the two seeded ones stay within 93.7% to 94.3%. What
remains for the paper is to replace Table 6 and its paragraph with these
tables and to retire the single-seed `64x128_OnlyC`, `64x128_OnlyE` and
`64x128_NoC_NoE` folders.

