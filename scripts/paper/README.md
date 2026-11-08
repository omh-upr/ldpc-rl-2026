# Paper generators

These scripts write every table, figure and number that the paper "Replay
Seeding and Training Budget in DQN-Based LDPC Bit-Flipping Decoding" quotes.
They read only the records under `docs/ablation/`, the ablation configurations
and the code's matrices. They never read raw training output, so they run on a
fresh checkout in seconds and need no GPU.

## Regenerating the paper's numbers

From the project root:

```bash
PYTHONPATH=. python scripts/paper/make_tables.py      # tables/*.md
PYTHONPATH=. python scripts/paper/make_variables.py   # _variables.yml, the numbers quoted in the text
PYTHONPATH=. python scripts/paper/make_figures.py     # figures/*.pdf and *.png
```

Output goes to `results/paper/`, or to the folder given with `--out DIR`.
Tables are Quarto pipe tables with their captions, and `_variables.yml` maps
each quoted number to its value, formatted as the paper prints it.
`paperlib.py` holds what the three share: the record loaders, the condition
names below, the confidence intervals and the number formatting.

## The records they read

Each record is generated in full by the script named next to it and is never
edited by hand. Regenerating a record needs the raw results of the runs, which
are not part of the repository. The paper's numbers need only the records.

| Record under `docs/ablation/` | Written by | Holds |
|---|---|---|
| `RESULTS.json` | `scripts/aggregate_ablation.py` | Held-out task success per cell, seed and epoch; the baselines; the contrasts |
| `DETAILS.json`, `learning_curves.csv` | `scripts/analyze_ablation_details.py` | Seen and unseen words, the word-by-word contingency, training curves |
| `GALLAGER_B_SELECTION.json` | `scripts/select_gallager_b_thresholds.py` | The Gallager-B flip thresholds and how they were chosen |
| `benchmark/decoder_cost_*.json` | `scripts/benchmark_decoders.py` | Latency, operation counts and throughput of the final policy and the baselines |
| `cascade/cascade_*.json`, `cascade/controls_*.json` | `scripts/cascade_minsum_agent.py`, `scripts/cascade_controls.py` | The agent combined with Min-Sum, and the controls for that combination |
| `DEVELOPMENT_CHAIN.json` | `scripts/summarize_development_chain.py` | The tuning steps that led to the training configuration |
| `WALL_CLOCK.json` | `scripts/record_wall_clock.py` | Wall-clock time of each cell |
| `example_episode.json` | `scripts/extract_example_episode.py` | The decoding episode drawn in the paper's episode figure |

When several benchmark or cascade records exist, the generators take the newest
one that covers the final policy on all five seeds.

## Final-policy weights

The trained networks of the final policy, one per seed, are in the repository
at `models/ablation/NoC_NoE3ep_s<seed>/ldpc_rl_agent_<timestamp>.pth`, where the
cell configurations and `benchmark_decoders.py` expect them. With them, the
final policy's evaluation reruns without training, from the configs under
`config/final_eval/` (`scripts/make_final_eval_configs.py`), and so do the cost
benchmark and the Min-Sum combination, which check their first stages against
those evaluation files (`--config-root config/final_eval`). No other weights are published: every cell retrains from its
configuration, bit for bit on the reference hardware (an RTX 3060 Ti with the
locked package versions); on other GPUs the floating-point results, and so the
scores, may differ slightly.
The evaluation records likewise regenerate from the configurations and are not
published.

## Condition names

The paper names conditions by how the training data are ordered and whether the
replay buffer is seeded with expert demonstrations. The repository, including
`docs/ablation/` and `config/ablation_config/`, uses short IDs instead.
`LABEL` in `paperlib.py` is the authoritative mapping.

| Repository ID | Name in the paper | Epochs |
|---|---|---|
| `NoC_NoE` | Flat, unseeded | 1 |
| `OnlyC` | Curriculum, unseeded | 1 |
| `OnlyE` | Flat, seeded | 1 |
| `Full` | Curriculum, seeded | 1 |
| `NoC_NoE3ep` | Flat, unseeded | 3 |
| `OnlyC3ep` | Curriculum, unseeded | 3 |
| `OnlyE3ep` | Flat, seeded | 3 |
| `Full3ep` | Curriculum, seeded | 3 |
| `Full3epFloor` | Curriculum, seeded, with ε held at its floor in epochs 2 and 3 (a control, seed 42 only) | 3 |

The paper states the training budget beside the name, as one or three epochs
(passes over the training data). The paper's **final policy** is `NoC_NoE3ep`
after its third epoch.

A three-epoch cell's first epoch trains exactly as the matching one-epoch cell,
bit for bit, so its epoch-1 scores equal that cell's scores. The three-epoch
cells are separate runs, not continuations of the one-epoch ones.

Cells are named `<ID>_s<seed>`, for seeds 42 to 46. In `results/ablation/`,
folders ending in `_ep1` or `_ep2` hold evaluations of a three-epoch cell's
earlier epoch checkpoints (configurations in `config/epoch_eval/`), and
`baseline_eval/` holds the Gallager-B scores (configurations in
`config/baseline_eval/`).
