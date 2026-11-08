"""
Size-invariant epsilon schedules for curriculum training.

The trainer consumes `epsilon_decay` as one linear per-update decrement per
curriculum phase. Those decrements are tied to the number of updates a phase
performs, so they silently stop meaning the same thing whenever the dataset
size or the curriculum weights change.

`epsilon_decay: "auto_episodes"` derives the schedule instead from an intent
that does not depend on size: "reach `epsilon_min` after `anneal_fraction` of
the phase, then exploit for the rest of it". Epsilon is set directly from the
episodes completed within a phase. Episode counts are known exactly and do not
shift during training, so the floor lands where asked. The trainer applies it
once per episode and the per-update decrement is zeroed.

Every function here is pure. Nothing imports the trainer, the agent, or torch.
"""
import math
from typing import Any, Callable, Dict, List, Sequence, Union

# Set epsilon from the fraction of the phase completed, counted in episodes.
AUTO_EPISODES = "auto_episodes"

# Used when the anneal fraction is not given.
DEFAULT_ANNEAL_FRACTION = 0.65


def is_auto_episodes(value: Any) -> bool:
    """True when a config value requests episode-progress scheduling."""
    return isinstance(value, str) and value.strip().lower() == AUTO_EPISODES


def as_per_phase(value: Union[Any, Sequence[Any]], num_phases: int,
                 name: str, cast: Callable[[Any], Any] = float,
                 fit: bool = False) -> List[Any]:
    """
    Broadcast a scalar to one value per phase, or validate a per-phase sequence.

    Args:
        value: Scalar applied to every phase, or one value per phase.
        num_phases: Number of curriculum phases.
        name: Field name, used in error messages.
        cast: Applied to each value. `str` for the decay-type field.
        fit: Reshape a wrong-length sequence instead of rejecting it, by
            truncating or by repeating its last value. The trainer sets this
            because it discovers the phase count from the dataset and has
            always tolerated a config written for a different curriculum. Use
            `needs_fitting` to report that a value was reshaped.

    Returns:
        List of `num_phases` values.

    Raises:
        ValueError: If a sequence has the wrong length and `fit` is False, or
            if it is empty.
    """
    if isinstance(value, (list, tuple)):
        if len(value) == num_phases:
            return [cast(v) for v in value]
        if not fit:
            raise ValueError(
                f"{name} has {len(value)} values but the curriculum has "
                f"{num_phases} phases; give one value per phase or a scalar."
            )
        if not value:
            raise ValueError(f"{name} is an empty list")
        fitted = list(value[:num_phases])
        fitted.extend([value[-1]] * (num_phases - len(fitted)))
        return [cast(v) for v in fitted]
    return [cast(value)] * num_phases


def needs_fitting(value: Any, num_phases: int) -> bool:
    """
    True when `fit` would have to reshape this value to span `num_phases`.

    Lets a caller report the adjustment once per field without this module
    printing anything of its own. Scalars broadcast rather than reshape, so
    only an explicit sequence of the wrong length counts.
    """
    rows = value if is_per_epoch(value) else [value]
    return any(isinstance(row, (list, tuple)) and len(row) != num_phases
               for row in rows)


def is_per_epoch(value: Any) -> bool:
    """
    True when a config value is written as one row per epoch.

    Nesting depth is what separates the two readings: `[1.0, 0.8]` is one value
    per phase, `[[1.0, 0.8], [0.3, 0.2]]` is one such row per epoch. Nothing
    else about the value matters, so no config key had to change to carry the
    epoch axis.
    """
    return (isinstance(value, (list, tuple)) and len(value) > 0
            and isinstance(value[0], (list, tuple)))


def first_value(value: Any, default: Any) -> Any:
    """
    The scalar an epsilon field opens on, whatever depth it is written at.

    A field may be a scalar, one value per phase, or one such row per epoch, and
    callers that need a single opening value want epoch one's phase one in every
    case. Descends through leading elements until it reaches a non-sequence,
    falling back to `default` when a list along the way is empty.

    Args:
        value: Scalar, per-phase sequence, or per-epoch sequence of those.
        default: Returned when the value bottoms out in an empty list.

    Returns:
        The first scalar, or `default`.
    """
    while isinstance(value, (list, tuple)):
        if not value:
            return default
        value = value[0]
    return value


def as_per_epoch(value: Union[Any, Sequence[Any]], num_epochs: int,
                 num_phases: int, name: str,
                 cast: Callable[[Any], Any] = float,
                 fit: bool = False) -> List[List[Any]]:
    """
    Normalise an epsilon field to one row per epoch, each one value per phase.

    A flat value keeps its existing meaning and that single row is used by
    every epoch. A nested value gives one row per epoch, and when it supplies
    fewer rows than there are epochs the last row covers the remainder. More
    rows than epochs is rejected rather than truncated, because it almost
    always means the config and `num_epochs` disagree.

    Args:
        value: Scalar, one value per phase, or one such row per epoch.
        num_epochs: Number of training epochs.
        num_phases: Number of curriculum phases.
        name: Field name, used in error messages.
        cast: Applied to each value. `str` for the decay-type field.
        fit: Reshape wrong-length rows rather than rejecting them. See
            `as_per_phase`.

    Returns:
        `num_epochs` rows of `num_phases` values.

    Raises:
        ValueError: If a row has the wrong length, or more rows are given than
            there are epochs.
    """
    if num_epochs < 1:
        raise ValueError(f"num_epochs must be >= 1, got {num_epochs}")

    if not is_per_epoch(value):
        row = as_per_phase(value, num_phases, name, cast, fit)
        return [list(row) for _ in range(num_epochs)]

    if len(value) > num_epochs:
        raise ValueError(
            f"{name} gives {len(value)} epoch schedules but training runs "
            f"{num_epochs} epochs; give at most one row per epoch."
        )
    rows = [as_per_phase(row, num_phases, f"{name} epoch {i + 1}", cast, fit)
            for i, row in enumerate(value)]
    rows.extend(list(rows[-1]) for _ in range(num_epochs - len(rows)))
    return rows


def phase_train_episodes(phase_sizes: Sequence[int],
                         test_size: float) -> List[int]:
    """
    Training episodes per phase, matching the generator's per-phase split.

    RobustDataGenerator.create_split truncates each phase independently with
    int(size * (1 - test_size)), so this mirrors that rather than splitting the
    dataset total.

    Args:
        phase_sizes: Total samples per phase.
        test_size: Held-out fraction, in [0, 1).

    Returns:
        Training episode count per phase.
    """
    if not 0.0 <= test_size < 1.0:
        raise ValueError(f"test_size must be in [0, 1), got {test_size}")
    return [int(size * (1.0 - test_size)) for size in phase_sizes]


def episode_epsilon(start: float, floor: float, anneal_fraction: float,
                    phase_episodes: int, index: int) -> float:
    """
    Epsilon for one episode, from how far into its phase that episode sits.

    Falls linearly from `start` to `floor` over the first
    `anneal_fraction * phase_episodes` episodes, then holds `floor` for the
    rest. An index beyond the phase, as a second epoch produces, also holds
    the floor.

    Args:
        start: Epsilon at the beginning of the phase.
        floor: Epsilon held once annealing completes.
        anneal_fraction: Share of the phase spent annealing, in (0, 1].
        phase_episodes: Training episodes in this phase.
        index: Zero-based position of this episode within the phase.

    Returns:
        Epsilon for the episode, always within [floor, start].
    """
    span = anneal_fraction * phase_episodes
    if span <= 0 or index >= span:
        return floor
    progress = max(0.0, index / span)
    return start - (start - floor) * progress


class EpisodeProgressSchedule:
    """
    Per-episode epsilon keyed on progress through each curriculum phase.

    Holds a start, floor and anneal fraction for every (epoch, phase) pair, plus
    one episode count per phase, and is called with an epoch, a phase and that
    episode's index within the phase. Every epoch runs the phase from its own
    index zero, so what a later epoch does is set by its own row rather than by
    how far a shared counter had already advanced. An epoch whose start equals
    its floor therefore holds the floor throughout. Nothing here depends on
    how long episodes turn out to be.
    """

    def __init__(self, epsilon_start: Sequence[Sequence[float]],
                 epsilon_min: Sequence[Sequence[float]],
                 anneal_fraction: Sequence[Sequence[float]],
                 phase_episodes: Sequence[int]):
        epochs = {len(epsilon_start), len(epsilon_min), len(anneal_fraction)}
        if len(epochs) != 1:
            raise ValueError(
                f"schedule inputs disagree on epoch count: "
                f"start={len(epsilon_start)}, min={len(epsilon_min)}, "
                f"fraction={len(anneal_fraction)}"
            )
        for n in phase_episodes:
            if n <= 0:
                raise ValueError("every phase must have training episodes")

        for e, (starts, floors, fractions) in enumerate(
                zip(epsilon_start, epsilon_min, anneal_fraction), 1):
            lengths = {len(starts), len(floors), len(fractions),
                       len(phase_episodes)}
            if len(lengths) != 1:
                raise ValueError(
                    f"epoch {e} inputs disagree on phase count: "
                    f"start={len(starts)}, min={len(floors)}, "
                    f"fraction={len(fractions)}, episodes={len(phase_episodes)}"
                )
            for i, (s, f, frac) in enumerate(zip(starts, floors, fractions)):
                phase = i + 1
                if s < f:
                    raise ValueError(
                        f"epsilon_start ({s}) is below epsilon_min ({f}) "
                        f"for epoch {e} phase {phase}"
                    )
                if not 0.0 < frac <= 1.0:
                    raise ValueError(
                        f"anneal_fraction for epoch {e} phase {phase} must be "
                        f"in (0, 1], got {frac}"
                    )

        self.epsilon_start = [list(row) for row in epsilon_start]
        self.epsilon_min = [list(row) for row in epsilon_min]
        self.anneal_fraction = [list(row) for row in anneal_fraction]
        self.phase_episodes = list(phase_episodes)

    @property
    def num_phases(self) -> int:
        return len(self.phase_episodes)

    @property
    def num_epochs(self) -> int:
        return len(self.epsilon_start)

    def __call__(self, epoch: int, phase: int, index: int) -> float:
        """
        Epsilon for episode `index` of `phase` in `epoch`, both 1-based.

        Epochs and phases outside the configured range clamp to the nearest one,
        matching how the trainer clamps its own per-phase lookups. Clamping the
        epoch is what makes the reuse-last rule hold even if more epochs run
        than the config wrote rows for.
        """
        e = max(0, min(epoch - 1, self.num_epochs - 1))
        i = max(0, min(phase - 1, self.num_phases - 1))
        return episode_epsilon(self.epsilon_start[e][i], self.epsilon_min[e][i],
                               self.anneal_fraction[e][i],
                               self.phase_episodes[i], index)


def build_episode_schedule(training_config: Dict[str, Any],
                           phase_sizes: Sequence[int],
                           test_size: float,
                           num_epochs: int = 1) -> Dict[str, Any]:
    """
    Build an EpisodeProgressSchedule from an experiment's training config.

    Reads `epsilon_start`, `epsilon_min` and the optional
    `epsilon_anneal_fraction`. Each accepts a scalar, one value per phase, or
    one such row per epoch with the last row covering any remaining epochs. No
    episode-length input is needed or accepted.

    Args:
        training_config: The experiment's `training` block. Not mutated.
        phase_sizes: Total samples per curriculum phase.
        test_size: Held-out fraction used to build the split.
        num_epochs: Number of training epochs the schedule must cover.

    Returns:
        A dict holding the `schedule` plus the resolved inputs, for logging and
        for the saved configuration.
    """
    num_phases = len(phase_sizes)
    if num_phases == 0:
        raise ValueError("cannot build a schedule with no phases")
    if 'epsilon_start' not in training_config or 'epsilon_min' not in training_config:
        raise ValueError(
            "epsilon_decay: 'auto_episodes' requires both epsilon_start "
            "and epsilon_min"
        )

    start = as_per_epoch(training_config['epsilon_start'], num_epochs,
                         num_phases, 'epsilon_start')
    floor = as_per_epoch(training_config['epsilon_min'], num_epochs,
                         num_phases, 'epsilon_min')
    fraction = as_per_epoch(
        training_config.get('epsilon_anneal_fraction', DEFAULT_ANNEAL_FRACTION),
        num_epochs, num_phases, 'epsilon_anneal_fraction')
    episodes = phase_train_episodes(phase_sizes, test_size)

    return {
        'schedule': EpisodeProgressSchedule(start, floor, fraction, episodes),
        'train_episodes': episodes,
        'anneal_fraction': fraction,
        'epsilon_start': start,
        'epsilon_min': floor,
    }


def _epoch_groups(rows: Sequence[Sequence[Any]]) -> List[List[int]]:
    """
    Group consecutive identical epoch rows, returning 1-based epoch numbers.

    Most runs give every epoch the same schedule, so the log should say that
    once rather than repeat an identical table per epoch.
    """
    groups: List[List[int]] = []
    for epoch, row in enumerate(rows, 1):
        if groups and rows[groups[-1][0] - 1] == row:
            groups[-1].append(epoch)
        else:
            groups.append([epoch])
    return groups


def _epoch_label(epochs: Sequence[int], total: int) -> str:
    """Header describing which epochs a table applies to."""
    if len(epochs) == total:
        return "all epochs" if total > 1 else "epoch 1"
    if len(epochs) == 1:
        return f"epoch {epochs[0]}"
    if epochs[-1] == total:
        return f"epochs {epochs[0]} onward"
    return f"epochs {epochs[0]} to {epochs[-1]}"


def format_episode_summary(summary: Dict[str, Any]) -> str:
    """Render an episode-progress schedule as a table for the training log."""
    starts = summary['epsilon_start']
    floors = summary['epsilon_min']
    fractions = summary['anneal_fraction']
    episodes = summary['train_episodes']
    total = len(starts)

    lines = [
        "Resolved epsilon_decay: 'auto_episodes' -> per-episode progress "
        "schedule",
    ]
    combined = [list(zip(s, f, frac))
                for s, f, frac in zip(starts, floors, fractions)]
    for group in _epoch_groups(combined):
        e = group[0] - 1
        lines.append(f"   {_epoch_label(group, total)}:")
        lines.append(
            f"   {'phase':>5} {'episodes':>9} {'start':>7} {'floor':>7} "
            f"{'anneal':>7} {'floor at ep':>12}"
        )
        for i, count in enumerate(episodes):
            start, floor, fraction = starts[e][i], floors[e][i], fractions[e][i]
            reached = 1 if start <= floor else math.ceil(fraction * count)
            lines.append(
                f"   {i + 1:>5} {count:>9,} {start:>7.2f} {floor:>7.2f} "
                f"{fraction:>7.2f} {reached:>12,}"
            )
    return "\n".join(lines)

