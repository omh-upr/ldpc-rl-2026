"""
One way to open an experiment's analysis database.

`scripts/run_analysis.py` and `notebooks/complete_analysis.ipynb` both need the
same four steps: find the run's result files, name the run from them, decide
whether a database already on disk can be reused, and load. Each carried its own
copy, and the copies drifted. The command line learned to read a run that only
evaluated, which is how an epoch checkpoint is scored; the notebook still refused
one. This module is the single copy, so the next change reaches both.

It also enforces the rule neither copy had: a database may only be reused for the
experiment it actually holds.
"""
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

from src.analysis.analysis import LDPCAnalysis
from src.utils.file_patterns import ExperimentFiles

EvaluationFiles = Optional[List[str]]


def discover_result_files(experiment_id: str, results_dir: Union[str, Path],
                          log: Callable[[str], None] = print
                          ) -> Tuple[List[str], EvaluationFiles]:
    """
    Find the training and evaluation files a run wrote.

    Either half may be missing and the run still be readable. A run with
    `run_training: false` writes no training files, and one with
    `run_evaluation: false` writes no evaluation files.

    Args:
        experiment_id: Run timestamp.
        results_dir: Directory the run wrote to.
        log: Where to report what was found.

    Returns:
        (training_files, evaluation_files). Training files are a list, possibly
        empty. Evaluation files are a list of agent-separated files, or None.
    """
    results_dir = str(results_dir)

    training_files = ExperimentFiles.find_training_files(experiment_id, results_dir)
    if training_files:
        flush = [f for f in training_files if ExperimentFiles.is_flush_file(f)]
        if flush:
            log(f"Found {len(training_files)} training files "
                f"({len(flush)} flush + {len(training_files) - len(flush)} phase-end):")
        else:
            log(f"Found {len(training_files)} training files:")
        for f in training_files:
            log(f"   - {Path(f).name}")
    else:
        log("No training data found (evaluation-only experiment)")

    found = ExperimentFiles.find_evaluation_files(experiment_id, results_dir)
    if found.format_type == "agent_separated":
        log(f"Found {len(found.files)} agent-separated evaluation files:")
        for agent_name, file_path in zip(found.agent_names, found.files):
            log(f"   - {agent_name}: {Path(file_path).name}")
        evaluation_files = found.files
    else:
        log("No evaluation data found (training-only experiment)")
        evaluation_files = None

    return training_files, evaluation_files


def experiment_name_from_files(experiment_id: str,
                               training_files: Sequence[str],
                               evaluation_files: EvaluationFiles) -> str:
    """
    Run name, read from whichever result files this experiment produced.

    The name is whatever precedes the `_{id}_training` or `_{id}_evaluation`
    marker. Both markers are stable across every variant of a filename: the
    phase, flush and epoch parts of a training name, and the agent part of an
    evaluation name. Reading it from either side is what lets a run that only
    evaluated be named without a training file to parse.

    Args:
        experiment_id: Run timestamp.
        training_files: Discovered training files, possibly empty.
        evaluation_files: Discovered evaluation files, possibly None.

    Returns:
        The experiment name, falling back to the id when no filename yields one.
    """
    for files, marker in ((training_files, 'training'),
                          (evaluation_files, 'evaluation')):
        if not files:
            continue
        stem = Path(files[0]).stem.replace('.json', '')
        name = stem.split(f"_{experiment_id}_{marker}")[0]
        if name:
            return name
    return experiment_id


def database_experiment_ids(analyzer: LDPCAnalysis) -> List[str]:
    """
    Which experiments a database holds episodes for.

    Constructing LDPCAnalysis opens the DuckDB connection, which creates the
    file, so asking whether the path exists always says yes. Ask what is in it
    instead. An empty list means a fresh or failed database, either of which
    should be built rather than reused.
    """
    try:
        return [row[0] for row in analyzer.con.execute(
            "SELECT DISTINCT experiment_id FROM episodes ORDER BY 1").fetchall()]
    except Exception:
        return []



def partial_import(analyzer: LDPCAnalysis) -> Optional[str]:
    """
    Whether a database holds episodes whose steps never arrived.

    The import is one transaction now, so this cannot be produced any more. It
    can still be found: any database written before that, interrupted between
    the episode and step loads, holds the episodes and none of their steps. The
    analyses do not notice. The bit error rate summary reconstructs corrective
    flips from step rows and, with none stored, reports that the decoder fixed
    nothing.

    Every episode records its own length, and that equals the number of steps
    stored for it, so two aggregates settle it without a join. On a database of
    5.9 million steps the pair returns promptly.

    Returns:
        None when the database is whole, otherwise a description of the shortfall.
    """
    try:
        declared, stored = analyzer.con.execute(
            "SELECT (SELECT COALESCE(SUM(episode_length), 0) FROM episodes), "
            "(SELECT COUNT(*) FROM steps)").fetchone()
    except Exception:
        return None      # no tables yet; the caller is about to build them

    if declared == stored:
        return None
    return (f"its episodes declare {declared:,} steps but only {stored:,} are "
            f"stored")


def open_experiment(definitions_path: Union[str, Path],
                    database_path: Optional[Union[str, Path]],
                    experiment_id: str,
                    results_dir: Union[str, Path],
                    rebuild: bool = False,
                    log: Callable[[str], None] = print) -> LDPCAnalysis:
    """
    Open an analysis database for one experiment, loading it if needed.

    Reuses the database on disk when it already holds this experiment, and
    refuses one holding anything else. Without that check a populated database
    passed with `--database-path` is reused whatever it contains, and the
    analysis answers with another run's episodes at exit code 0. The
    single-experiment guard inside `run_analysis` does not catch it, because a
    database holding exactly one wrong experiment is still a database holding
    exactly one experiment.

    Args:
        definitions_path: `config/analysis_definitions.json`.
        database_path: Database file, or None for an in-memory one.
        experiment_id: Run timestamp to load.
        results_dir: Directory holding the run's result files.
        rebuild: Discard and reload even when the database already holds this run.
        log: Where to report progress.

    Returns:
        An LDPCAnalysis whose database holds this experiment. On any failure the
        connection is closed before the exception leaves, so a refused call holds
        no lock on the database file.

    Raises:
        ValueError: The database holds a different experiment, or was left
            incomplete by an interrupted import. Nothing is overwritten; delete
            it, choose another path, or pass rebuild to load it again.
        FileNotFoundError: No result files for this experiment.
    """
    analyzer = LDPCAnalysis(str(definitions_path),
                            str(database_path) if database_path else None)

    # Everything below can raise: the guards here, and the loader itself on a
    # result file that will not read. DuckDB holds a file lock for as long as the
    # connection is open, and a traceback keeps this frame and so this analyzer
    # alive, which in a notebook is until the next exception. A refused call would
    # then block the command line from touching the same database. On success the
    # connection is the caller's to keep.
    try:
        present = database_experiment_ids(analyzer)
        if present and set(present) != {experiment_id}:
            raise ValueError(
                f"{database_path} holds {', '.join(present)}, not {experiment_id}. "
                f"Reusing it would answer with another run's episodes, and rebuilding "
                f"it would discard them. Delete that file, or point --database-path "
                f"somewhere else, or drop the flag to use the default one database "
                f"per experiment.")

        if present and not rebuild:
            shortfall = partial_import(analyzer)
            if shortfall is not None:
                raise ValueError(
                    f"{database_path} was left incomplete by an interrupted "
                    f"import: {shortfall}. Analyses would read the episodes and "
                    f"answer without the steps, which is wrong rather than empty. "
                    f"Load it again with --rebuild-database.")
            log(f"Using existing database: {database_path}")
            return analyzer

        training_files, evaluation_files = discover_result_files(
            experiment_id, results_dir, log=log)

        # Either half is enough. A run with `run_training: false` evaluates an
        # existing checkpoint and writes no training files at all, and requiring
        # them made that supported pipeline configuration unreadable. The loader
        # already handles an empty training set, and the training-side analyses
        # simply return nothing.
        if not training_files and not evaluation_files:
            raise FileNotFoundError(
                f"Could not find any result files for experiment {experiment_id}\n"
                f"   Tried training:   *{experiment_id}_training_epoch*_phase*.json.gz\n"
                f"   Tried evaluation: *{experiment_id}_evaluation_agent_*.json.gz\n"
                f"   In directory: {results_dir}")
        if not training_files:
            log("Loading evaluation data only; analyses over training "
                "episodes will return no rows.")

        log("Creating database schema...")
        analyzer.create_database_schema()

        log("Loading experiment data into database...")
        analyzer.load_experiment_data(
            training_files,
            evaluation_files,
            experiment_id,
            experiment_name_from_files(experiment_id, training_files, evaluation_files))

        log("\nDatabase Statistics:")
        log(analyzer.get_database_stats().to_string(index=False))
        return analyzer
    except BaseException:
        try:
            analyzer.con.close()
        except Exception:
            pass
        raise
