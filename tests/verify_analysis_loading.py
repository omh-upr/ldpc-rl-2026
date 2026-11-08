#!/usr/bin/env python3
"""
Checks how an analysis database is opened.

Two rules are asserted here, both of which were broken before:

  1. Either half of a run's output is enough to load it. A run with
     `run_training: false` scores an existing checkpoint and writes only
     evaluation files, which is how a per-epoch checkpoint gets a held-out
     number. The command line refused such a run until recently and the
     notebook refused it until now.
  2. A database is reused only for the experiment it actually holds. A
     populated database named with `--database-path` used to be reused
     whatever was in it, so the analysis answered with another run's episodes
     at exit code 0. The single-experiment guard does not catch that: a
     database holding one wrong experiment holds exactly one experiment.
  3. A request that produces no analysis exits nonzero, with its diagnostic on
     stderr. `run_experiment_report.py` runs the analysis command as a
     subprocess and judges it by return code, so a failure that exited 0 became
     an analysis result: it went into the report as a section and the report was
     announced as a success.
  4. A failed load closes its database connection. DuckDB holds a file lock
     while a connection is open, and a traceback keeps the frame that holds the
     analyzer alive, so a refused call in a notebook used to block the command
     line from touching the same database.
  5. An interrupted import leaves nothing behind, and a database left partial by
     an older one is refused rather than answered from. DuckDB commits each
     statement on its own, so the episodes used to land before their steps were
     built; the next call saw the experiment present and reused it, and the
     analyses answered from episodes alone.

The last two need the entry points themselves: an exit status is invisible to
anything that calls the loader as a function, which is why the first version of
these tests missed both.

Both entry points go through `src.analysis.experiment_loader`, so testing it
tests the notebook and `scripts/run_analysis.py` together. Result files are
synthesized here rather than taken from `results/`, so this runs anywhere.

Run:
    PYTHONPATH=. python tests/verify_analysis_loading.py
"""
import gc
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from src.analysis import analysis as analysis_module
from src.analysis.experiment_loader import (
    database_experiment_ids, discover_result_files, experiment_name_from_files,
    open_experiment, partial_import)
from src.utils.path_utils import resolve_config_path

N = 8            # codeword length; nothing here decodes, the shape just has to be consistent
SEED = 2024


def an_episode(episode_id, episode_type, agent_type, epoch, phase, rng):
    """One episode in the shape the progressive saver writes."""
    original = rng.integers(0, 2, N).tolist()
    error = [0] * N
    error[episode_id % N] = 1
    noisy = [b ^ e for b, e in zip(original, error)]
    return {
        'episode_id': episode_id,
        'episode_type': episode_type,
        'agent_type': agent_type,
        'epoch': epoch,
        'phase': phase,
        'original_codeword': original,
        'noisy_codeword': noisy,
        'error_info': {'error_model': 'uniform_count',
                       'parameters': {'num_errors': 1},
                       'error_pattern': error},
        'steps': [{'step': 1,
                   'state': {'codeword_weight': sum(noisy), 'syndrome_weight': 1,
                             'flip_weight': 0, 'penalty_sum': 0},
                   'next_state': {'codeword_weight': sum(original), 'syndrome_weight': 0,
                                  'flip_weight': 1, 'penalty_sum': 0},
                   'action': episode_id % N, 'reward': 1.0,
                   'done': True, 'success': True}],
        'terminal_reason': 'SUCCESS_STOP',
    }


def write_run(results_dir, name, experiment_id, training, evaluation):
    """Write one run's result files. Either half may be empty."""
    rng = np.random.default_rng(SEED)
    episode_id = 0

    for phase in training:
        episodes = []
        for _ in range(3):
            episodes.append(an_episode(episode_id, 'train', 'agent', 1, phase, rng))
            episode_id += 1
        path = results_dir / f'{name}_{experiment_id}_training_epoch1_phase{phase}.json.gz'
        with gzip.open(path, 'wt') as f:
            json.dump({'experiment_configuration': {'experiment_name': name,
                                                    'code_id': 'test'},
                       'epoch': 1, 'curriculum_phase': phase,
                       'training_episodes': episodes}, f)

    for agent in evaluation:
        episodes = [an_episode(i, 'eval', agent, None, 1, rng) for i in range(4)]
        path = results_dir / f'{name}_{experiment_id}_evaluation_agent_{agent}.json.gz'
        with gzip.open(path, 'wt') as f:
            json.dump({'phase_id': f'agent_{agent}', 'save_type': 'evaluation',
                       'episodes': episodes}, f)


def episode_counts(analyzer):
    rows = analyzer.con.execute(
        "SELECT episode_type, COUNT(*) FROM episodes GROUP BY 1 ORDER BY 1").fetchall()
    return dict(rows)


def check_both_halves(results_dir, definitions):
    print("1. either half of a run's output is enough to load it")

    both = open_experiment(definitions, results_dir / 'both.db',
                           '20260101_000000', results_dir, log=lambda *_: None)
    counts = episode_counts(both)
    assert counts == {'train': 6, 'eval': 8}, counts

    # The case the notebook refused: evaluation files only, no training files.
    eval_only = open_experiment(definitions, results_dir / 'evalonly.db',
                                '20260202_000000', results_dir, log=lambda *_: None)
    counts = episode_counts(eval_only)
    assert counts == {'eval': 4}, counts
    name = eval_only.con.execute(
        "SELECT experiment_name FROM experiments").fetchone()[0]
    assert name == 'EpochScore', f"name taken from evaluation files, got {name!r}"

    # The mirror case: training files only.
    train_only = open_experiment(definitions, results_dir / 'trainonly.db',
                                 '20260303_000000', results_dir, log=lambda *_: None)
    counts = episode_counts(train_only)
    assert counts == {'train': 3}, counts

    # Neither half is an error, and it names both patterns it looked for.
    try:
        open_experiment(definitions, results_dir / 'missing.db',
                        '19990101_000000', results_dir, log=lambda *_: None)
    except FileNotFoundError as e:
        assert 'training' in str(e) and 'evaluation' in str(e)
    else:
        raise AssertionError("a run with no result files at all must be an error")

    print("   loaded both halves, evaluation only, training only; named the "
          "evaluation-only run from its evaluation files")


def check_database_identity(results_dir, definitions):
    print("2. a database is reused only for the experiment it holds")

    shared = results_dir / 'shared.db'
    first = open_experiment(definitions, shared, '20260101_000000',
                            results_dir, log=lambda *_: None)
    assert database_experiment_ids(first) == ['20260101_000000']
    first.con.close()

    # Asking for a different run against that database used to succeed and
    # answer with the first run's episodes.
    try:
        open_experiment(definitions, shared, '20260202_000000',
                        results_dir, log=lambda *_: None)
    except ValueError as e:
        assert '20260101_000000' in str(e) and '20260202_000000' in str(e), str(e)
    else:
        raise AssertionError("reused a database belonging to another experiment")

    # Refusing must not have destroyed what was there.
    after = open_experiment(definitions, shared, '20260101_000000',
                            results_dir, log=lambda *_: None)
    assert episode_counts(after) == {'train': 6, 'eval': 8}, "refusal damaged the database"

    # A rebuild is not a way around it either: the schema is dropped and
    # recreated, so allowing it would discard the other run.
    try:
        open_experiment(definitions, shared, '20260202_000000', results_dir,
                        rebuild=True, log=lambda *_: None)
    except ValueError:
        pass
    else:
        raise AssertionError("--rebuild-database overwrote another experiment")
    after.con.close()

    # Rebuilding the run the database does hold is fine.
    same = open_experiment(definitions, shared, '20260101_000000', results_dir,
                           rebuild=True, log=lambda *_: None)
    assert episode_counts(same) == {'train': 6, 'eval': 8}
    print("   refused a foreign database with and without --rebuild-database, "
          "left it intact, reloaded its own run")


def check_discovery(results_dir):
    print("3. discovery and naming read from whichever half exists")
    training, evaluation = discover_result_files('20260202_000000', results_dir,
                                                 log=lambda *_: None)
    assert training == [], training
    assert experiment_name_from_files('20260202_000000', training, evaluation) == 'EpochScore'

    training, evaluation = discover_result_files('20260303_000000', results_dir,
                                                 log=lambda *_: None)
    assert evaluation is None, evaluation
    assert experiment_name_from_files('20260303_000000', training, evaluation) == 'TrainOnly'
    print("   names resolved from evaluation files alone and training files alone")



def run_cli(*args):
    """Run the analysis command as a subprocess, as the report generator does."""
    env = dict(os.environ, PYTHONPATH='.')
    return subprocess.run([sys.executable, 'scripts/run_analysis.py', *args],
                          capture_output=True, text=True, env=env)


def check_exit_status(results_dir):
    print("4. a request that produces no analysis exits nonzero")

    shared = results_dir / 'status.db'
    ok = run_cli('--experiment_id', '20260101_000000',
                 '--analysis_id', 'overall_performance_summary',
                 '--database-path', str(shared),
                 '--results-dir', str(results_dir), '--data-only')
    assert ok.returncode == 0, f"a valid run exited {ok.returncode}:\n{ok.stderr}"
    assert '--- Analysis Results ---' in ok.stdout

    listing = run_cli('--list')
    assert listing.returncode == 0, "--list must stay successful"

    for description, argv in (
            ('a foreign database',
             ('--experiment_id', '20260202_000000',
              '--analysis_id', 'overall_performance_summary',
              '--database-path', str(shared), '--results-dir', str(results_dir))),
            ('a foreign database with --rebuild-database',
             ('--experiment_id', '20260202_000000',
              '--analysis_id', 'overall_performance_summary',
              '--database-path', str(shared), '--results-dir', str(results_dir),
              '--rebuild-database')),
            ('an experiment with no result files',
             ('--experiment_id', '19990101_000000',
              '--analysis_id', 'overall_performance_summary',
              '--results-dir', str(results_dir)))):
        result = run_cli(*argv, '--data-only')
        assert result.returncode != 0, f"{description} exited 0"
        assert result.stderr.strip(), f"{description} said nothing on stderr"
        assert '--- Analysis Results ---' not in result.stdout
    print("   valid run and --list are 0; refusals and missing files are nonzero "
          "with their diagnostic on stderr")


def check_failure_releases_lock(results_dir):
    print("5. a failed load closes its database connection")

    def free(database):
        probe = subprocess.run(
            [sys.executable, '-c', f"import duckdb; duckdb.connect(r'{database}')"],
            capture_output=True, text=True)
        return probe.returncode == 0

    definitions = resolve_config_path('analysis_definitions.json')
    shared = results_dir / 'lock.db'
    first = open_experiment(definitions, shared, '20260101_000000',
                            results_dir, log=lambda *_: None)
    first.con.close()

    # A corrupt result file fails from inside the loader rather than at a guard,
    # which is the path most likely to grow new failure modes.
    corrupt = results_dir / 'Corrupt_20270101_000000_evaluation_agent_rl.json.gz'
    corrupt.write_text('not gzip at all')

    held = []
    for description, database, experiment_id in (
            ('a foreign database', shared, '20260202_000000'),
            ('no result files', results_dir / 'missing.db', '19990101_000000'),
            ('a corrupt result file', results_dir / 'corrupt.db', '20270101_000000')):
        try:
            open_experiment(definitions, database, experiment_id, results_dir,
                            log=lambda *_: None)
        except BaseException as e:
            held.append(e)          # keep the traceback alive, as a notebook does
        else:
            raise AssertionError(f"{description} did not fail")
        assert free(database), (
            f"the connection from {description} still holds the file lock")
    del held
    gc.collect()
    corrupt.unlink()
    print("   guard refusals and a failure from inside the loader both release "
          "the lock while the traceback is still held")


def check_report_generator(results_dir):
    print("6. the report generator does not pass an error off as a result")

    template = results_dir / 'template.md'
    template.write_text('# {{experiment_id}}\n\n{{overall_performance_summary}}\n')
    env = dict(os.environ, PYTHONPATH='.')

    def generate(experiment_id):
        return subprocess.run(
            [sys.executable, 'scripts/run_experiment_report.py',
             '--experiment-id', experiment_id, '--template', str(template),
             '--results-dir', str(results_dir)],
            capture_output=True, text=True, env=env)

    failed = generate('19990101_000000')
    assert failed.returncode != 0, "a report of nothing but failures exited 0"
    assert 'Successfully generated report' not in failed.stdout, (
        "a report of nothing but failures announced success")
    report = results_dir / '19990101_000000_report.md'
    assert report.exists(), "the partial report should still be written"
    body = report.read_text()
    assert 'ANALYSIS FAILED' in body, (
        "the error was placed in the report as though it were a result")

    good = generate('20260101_000000')
    assert good.returncode == 0, f"a valid report exited {good.returncode}:\n{good.stderr}"
    assert 'Successfully generated report' in good.stdout
    written = (results_dir / '20260101_000000_report.md').read_text()
    assert 'ANALYSIS FAILED' not in written and 'success_rate' in written
    print("   failure is reported as failure and exits nonzero; a valid report "
          "is unchanged")



def check_import_is_atomic(results_dir, definitions):
    print("7. an interrupted import leaves nothing, and a partial one is refused")

    write_run(results_dir, 'Interrupted', '20280101_000000',
              training=[1], evaluation=['rl'])
    database = results_dir / 'interrupted.db'

    # Interrupt where it actually happens: after the episodes are copied in, while
    # the step rows are still being built. Ctrl-C during a long import lands here.
    real = analysis_module.LDPCAnalysis._write_steps_to_csv

    def interrupt(self, *args, **kwargs):
        raise KeyboardInterrupt("simulated Ctrl-C during the step import")

    analysis_module.LDPCAnalysis._write_steps_to_csv = interrupt
    try:
        open_experiment(definitions, database, '20280101_000000', results_dir,
                        log=lambda *_: None)
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("the interruption did not propagate")
    finally:
        analysis_module.LDPCAnalysis._write_steps_to_csv = real

    import duckdb
    con = duckdb.connect(str(database))
    try:
        rows = con.execute(
            "SELECT (SELECT COUNT(*) FROM episodes), (SELECT COUNT(*) FROM steps), "
            "(SELECT COUNT(*) FROM experiments)").fetchone()
        assert rows == (0, 0, 0), f"the interrupted import committed {rows}"
    except duckdb.CatalogException:
        pass                     # no tables at all is just as good
    finally:
        con.close()

    # A plain retry must now finish the job rather than reuse the wreckage.
    retried = open_experiment(definitions, database, '20280101_000000',
                              results_dir, log=lambda *_: None)
    assert episode_counts(retried) == {'train': 3, 'eval': 4}
    complete = retried.con.execute(
        "SELECT (SELECT COALESCE(SUM(episode_length), 0) FROM episodes), "
        "(SELECT COUNT(*) FROM steps)").fetchone()
    assert complete[0] == complete[1] and complete[1] > 0, complete
    assert partial_import(retried) is None
    retried.con.close()

    # A database damaged before the transaction existed is still out there, so
    # reuse has to check rather than trust. Reproduce that state directly.
    damaged = results_dir / 'damaged.db'
    shutil.copy(database, damaged)
    con = duckdb.connect(str(damaged))
    con.execute("DELETE FROM steps")
    con.close()

    try:
        open_experiment(definitions, damaged, '20280101_000000', results_dir,
                        log=lambda *_: None)
    except ValueError as e:
        assert 'rebuild-database' in str(e), str(e)
    else:
        raise AssertionError("answered from a database whose steps are missing")

    # And the instruction it gives has to work.
    repaired = open_experiment(definitions, damaged, '20280101_000000',
                               results_dir, rebuild=True, log=lambda *_: None)
    assert partial_import(repaired) is None
    repaired.con.close()
    print("   interruption committed nothing, a retry completed it, a database "
          "missing its steps is refused and the rebuild it names repairs it")


def main():
    print("Verifying how an analysis database is opened\n")
    definitions = resolve_config_path('analysis_definitions.json')
    results_dir = Path(tempfile.mkdtemp(prefix='verify_analysis_'))
    try:
        write_run(results_dir, 'BothHalves', '20260101_000000',
                  training=[1, 2], evaluation=['rl', 'greedy'])
        write_run(results_dir, 'EpochScore', '20260202_000000',
                  training=[], evaluation=['rl'])
        write_run(results_dir, 'TrainOnly', '20260303_000000',
                  training=[1], evaluation=[])

        check_both_halves(results_dir, definitions)
        check_database_identity(results_dir, definitions)
        check_discovery(results_dir)
        check_exit_status(results_dir)
        check_failure_releases_lock(results_dir)
        check_report_generator(results_dir)
        check_import_is_atomic(results_dir, definitions)
    finally:
        shutil.rmtree(results_dir, ignore_errors=True)

    print("\nAll checks passed.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
