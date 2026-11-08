"""Checks on how evaluation results reach disk.

Run from the project root:

    PYTHONPATH=. python tests/verify_progressive_saver.py

Three properties, each of which failed before:

  streaming  - a collector's episode limit bounds memory. Flushed episodes are
               written through to the file instead of being moved into the saver
               and held there until the agent finishes.
  isolation  - one agent's episodes never appear in another agent's file, even
               when the first agent flushes and then fails.
  atomicity  - a phase's file appears under its final name only once complete, so
               an interrupted run leaves nothing that looks finished, and the
               in-memory copies are dropped only after that.

Exits nonzero on the first failed assertion.
"""
import gzip
import json
import sys
import tempfile
from pathlib import Path

from src.evaluation.metrics_callbacks import EpisodeDataCollector
from src.evaluation.progressive_saver import EvaluationProgressiveSaver


def make_collector(limit):
    return EpisodeDataCollector(max_episodes=limit, memory_check_frequency=1)


def feed(collector, tag, count, start=0):
    for index in range(start, start + count):
        collector.on_episode_begin({'episode_id': index, 'agent_type': tag})
        collector.on_episode_end({'terminal_reason': 'SUCCESS_STOP'})


def read_episodes(path):
    with gzip.open(path, 'rt') as handle:
        return json.load(handle)['episodes']


def check_streaming(tmp):
    saver = EvaluationProgressiveSaver('run', output_dir=str(tmp))
    collector = make_collector(limit=2)
    saver.start_agent_evaluation('rl', {'episodes': collector})

    high_water = 0
    for index in range(7):
        feed(collector, 'rl', 1, start=index)
        high_water = max(high_water, len(collector.episodes))

    assert high_water <= 2, (
        f"collector held {high_water} episodes at once against a limit of 2")
    assert saver._episodes_written >= 6, (
        f"only {saver._episodes_written} episodes reached the file; flushes are "
        f"being retained in memory instead of written")

    path = saver.end_phase()
    episodes = read_episodes(path)
    assert len(episodes) == 7, f"{len(episodes)} episodes in the file, expected 7"
    assert [e['agent_type'] for e in episodes] == ['rl'] * 7
    assert collector.episodes == [], "collector was not cleared after a durable write"
    print(f"  streaming ok: 7 episodes saved, never more than {high_water} "
          f"in memory at once against a limit of 2")


def check_isolation(tmp):
    saver = EvaluationProgressiveSaver('run', output_dir=str(tmp))

    # Agent A flushes, then fails before its phase can end.
    collector = make_collector(limit=2)
    saver.start_agent_evaluation('a', {'episodes': collector})
    feed(collector, 'a', 5)
    saver.abort_phase()
    collector.reset()

    # Agent B runs to completion.
    saver.start_agent_evaluation('b', {'episodes': collector})
    feed(collector, 'b', 3)
    path = saver.end_phase()

    episodes = read_episodes(path)
    tags = {e['agent_type'] for e in episodes}
    assert tags == {'b'}, f"agent b's file contains {sorted(tags)}"
    assert len(episodes) == 3, f"{len(episodes)} episodes in agent b's file, expected 3"
    assert not (tmp / 'run_agent_a.json.gz').exists(), (
        "the failed agent left a file that looks complete")
    print("  isolation ok: the failed agent left no file and no episodes behind")


def check_atomicity(tmp):
    saver = EvaluationProgressiveSaver('run', output_dir=str(tmp))
    collector = make_collector(limit=2)
    saver.start_agent_evaluation('c', {'episodes': collector})
    feed(collector, 'c', 5)

    final = tmp / 'run_agent_c.json.gz'
    partial = tmp / 'run_agent_c.json.gz.partial'
    assert not final.exists(), "the final name exists while the phase is still open"
    assert partial.exists(), "no partial file is being written"
    assert collector.episodes, "collector was cleared before the write completed"

    saver.end_phase()
    assert final.exists(), "the final file was not created"
    assert not partial.exists(), "the partial file was left behind"
    assert len(read_episodes(final)) == 5
    print("  atomicity ok: final name appears only on completion")


def main():
    with tempfile.TemporaryDirectory() as name:
        tmp = Path(name)
        print("Progressive saver")
        check_streaming(tmp / 'a')
        check_isolation(tmp / 'b')
        check_atomicity(tmp / 'c')
    print("all checks passed")


if __name__ == '__main__':
    main()
