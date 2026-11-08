"""Equivalence checks for the two Min-Sum implementations.

`MinSumDecoder` is edge-vectorized; `ReferenceMinSumDecoder` walks the Tanner graph
in Python and is the specification. This asserts they agree on everything that can
reach a result.

Run from the project root:

    PYTHONPATH=. python tests/verify_minsum_equivalence.py          # sampled, ~1 min
    PYTHONPATH=. python tests/verify_minsum_equivalence.py --full   # all 105,000, ~14 min

Checked at four levels:

  decoder   - identical decoded codewords and identical converged flags, on random
              error patterns for every bundled code at the shipped damping.
  settings  - across damping 0.0 to 1.0 and on matrices with isolated variables or
              degree-one checks, the two must agree whenever either produces a valid
              codeword. Disagreement between two failed decodes is recorded, not
              failed: with damping off, a failed decode leaves reliabilities on the
              decision threshold and accumulation order can put a bit on the other
              side of it. Damping 1.0 is included on the degree-one check matrix,
              where the blend used to reach 0.0 times infinity and fill the message
              array with NaN, so that sweep runs with RuntimeWarning promoted to an
              error.
  boundary  - the matrix where damping 0.0 breaks the claim above: one implementation
              decodes the word, the other does not. Nothing is asserted at 0.0, which
              is the point; every non-zero damping must agree on it.
  episode   - the real MinSumExpertAgent driven through the real environment over the
              64x128 test split: identical success flag, step count and terminal
              reason, episode by episode.

The order in which an episode's flips are applied is reported but not asserted. The
agent sorts flips by reliability magnitude, the two implementations sum a variable's
incoming messages in a different order, and that changes the last bits of a float, so
near-equal reliabilities can swap. It moves no outcome: the same bits are flipped, so
the final codeword, the flip count and therefore success are unchanged.

Exits nonzero on the first failed assertion.
"""
import argparse
import collections
import sys
import time
import warnings

import numpy as np

from src.advanced.min_sum_core import MinSumDecoder, ReferenceMinSumDecoder
from src.core.baseline_agents import MinSumExpertAgent
from src.core.environment import LDPCEnvironment
from src.core.ldpc_code import LDPC_Code
from src.data.robust_data_generator import RobustDataGenerator
from src.utils.path_utils import resolve_project_path

# Matches what MinSumExpertAgent constructs.
DECODER_KWARGS = dict(max_iterations=20, damping_factor=0.8,
                      convergence_threshold=1e-4, max_magnitude_threshold=50.0)


def random_error_cases(code, count, max_errors, seed=0):
    """Error patterns on the all-zero codeword, which is valid for any linear code."""
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(count):
        word = np.zeros(code.n, dtype=np.int8)
        for position in rng.choice(code.n, size=rng.integers(1, max_errors + 1), replace=False):
            word[position] ^= 1
        syndrome = ((code.H @ word) % 2).astype(np.int8)
        if syndrome.sum() > 0:
            cases.append((word, syndrome))
    return cases


def check_decoder_level(code_id, max_errors, count=400):
    code = LDPC_Code.from_default(code_id=code_id)
    cases = random_error_cases(code, count, max_errors)
    fast = MinSumDecoder(code.H, **DECODER_KWARGS)
    reference = ReferenceMinSumDecoder(code.H, **DECODER_KWARGS)

    largest_gap = 0.0
    fast_seconds = reference_seconds = 0.0
    for word, syndrome in cases:
        start = time.perf_counter()
        ref_rel, ref_word, ref_converged = reference.decode(word, syndrome)
        reference_seconds += time.perf_counter() - start

        start = time.perf_counter()
        rel, decoded, converged = fast.decode(word, syndrome)
        fast_seconds += time.perf_counter() - start

        assert np.array_equal(decoded, ref_word), (
            f"{code_id}: decoded codewords differ")
        assert converged == ref_converged, (
            f"{code_id}: converged flags differ")
        largest_gap = max(largest_gap, float(np.max(np.abs(rel - ref_rel))))

    speedup = reference_seconds / fast_seconds if fast_seconds else float('nan')
    print(f"  {code_id:<9} {len(cases):>4} cases  decoded identical  "
          f"max reliability gap {largest_gap:.1e}  {speedup:>5.1f}x faster")


# Matrices the bundled codes do not exercise: a variable connected to nothing, and a
# check with a single edge. Both are degenerate but legal, and both once crashed or
# diverged in the vectorized implementation.
DEGENERATE_MATRICES = {
    'trailing isolated variable': [[1, 1, 0]],
    'leading isolated variable': [[0, 1, 1]],
    'middle isolated variable': [[1, 0, 1], [1, 0, 1]],
    'several isolated variables': [[1, 0, 1, 0, 0], [0, 0, 1, 0, 1]],
    'degree-one check': [[1, 0, 0], [1, 1, 1]],
}

# The matrix that bounds the equivalence claim. Full row rank, no isolated
# variable, and at damping 0.0 the reference decodes the word below to a valid
# codeword while the vectorized implementation returns a different word with a
# non-zero syndrome. Found by probing random small matrices; kept here because it
# is the counterexample to the guarantee this file used to assert without
# qualification. Every non-zero damping value agrees on it.
ZERO_DAMPING_COUNTEREXAMPLE = np.array([
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1],
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
], dtype=np.int8)
ZERO_DAMPING_WORD = np.array([1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1], dtype=np.int8)


def solved(H, word):
    """Did this decode produce a valid codeword?"""
    return not ((H @ word) % 2).any()


def compare(H, words, damping):
    """Compare both implementations, returning (disagreements, disagreements on a solved word)."""
    settings = dict(DECODER_KWARGS)
    settings['damping_factor'] = damping
    fast = MinSumDecoder(H, **settings)
    reference = ReferenceMinSumDecoder(H, **settings)

    disagreements = on_solved = 0
    for word in words:
        _, ref_word, _ = reference.decode(word)
        _, new_word, _ = fast.decode(word)
        if np.array_equal(ref_word, new_word):
            continue
        disagreements += 1
        if solved(H, ref_word) or solved(H, new_word):
            on_solved += 1
    return disagreements, on_solved


def check_settings_range():
    """The guarantee holds across the accepted damping range."""
    total_disagreements = 0
    for code_id in ('24x48', '56x80', '64x128'):
        code = LDPC_Code.from_default(code_id=code_id)
        words = [word for word, _ in random_error_cases(code, 250, 9, seed=11)]
        for damping in (0.0, 0.3, 0.5, 0.8, 1.0):
            disagreements, on_solved = compare(code.H, words, damping)
            assert on_solved == 0, (
                f"{code_id} at damping {damping}: {on_solved} disagreements on words "
                f"one implementation had decoded; the guarantee is broken")
            total_disagreements += disagreements
    print(f"  damping 0.0 to 1.0 on three codes: no disagreement on any decoded word "
          f"({total_disagreements} on failed decodes, all expected)")

    rng = np.random.default_rng(3)
    for name, rows in DEGENERATE_MATRICES.items():
        H = np.array(rows, dtype=np.int8)
        words = []
        for _ in range(60):
            word = rng.integers(0, 2, size=H.shape[1]).astype(np.int8)
            if ((H @ word) % 2).any():
                words.append(word)
        for damping in (0.0, 0.5, 0.8, 1.0):
            # Damping 1.0 is included for the degree-one check: its outgoing
            # magnitude is the padding infinity, and the blend multiplied that
            # by (1 - damping), so the message array filled with NaN before the
            # inactive-edge mask discarded it.
            with warnings.catch_warnings():
                warnings.simplefilter('error', RuntimeWarning)
                disagreements, _ = compare(H, words, damping)
            assert disagreements == 0, (
                f"{name} at damping {damping}: {disagreements} of {len(words)} differ")
        print(f"  {name:28} {len(words)} words identical at every damping")



def check_zero_damping_boundary():
    """
    The guarantee is bounded to non-zero damping, and this is the matrix that
    bounds it.

    With damping off, nothing mixes the previous message into the new one, terms
    can cancel exactly, and a bit's reliability lands on zero. Its sign is then
    decided by accumulation order, which is the one thing the two implementations
    do differently. On this matrix that flips the outcome from a valid codeword to
    an invalid one, so the claim "whenever either produces a valid codeword, both
    produce the same one" is false in general.

    Nothing is asserted about damping 0.0 here: the point is that it is not
    covered. What is asserted is that every damping value the project actually
    uses agrees, on the very matrix that breaks the zero case.
    """
    H = ZERO_DAMPING_COUNTEREXAMPLE
    word = ZERO_DAMPING_WORD
    settings = dict(DECODER_KWARGS)

    settings['damping_factor'] = 0.0
    _, reference_word, _ = ReferenceMinSumDecoder(H, **settings).decode(word)
    _, fast_word, _ = MinSumDecoder(H, **settings).decode(word)
    if np.array_equal(reference_word, fast_word):
        print("  damping 0.0: the known counterexample now agrees; the boundary "
              "may have moved, and the claim in min_sum_core.py should be re-measured")
    else:
        print(f"  damping 0.0: differ as recorded, reference "
              f"{'solves' if solved(H, reference_word) else 'fails'} and vectorized "
              f"{'solves' if solved(H, fast_word) else 'fails'}")

    for damping in (0.3, 0.5, 0.8, 1.0):
        settings['damping_factor'] = damping
        _, reference_word, _ = ReferenceMinSumDecoder(H, **settings).decode(word)
        _, fast_word, _ = MinSumDecoder(H, **settings).decode(word)
        assert np.array_equal(reference_word, fast_word), (
            f"the counterexample matrix disagrees at damping {damping}, which is "
            f"inside the range the project uses")
    print("  damping 0.3 to 1.0: identical on the same matrix and word")


def run_episodes(agent, env, samples):
    outcomes = []
    for sample in samples:
        agent.reset()
        state = env.reset(sample['noisy'], sample['original'])
        done = False
        actions = []
        while not done:
            action = agent.select_action(state, training=False,
                                         legal_actions=env.get_legal_actions())
            actions.append(action)
            state, _, done, info = env.step(action)
        outcomes.append((info['terminal_reason'], info['steps_taken'],
                         bool(info['success']), tuple(actions)))
    return outcomes


def check_episode_level(full):
    dataset = resolve_project_path('data/ldpc_64x128_main_dataset.h5')
    if not dataset.exists():
        print(f"  skipped: {dataset} not present")
        return

    code = LDPC_Code.from_default(code_id='64x128')
    generator = RobustDataGenerator.from_h5(str(dataset), code=code)
    samples = list(generator.get_iterator('test'))
    if not full:
        # The test split is ordered by curriculum phase, so take a stride rather
        # than a prefix: a prefix would be phase 1 only.
        samples = samples[::max(1, len(samples) // 4000)][:4000]

    env = LDPCEnvironment(code=code, max_steps=12, max_flips_vector=7)

    fast_agent = MinSumExpertAgent(code=code, max_iterations=20, damping_factor=0.8)
    reference_agent = MinSumExpertAgent(code=code, max_iterations=20, damping_factor=0.8)
    reference_agent.decoder = ReferenceMinSumDecoder(code.H, **DECODER_KWARGS)
    assert isinstance(fast_agent.decoder, MinSumDecoder), (
        "MinSumExpertAgent no longer builds the vectorized decoder")

    start = time.perf_counter()
    reference_outcomes = run_episodes(reference_agent, env, samples)
    reference_seconds = time.perf_counter() - start

    start = time.perf_counter()
    fast_outcomes = run_episodes(fast_agent, env, samples)
    fast_seconds = time.perf_counter() - start

    same_order = 0
    reference_successes = fast_successes = 0
    reference_steps = fast_steps = 0
    reasons = collections.Counter()
    for index, (ref, new) in enumerate(zip(reference_outcomes, fast_outcomes)):
        assert ref[0] == new[0], f"episode {index}: terminal reason {new[0]} != {ref[0]}"
        assert ref[1] == new[1], f"episode {index}: step count {new[1]} != {ref[1]}"
        assert ref[2] == new[2], f"episode {index}: success {new[2]} != {ref[2]}"
        assert set(ref[3]) == set(new[3]), f"episode {index}: different actions taken"
        same_order += (ref[3] == new[3])
        reference_successes += ref[2]
        fast_successes += new[2]
        reference_steps += ref[1]
        fast_steps += new[1]
        reasons[new[0]] += 1

    total = len(samples)
    print(f"  episodes            {total:,}")
    print(f"  success             {fast_successes:,} both  ({fast_successes / total:.6%})")
    print(f"  total steps         {fast_steps:,} both")
    print(f"  terminal reasons    {dict(reasons)}")
    print(f"  identical flip order {same_order:,}/{total:,} ({same_order / total:.2%})")
    print(f"  wall time           reference {reference_seconds / 60:.2f} min, "
          f"vectorized {fast_seconds / 60:.2f} min "
          f"({reference_seconds / fast_seconds:.1f}x)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true',
                        help='use the whole 105,000-case test split instead of a stride sample')
    args = parser.parse_args()

    print("Decoder level, random error patterns per code")
    for code_id, max_errors in (('24x48', 3), ('56x80', 5), ('64x128', 6), ('128x256', 6)):
        check_decoder_level(code_id, max_errors)

    print("\nSettings range and degenerate matrices")
    check_settings_range()

    print("\nWhere the equivalence claim stops")
    check_zero_damping_boundary()

    print("\nEpisode level, MinSumExpertAgent over the 64x128 test split")
    check_episode_level(args.full)
    print("\nall checks passed")


if __name__ == '__main__':
    main()
