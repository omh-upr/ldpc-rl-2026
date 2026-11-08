"""Checks for the Gallager-B decoder and its agent.

`GallagerBDecoder` is edge-vectorized; `ReferenceGallagerBDecoder` walks the Tanner
graph node by node and is the specification. Messages are bits, so the two must
agree exactly: same decision, same converged flag, same iteration count.

Run from the project root:

    PYTHONPATH=. python tests/verify_gallager_b.py

Checked at four levels:

  decoder     - fast against reference on random error patterns of weight 1 to 8
                for every bundled code, at the default majority thresholds and at
                Gallager-A thresholds (b = d_v - 1).
  definition  - hand-checkable facts of the algorithm: a word with zero syndrome
                is returned untouched after zero iterations; every converged
                decision is a codeword; the default thresholds are ceil(d_v / 2).
  radius      - every single-bit error on the 64x128 code is corrected.
  episode     - GallagerBAgent driven through the real environment with the
                re-flip mask on the seed-42 ablation split: the episode succeeds
                exactly when the native decoder converges to the transmitted word
                within the flip budget, and the final word is the decision.

Exits nonzero on the first failed assertion.
"""
import collections
import sys

import numpy as np

from src.advanced.gallager_b import GallagerBDecoder, ReferenceGallagerBDecoder
from src.core.baseline_agents import GallagerBAgent
from src.core.environment import LDPCEnvironment
from src.core.ldpc_code import LDPC_Code
from src.data.robust_data_generator import RobustDataGenerator
from src.utils.path_utils import resolve_project_path

CODES = ["24x48", "56x80", "64x128"]


def random_words(n, count, max_errors, seed):
    """Error patterns on the all-zero codeword, valid for any linear code."""
    rng = np.random.default_rng(seed)
    words = []
    for _ in range(count):
        word = np.zeros(n, dtype=np.int64)
        word[rng.choice(n, size=rng.integers(1, max_errors + 1), replace=False)] = 1
        words.append(word)
    return words


def check_decoder_level():
    for code_id in CODES:
        code = LDPC_Code.from_default(code_id=code_id)
        degrees = sorted(set(code.H.sum(axis=0).astype(int).tolist()))
        for label, thresholds in [("majority", None),
                                  ("Gallager A", {d: d - 1 for d in degrees})]:
            fast = GallagerBDecoder(code.H, thresholds=thresholds)
            ref = ReferenceGallagerBDecoder(code.H, thresholds=thresholds)
            words = random_words(code.n, 150, 8, seed=len(code_id))
            converged = 0
            for i, w in enumerate(words):
                a, b = fast.decode(w), ref.decode(w)
                assert np.array_equal(a[0], b[0]), f"{code_id} {label} word {i}: decisions differ"
                assert a[1] == b[1] and a[2] == b[2], \
                    f"{code_id} {label} word {i}: converged/iterations {a[1:]} != {b[1:]}"
                converged += a[1]
            print(f"  {code_id:<7} {label:<10} {len(words)} words identical "
                  f"({converged} converged)")


def check_definition():
    code = LDPC_Code.from_default(code_id="64x128")
    dec = GallagerBDecoder(code.H)
    assert dec.thresholds == {3: 2, 5: 3}, dec.thresholds
    zero = np.zeros(code.n, dtype=np.int64)
    out, ok, it = dec.decode(zero)
    assert ok and it == 0 and not out.any()
    for w in random_words(code.n, 300, 10, seed=7):
        out, ok, _ = dec.decode(w)
        if ok:
            assert not ((code.H @ out) % 2).any()
    print("  thresholds {3: 2, 5: 3}; zero syndrome untouched; converged decisions are codewords")


def check_radius():
    code = LDPC_Code.from_default(code_id="64x128")
    dec = GallagerBDecoder(code.H)
    for i in range(code.n):
        w = np.zeros(code.n, dtype=np.int64)
        w[i] = 1
        out, ok, _ = dec.decode(w)
        assert ok and not out.any(), f"single error at bit {i} not corrected"
    print(f"  all {code.n} single-bit errors corrected")


def check_episode_level(limit=6000):
    dataset = resolve_project_path("data/ablation/s42/curriculum.h5")
    if not dataset.exists():
        print(f"  skipped: {dataset} not present")
        return
    code = LDPC_Code.from_default(code_id="64x128")
    generator = RobustDataGenerator.from_h5(str(dataset), code=code)
    samples = list(generator.get_iterator("test"))
    samples = samples[::max(1, len(samples) // limit)][:limit]
    env = LDPCEnvironment(code=code, max_steps=12, max_flips_vector=7,
                          forbid_immediate_reflip=True)
    agent = GallagerBAgent(code=code)
    native = GallagerBDecoder(code.H)
    reasons = collections.Counter()
    successes = 0
    for index, sample in enumerate(samples):
        agent.reset()
        state = env.reset(sample["noisy"], sample["original"])
        word = np.array(sample["noisy"], dtype=np.int64)
        done = False
        while not done:
            action = agent.select_action(state, training=False,
                                         legal_actions=env.get_legal_actions())
            if action < code.n:
                word[action] ^= 1
            state, _, done, info = env.step(action)
        decision, converged, _ = native.decode(sample["noisy"])
        flips = int((decision != np.array(sample["noisy"])).sum())
        expected = bool(converged and np.array_equal(decision, sample["original"]) and flips <= 7)
        assert bool(info["success"]) == expected, f"episode {index}: success {info['success']}, expected {expected}"
        if converged and flips <= 7:
            assert np.array_equal(word, decision), f"episode {index}: final word is not the decision"
        reasons[info["terminal_reason"]] += 1
        successes += bool(info["success"])
    print(f"  {len(samples):,} episodes, success {successes / len(samples):.2%}, "
          f"terminal reasons {dict(reasons)}")


def main():
    for name, fn in [("decoder", check_decoder_level), ("definition", check_definition),
                     ("radius", check_radius), ("episode", check_episode_level)]:
        print(f"{name}:")
        fn()
    print("all checks passed")


if __name__ == "__main__":
    sys.exit(main())
