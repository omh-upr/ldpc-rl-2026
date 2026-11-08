"""Certify the minimum distance of a binary linear code from its parity-check matrix.

Uses the two-information-set form of the Brouwer-Zimmermann argument. If the
coordinates split into two disjoint information sets A and B, every codeword is
fixed by its restriction to either set, and a codeword of weight at most 2w+1
has weight at most w on A or on B. Enumerating every message of weight <= w on
both sets therefore finds every codeword of weight <= 2w+1, and the minimum
found is exact whenever it is <= 2w+2.

Usage:
    PYTHONPATH=. python scripts/verify_min_distance.py --code-id 64x128 --max-info-weight 6
"""

import argparse
import sys
import time

import numpy as np

from src.core.ldpc_utils import load_matrices


def gf2_rref(M):
    """Row-reduce M over GF(2). Returns (reduced matrix, pivot columns)."""
    M = M.copy() % 2
    rows, cols = M.shape
    pivots = []
    r = 0
    for c in range(cols):
        if r == rows:
            break
        hit = np.nonzero(M[r:, c])[0]
        if hit.size == 0:
            continue
        p = r + hit[0]
        M[[r, p]] = M[[p, r]]
        others = np.nonzero(M[:, c])[0]
        others = others[others != r]
        M[others] ^= M[r]
        pivots.append(c)
        r += 1
    return M[:r], pivots


def generator_from_parity(H):
    """A basis of the null space of H over GF(2), one codeword per row."""
    R, pivots = gf2_rref(H)
    n = H.shape[1]
    free = [c for c in range(n) if c not in set(pivots)]
    G = np.zeros((len(free), n), dtype=np.uint8)
    for i, f in enumerate(free):
        G[i, f] = 1
        for row, p in enumerate(pivots):
            G[i, p] = R[row, f]
    assert not ((H.astype(np.int64) @ G.T.astype(np.int64)) % 2).any()
    return G


def systematic_on(G, info):
    """Row-reduce G so its columns `info` form the identity. Returns the
    parity part (k x (n-k)) as packed uint64 words, or None if `info` is not an
    information set."""
    k, n = G.shape
    rest = [c for c in range(n) if c not in set(info)]
    R, pivots = gf2_rref(G[:, list(info) + rest])
    if len(pivots) < k or pivots != list(range(k)):
        return None
    parity = R[:, k:]
    return pack_rows(parity)


def pack_rows(bits):
    """Pack each row of a 0/1 matrix into little-endian uint64 words."""
    k, width = bits.shape
    words = (width + 63) // 64
    out = np.zeros((k, words), dtype=np.uint64)
    for w in range(words):
        chunk = bits[:, 64 * w:64 * (w + 1)].astype(np.uint64)
        shifts = np.arange(chunk.shape[1], dtype=np.uint64)
        out[:, w] = (chunk << shifts).sum(axis=1, dtype=np.uint64)
    return out


def min_weight_up_to(parity, max_w):
    """Minimum total weight over all nonzero messages of weight <= max_w.

    Total weight is the message weight plus the parity weight. Level-by-level
    enumeration keeps each level's XORs sorted by largest index, so level w+1
    extends each entry only with larger indices. The last level is never
    materialised."""
    k = parity.shape[0]
    best = (np.inf, None)
    # Level 1: single rows, largest index = the row itself.
    level = parity.copy()
    level_max = np.arange(k)
    for w in range(1, max_w + 1):
        pw = np.bitwise_count(level).sum(axis=1).astype(np.int64) + w
        i = int(np.argmin(pw))
        if pw[i] < best[0]:
            best = (int(pw[i]), w)
        print(f"  message weight {w}: {len(level):>12,} codewords, min weight {int(pw.min())}",
              flush=True)
        if w == max_w:
            break
        if w + 1 == max_w:
            # Final level: stream it instead of storing it.
            order = np.argsort(level_max, kind="stable")
            level, level_max = level[order], level_max[order]
            counts = np.searchsorted(level_max, np.arange(k))  # entries with max < j
            fmin, total = np.inf, 0
            for j in range(k):
                prefix = level[:counts[j]]
                if prefix.size == 0:
                    continue
                x = np.bitwise_count(prefix ^ parity[j]).sum(axis=1)
                fmin = min(fmin, int(x.min()) + max_w)
                total += len(prefix)
            print(f"  message weight {max_w}: {total:>12,} codewords, min weight {fmin}", flush=True)
            if fmin < best[0]:
                best = (fmin, max_w)
            break
        new_levels, new_max = [], []
        for j in range(k):
            sel = level_max < j
            if sel.any():
                new_levels.append(level[sel] ^ parity[j])
                new_max.append(np.full(sel.sum(), j))
        level = np.concatenate(new_levels)
        level_max = np.concatenate(new_max)
    return best[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--code-id", default="64x128")
    ap.add_argument("--max-info-weight", type=int, default=None,
                    help="w; the result is exact if the minimum found is <= 2w+2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    H, _ = load_matrices(args.code_id)
    H = (np.asarray(H) % 2).astype(np.uint8)
    m, n = H.shape
    G = generator_from_parity(H)
    k = G.shape[0]
    print(f"code {args.code_id}: n={n}, rank(H)={n - k}, k={k}")
    if n < 2 * k:
        sys.exit("two disjoint information sets need n >= 2k; this script does not cover this code")

    rng = np.random.default_rng(args.seed)
    for attempt in range(1, 10001):
        perm = rng.permutation(n)
        A, B = sorted(perm[:k]), sorted(perm[k:2 * k])
        pa, pb = systematic_on(G, A), systematic_on(G, B)
        if pa is not None and pb is not None:
            break
    else:
        sys.exit("found no pair of disjoint information sets")
    print(f"disjoint information sets found after {attempt} draw(s)")

    w = args.max_info_weight if args.max_info_weight is not None else k
    t0 = time.time()
    found = []
    for name, par in (("A", pa), ("B", pb)):
        print(f"information set {name}:")
        found.append(min_weight_up_to(par, w))
    d = min(found)
    print(f"minimum weight found: {d}  ({time.time() - t0:.0f} s)")
    if d <= 2 * w + 2:
        print(f"CERTIFIED: d_min = {d} (every codeword of weight <= {2 * w + 1} was enumerated)")
    else:
        print(f"lower bound only: d_min >= {2 * w + 2}, upper bound {d}")


if __name__ == "__main__":
    main()
