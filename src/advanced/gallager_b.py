"""
Gallager's decoding algorithm B for LDPC codes on hard input.

Algorithm B is Gallager's hard-decision message-passing decoder [Gallager, *Low-
Density Parity-Check Codes*, MIT Press, 1963, p. 50], stated as in Richardson and
Urbanke, "The capacity of low-density parity-check codes under message-passing
decoding", IEEE Trans. Inf. Theory 47(2), 2001, Example 4. Messages are bits:

- **Check to variable.** The modulo-two sum of the messages from the check's other
  variables.
- **Variable to check.** The received bit, unless at least ``b`` of the messages from
  the variable's other checks disagree with it, in which case its complement.
- **Initialization.** Every variable sends its received bit.

Algorithm A is the special case ``b = d_v - 1`` (the other checks are unanimous); the
two coincide for degree-three variables.

**Threshold.** Gallager's optimal ``b`` is the smallest integer satisfying

    (1 - p0) / p0  <=  [ (1 + (1 - 2 p)^(d_c - 1)) / (1 - (1 - 2 p)^(d_c - 1)) ]^(2b - d_v + 1)

where ``p0`` is the channel crossover probability and ``p`` the message error rate of
the current iteration. The experiments here have no channel and no ``p0``, and as
``p`` falls the condition reduces to ``2b - d_v + 1 >= 1``. The decoder therefore
uses the fixed majority threshold ``b = ceil(d_v / 2)`` per variable degree by
default: 2 for degree 3 and 3 for degree 5 on the 64x128 code. It is fixed by the
degree, not tuned on data. ``thresholds`` overrides it per degree.

**Decision.** The source leaves the final decision to the implementation. Here each
bit takes the majority of its received value and all its incoming check messages,
and a tie keeps the received value. Decoding stops as soon as the decision has a
zero syndrome (converged), or after ``max_iterations`` (not converged).

Two implementations of the one algorithm live here, as for Min-Sum:
``GallagerBDecoder`` works on flat per-edge arrays and is what the agent uses;
``ReferenceGallagerBDecoder`` walks the Tanner graph one node at a time and is the
executable specification ``tests/verify_gallager_b.py`` checks it against. Messages
are bits, so the two must agree exactly.
"""

from typing import Dict, Optional, Tuple

import numpy as np


def majority_thresholds(degrees: np.ndarray) -> Dict[int, int]:
    """b = ceil(d_v / 2) for every variable degree present."""
    return {int(d): int(np.ceil(d / 2)) for d in np.unique(degrees)}


class GallagerBDecoder:
    """Edge-vectorized Gallager-B decoder."""

    def __init__(self, H_matrix: np.ndarray, max_iterations: int = 20,
                 thresholds: Optional[Dict[int, int]] = None):
        H = (np.asarray(H_matrix) % 2).astype(np.int64)
        self.H = H
        self.m, self.n = H.shape
        self.max_iterations = int(max_iterations)
        self.edge_c, self.edge_v = np.nonzero(H)
        self.var_degree = H.sum(axis=0)
        if (self.var_degree == 0).any():
            raise ValueError("every variable must take part in at least one check")
        table = majority_thresholds(self.var_degree)
        if thresholds:
            table.update({int(k): int(v) for k, v in thresholds.items()})
        self.thresholds = table
        self.edge_b = np.array([table[int(d)] for d in self.var_degree[self.edge_v]])
        self.last_iterations = 0

    def syndrome(self, word: np.ndarray) -> np.ndarray:
        return (self.H @ np.asarray(word, dtype=np.int64)) % 2

    def decode(self, received: np.ndarray) -> Tuple[np.ndarray, bool, int]:
        """Returns (decision, converged, iterations run)."""
        y = (np.asarray(received) % 2).astype(np.int64)
        if not self.syndrome(y).any():
            self.last_iterations = 0
            return y.copy(), True, 0
        y_edge = y[self.edge_v]
        v_to_c = y_edge.copy()
        decision = y
        for iteration in range(1, self.max_iterations + 1):
            # Check to variable: parity of the check's other incoming messages.
            parity = np.bincount(self.edge_c, weights=v_to_c, minlength=self.m).astype(np.int64) % 2
            c_to_v = parity[self.edge_c] ^ v_to_c
            # Disagreements with the received bit, per edge and per variable.
            disagree = (c_to_v != y_edge).astype(np.int64)
            disagree_v = np.bincount(self.edge_v, weights=disagree, minlength=self.n).astype(np.int64)
            # Decision: majority of the received bit and all incoming messages.
            decision = y ^ (2 * disagree_v > self.var_degree + 1).astype(np.int64)
            if not self.syndrome(decision).any():
                self.last_iterations = iteration
                return decision, True, iteration
            # Variable to check: flip when at least b other checks disagree.
            extrinsic = disagree_v[self.edge_v] - disagree
            v_to_c = y_edge ^ (extrinsic >= self.edge_b).astype(np.int64)
        self.last_iterations = self.max_iterations
        return decision, False, self.max_iterations


class ReferenceGallagerBDecoder:
    """Node-by-node Gallager-B, the specification for GallagerBDecoder."""

    def __init__(self, H_matrix: np.ndarray, max_iterations: int = 20,
                 thresholds: Optional[Dict[int, int]] = None):
        H = (np.asarray(H_matrix) % 2).astype(np.int64)
        self.H = H
        self.m, self.n = H.shape
        self.max_iterations = int(max_iterations)
        self.checks_of = [list(np.flatnonzero(H[:, v])) for v in range(self.n)]
        self.vars_of = [list(np.flatnonzero(H[c])) for c in range(self.m)]
        table = majority_thresholds(H.sum(axis=0))
        if thresholds:
            table.update({int(k): int(v) for k, v in thresholds.items()})
        self.thresholds = table

    def decode(self, received: np.ndarray) -> Tuple[np.ndarray, bool, int]:
        y = [int(b) % 2 for b in received]
        if not ((self.H @ np.array(y)) % 2).any():
            return np.array(y), True, 0
        v_to_c = {(v, c): y[v] for v in range(self.n) for c in self.checks_of[v]}
        decision = np.array(y)
        for iteration in range(1, self.max_iterations + 1):
            c_to_v = {}
            for c in range(self.m):
                for v in self.vars_of[c]:
                    c_to_v[(c, v)] = sum(v_to_c[(u, c)] for u in self.vars_of[c] if u != v) % 2
            decision = np.array(y)
            for v in range(self.n):
                votes = [c_to_v[(c, v)] for c in self.checks_of[v]]
                disagree = sum(1 for x in votes if x != y[v])
                if 2 * disagree > len(votes) + 1:
                    decision[v] = 1 - y[v]
            if not ((self.H @ decision) % 2).any():
                return decision, True, iteration
            for v in range(self.n):
                b = self.thresholds[len(self.checks_of[v])]
                for c in self.checks_of[v]:
                    others = sum(1 for c2 in self.checks_of[v]
                                 if c2 != c and c_to_v[(c2, v)] != y[v])
                    v_to_c[(v, c)] = 1 - y[v] if others >= b else y[v]
        return decision, False, self.max_iterations
