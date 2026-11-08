"""
Min-Sum Algorithm Core Implementation for LDPC Decoding

This module implements the Min-Sum algorithm, a simplified version of Belief Propagation
for LDPC decoding. The Min-Sum algorithm approximates the sum-product operations with
min-sum operations, providing near-optimal performance with reduced computational complexity.

The implementation follows standard Min-Sum message passing between variable nodes
and check nodes in the LDPC bipartite graph.

Two implementations of that one algorithm live here:

- ``MinSumDecoder`` holds messages in flat per-edge arrays, so each half-iteration is
  a handful of array operations. This is what the agents use.
- ``ReferenceMinSumDecoder`` walks the Tanner graph in Python, one node at a time. It
  is the original implementation, kept as the executable specification that the fast
  one is tested against, not because anything calls it in normal operation.

They are the same algorithm: same min-sum rule with the two-minimum trick, same
damping, same saturation, same early termination on a valid codeword and same
secondary test on message stability. What differs is the order in which a
variable's incoming messages are accumulated, which moves results by the last bits
of a float.

Where that matters, stated as what has been measured rather than as a guarantee.

**With any non-zero damping the two have never been observed to differ at all.**
That covers roughly 15,600 comparisons over random small matrices, 9,000 over the
three bundled codes at damping 0.3 through 1.0 with one to nine errors, and the
real 64x128 test split episode by episode. The shipped values, 0.8 for the expert
and 0.5 for the step-wise agent, are inside that range, so nothing in the pipeline
sits near the boundary below.

**With damping switched off entirely the two can differ, and the difference can
decide whether a word is decoded.** Nothing mixes the previous message into the new
one, so terms can cancel exactly and a reliability lands on zero. A bit whose
log-likelihood ratio is zero is undecided, and its sign then follows the order in
which its incoming messages were summed, which is the one thing these two
implementations do differently.

On the bundled codes that only ever separated two failed decodes: 71 disagreements
in 9,000 comparisons, all at damping 0.0 on 24x48, none on a word either
implementation had decoded. On arbitrary irregular matrices it is not so contained.
Probing random 4x12 matrices at damping 0.0 gave 23 disagreements in 3,898
comparisons, 6 of them on a word one implementation decoded validly and the other
did not. One such matrix is kept in ``tests/verify_minsum_equivalence.py`` as
``ZERO_DAMPING_COUNTEREXAMPLE``.

So damping 0.0 is outside the range these two agree on. ``MinSumExpertAgent`` is
unaffected for two independent reasons: it never runs at that damping, and it acts
on a decode only after confirming its syndrome is zero.

``tests/verify_minsum_equivalence.py`` asserts agreement across damping 0.3 to 1.0,
including on the counterexample matrix, and over the real test split. At damping 0.0
it asserts agreement only on the bundled codes and the degenerate fixtures, which is
as far as the evidence goes.
"""

import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class ReferenceMinSumDecoder:
    """
    Reference Min-Sum implementation: per-node Python loops over the Tanner graph.

    Retained as the specification that ``MinSumDecoder`` is verified against. It is
    roughly 30 to 75 times slower depending on the code, so nothing in the normal
    pipeline should instantiate it.
    
    The Min-Sum algorithm performs iterative message passing between variable nodes
    and check nodes to decode LDPC codes. It approximates the optimal Belief Propagation
    algorithm by replacing sum-product operations with min-sum operations.
    
    Key Features:
    - Iterative message passing with configurable iterations
    - Damping factor for improved convergence
    - Early convergence detection
    - Numerical stability handling
    """
    
    def __init__(self, 
                 H_matrix: np.ndarray,
                 max_iterations: int = 20,
                 damping_factor: float = 0.8,
                 convergence_threshold: float = 1e-6,
                 min_magnitude_threshold: float = 1e-8,
                 max_magnitude_threshold: float = 50.0,
                 llr_scale: float = 2.0):
        """
        Initialize Min-Sum decoder.
        
        Args:
            H_matrix: Parity check matrix (m x n)
            max_iterations: Maximum number of message passing iterations
            damping_factor: Damping factor for message updates (0.0 = no damping, 1.0 = full damping)
            convergence_threshold: Threshold for convergence detection
            min_magnitude_threshold: Minimum magnitude threshold for numerical stability
            max_magnitude_threshold: Maximum magnitude threshold to prevent overflow
            llr_scale: Constant magnitude for hard-decision LLR mapping (0 -> +llr_scale, 1 -> -llr_scale)
        """
        self.H = H_matrix.astype(np.int8)  # Ensure integer type for efficiency
        self.m, self.n = self.H.shape  # m = number of checks, n = number of variables
        self.max_iterations = max_iterations
        self.damping_factor = damping_factor
        self.convergence_threshold = convergence_threshold
        self.min_magnitude_threshold = min_magnitude_threshold
        self.max_magnitude_threshold = max_magnitude_threshold
        self.llr_scale = llr_scale
        
        # Precompute adjacency lists for efficient message passing
        self._precompute_adjacency_lists()
        
        logger.debug(f"ReferenceMinSumDecoder initialized: {self.m}x{self.n} matrix, "
                    f"max_iter={max_iterations}, damping={damping_factor}")
    
    def _precompute_adjacency_lists(self):
        """Precompute adjacency lists for efficient message passing."""
        # For each check, store connected variable indices
        self.check_to_vars = []
        for check_idx in range(self.m):
            connected_vars = np.where(self.H[check_idx] == 1)[0].tolist()
            self.check_to_vars.append(connected_vars)
        
        # For each variable, store connected check indices
        self.var_to_checks = []
        for var_idx in range(self.n):
            connected_checks = np.where(self.H[:, var_idx] == 1)[0].tolist()
            self.var_to_checks.append(connected_checks)
        
        # For each variable, map its connected check_idx to its position in the list
        self.var_to_check_pos = [{} for _ in range(self.n)]
        for var_idx, checks in enumerate(self.var_to_checks):
            for i, check_idx in enumerate(checks):
                self.var_to_check_pos[var_idx][check_idx] = i

        # For each check, map its connected var_idx to its position in the list
        self.check_to_var_pos = [{} for _ in range(self.m)]
        for check_idx, variables in enumerate(self.check_to_vars):
            for i, var_idx in enumerate(variables):
                self.check_to_var_pos[check_idx][var_idx] = i
        
        logger.debug(f"Adjacency lists computed: avg degree = "
                    f"{np.mean([len(vars) for vars in self.check_to_vars]):.1f}")
    
    def decode(self, 
               received_codeword: np.ndarray, 
               syndrome: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Run Min-Sum decoding algorithm.
        
        Args:
            received_codeword: Received codeword (hard decisions)
            syndrome: Known syndrome vector (computed if not provided)
            
        Returns:
            Tuple of (bit_reliabilities, decoded_codeword, converged)
            - bit_reliabilities: Log-likelihood ratios for each bit
            - decoded_codeword: Hard decision decoded codeword
            - converged: Whether algorithm converged
        """
        # Compute syndrome if not provided
        if syndrome is None:
            syndrome = (self.H @ received_codeword) % 2
        
        # Check if already valid codeword
        if np.sum(syndrome) == 0:
            reliabilities = np.ones(self.n) * 5.0  # High confidence in all bits
            return reliabilities, received_codeword.copy(), True
        
        # Initialize channel log-likelihood ratios
        channel_llr = self._compute_initial_llr(received_codeword)
        
        # Initialize message arrays
        # v_to_c_messages[var_idx][check_pos] = message from variable var_idx to check at position check_pos
        v_to_c_messages = self._initialize_var_to_check_messages(channel_llr)
        
        # c_to_v_messages[check_idx][var_pos] = message from check check_idx to variable at position var_pos  
        c_to_v_messages = self._initialize_check_to_var_messages()
        
        converged = False
        found_valid_codeword = False

        for iteration in range(self.max_iterations):
            # Store previous messages for convergence check
            prev_c_to_v = [msg.copy() for msg in c_to_v_messages]

            # Update check-to-variable messages (Min-Sum rule)
            self._update_check_to_var_messages(v_to_c_messages, c_to_v_messages)

            # Update variable-to-check messages
            self._update_var_to_check_messages(channel_llr, c_to_v_messages, v_to_c_messages)

            # Early termination: tentative hard decisions satisfy parity checks
            bit_reliabilities = self._compute_final_reliabilities(channel_llr, c_to_v_messages)
            decoded_codeword = (bit_reliabilities < 0).astype(np.int8)
            if np.sum(self.compute_syndrome(decoded_codeword)) == 0:
                converged = True
                found_valid_codeword = True
                logger.debug(f"Min-Sum reached valid codeword at iteration {iteration + 1}")
                break

            # Secondary convergence check based on message stability
            if self._check_convergence(prev_c_to_v, c_to_v_messages):
                converged = True
                logger.debug(f"Min-Sum converged at iteration {iteration + 1}")
                break

        if not converged:
            logger.debug(f"Min-Sum did not converge after {self.max_iterations} iterations")

        # Compute final bit reliabilities if not already computed by early termination
        if not found_valid_codeword:
            bit_reliabilities = self._compute_final_reliabilities(channel_llr, c_to_v_messages)
            decoded_codeword = (bit_reliabilities < 0).astype(np.int8)
        
        return bit_reliabilities, decoded_codeword, converged
    
    def _compute_initial_llr(self, received_codeword: np.ndarray) -> np.ndarray:
        """
        Compute initial channel log-likelihood ratios.
        
        For hard-decision decoding, we assign high confidence to received bits
        and lower confidence based on uncertainty.
        
        Args:
            received_codeword: Received hard decisions
            
        Returns:
            Initial log-likelihood ratios
        """
        # Deterministic hard-decision LLR mapping for speed and reproducibility
        # Positive = confidence in 0, Negative = confidence in 1
        return np.where(received_codeword == 0, self.llr_scale, -self.llr_scale)
    
    def _initialize_var_to_check_messages(self, channel_llr: np.ndarray) -> list:
        """Initialize variable-to-check messages with channel LLRs."""
        v_to_c_messages = []
        for var_idx in range(self.n):
            connected_checks = self.var_to_checks[var_idx]
            # Initialize with channel LLR for all connected checks
            var_messages = [channel_llr[var_idx]] * len(connected_checks)
            v_to_c_messages.append(var_messages)
        
        return v_to_c_messages
    
    def _initialize_check_to_var_messages(self) -> list:
        """Initialize check-to-variable messages to zeros."""
        c_to_v_messages = []
        for check_idx in range(self.m):
            connected_vars = self.check_to_vars[check_idx]
            # Initialize with zeros
            check_messages = [0.0] * len(connected_vars)
            c_to_v_messages.append(check_messages)
        
        return c_to_v_messages
    
    def _update_check_to_var_messages(self, 
                                     v_to_c_messages: list, 
                                     c_to_v_messages: list):
        """
        Update check-to-variable messages using Min-Sum rule.
        
        Min-Sum rule: message = sign_product * minimum_magnitude
        where sign_product is XOR of signs and minimum_magnitude is smallest absolute value.
        """
        for check_idx in range(self.m):
            connected_vars = self.check_to_vars[check_idx]
            
            if len(connected_vars) < 2:
                continue  # Skip checks with less than 2 variables
            
            # Collect incoming messages from variables
            incoming_messages = np.zeros(len(connected_vars))
            for var_pos, var_idx in enumerate(connected_vars):
                check_pos_in_var = self.var_to_check_pos[var_idx][check_idx]
                incoming_messages[var_pos] = v_to_c_messages[var_idx][check_pos_in_var]
            
            # Compute signs and magnitudes once
            signs = np.sign(incoming_messages)
            # Treat zeros as +1 to avoid zero sign product
            signs[signs == 0] = 1.0
            magnitudes = np.abs(incoming_messages)
            magnitudes = np.maximum(magnitudes, self.min_magnitude_threshold) # Numerical stability
            
            # Find the two smallest magnitudes and their indices
            # Use np.partition to efficiently find the two smallest elements
            if len(magnitudes) > 1:
                min_magnitudes_sorted_indices = np.argpartition(magnitudes, 1)
                min1_idx = min_magnitudes_sorted_indices[0]
                min2_idx = min_magnitudes_sorted_indices[1]
                
                min1 = magnitudes[min1_idx]
                min2 = magnitudes[min2_idx]
            else: # Handle case with only one connected variable (should be caught by len(connected_vars) < 2)
                min1_idx = 0
                min1 = magnitudes[0] if len(magnitudes) > 0 else 0.0
                min2 = 0.0 # Not applicable
            
            total_sign_product = np.prod(signs)
            
            # Compute outgoing messages for each connected variable
            for var_pos, var_idx in enumerate(connected_vars):
                # Calculate sign product for other messages
                # total_sign_product / signs[var_pos] is equivalent to multiplying all other signs
                sign_product_others = total_sign_product * signs[var_pos] # Since signs are -1 or 1, 1/sign is sign
                
                # Calculate minimum magnitude for other messages
                if var_pos == min1_idx:
                    min_magnitude_others = min2
                else:
                    min_magnitude_others = min1
                
                new_message = sign_product_others * min_magnitude_others
                
                # Apply damping
                old_message = c_to_v_messages[check_idx][var_pos]
                damped_message = (self.damping_factor * old_message + 
                                (1 - self.damping_factor) * new_message)
                
                # Apply saturation to prevent numerical overflow
                damped_message = np.clip(damped_message, -self.max_magnitude_threshold, self.max_magnitude_threshold)
                
                c_to_v_messages[check_idx][var_pos] = damped_message
    
    def _update_var_to_check_messages(self, 
                                     channel_llr: np.ndarray,
                                     c_to_v_messages: list, 
                                     v_to_c_messages: list):
        """Update variable-to-check messages by summing channel LLR and other check messages."""
        for var_idx in range(self.n):
            connected_checks = self.var_to_checks[var_idx]
            
            # Collect incoming messages from checks using precomputed positions
            incoming_messages = np.zeros(len(connected_checks))
            for check_pos, check_idx in enumerate(connected_checks):
                var_pos_in_check = self.check_to_var_pos[check_idx][var_idx]
                incoming_messages[check_pos] = c_to_v_messages[check_idx][var_pos_in_check]
            
            total_incoming_sum = np.sum(incoming_messages)
            
            # Compute outgoing messages for each connected check
            for check_pos, check_idx in enumerate(connected_checks):
                # Sum channel LLR with all other check messages (excluding current check)
                sum_of_other_messages = total_incoming_sum - incoming_messages[check_pos]
                
                new_message = channel_llr[var_idx] + sum_of_other_messages
                
                # Apply saturation to prevent numerical overflow
                new_message = np.clip(new_message, -self.max_magnitude_threshold, self.max_magnitude_threshold)
                
                v_to_c_messages[var_idx][check_pos] = new_message
    
    def _compute_final_reliabilities(self, 
                                   channel_llr: np.ndarray, 
                                   c_to_v_messages: list) -> np.ndarray:
        """Compute final bit reliabilities by combining channel LLR with all check messages."""
        bit_reliabilities = channel_llr.copy()
        
        for var_idx in range(self.n):
            connected_checks = self.var_to_checks[var_idx]
            
            # Sum all incoming check messages using precomputed positions
            check_message_sum = 0.0
            for check_pos, check_idx in enumerate(connected_checks):
                var_pos_in_check = self.check_to_var_pos[check_idx][var_idx]
                check_message_sum += c_to_v_messages[check_idx][var_pos_in_check]
            
            bit_reliabilities[var_idx] += check_message_sum
        
        # Apply final saturation to reliabilities
        bit_reliabilities = np.clip(bit_reliabilities, -self.max_magnitude_threshold, self.max_magnitude_threshold)
        
        return bit_reliabilities
    
    def _check_convergence(self, prev_messages: list, current_messages: list) -> bool:
        """Check if message passing has converged by comparing message changes."""
        total_change = 0.0
        message_count = 0
        
        for check_idx in range(self.m):
            for var_pos in range(len(current_messages[check_idx])):
                change = abs(current_messages[check_idx][var_pos] - 
                           prev_messages[check_idx][var_pos])
                total_change += change
                message_count += 1
        
        if message_count == 0:
            return True
        
        avg_change = total_change / message_count
        return avg_change < self.convergence_threshold
    
    def get_bit_reliabilities(self, 
                            received_codeword: np.ndarray, 
                            syndrome: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Convenience method to get bit reliabilities without full decode output.
        
        Args:
            received_codeword: Received codeword
            syndrome: Known syndrome (computed if not provided)
            
        Returns:
            Bit reliability scores (lower = less reliable, higher = more reliable)
        """
        bit_reliabilities, _, _ = self.decode(received_codeword, syndrome)
        
        # Convert LLRs to reliability scores
        # LLR > 0 means confident in bit=0, LLR < 0 means confident in bit=1
        # For reliability, we want magnitude (confidence level regardless of bit value)
        reliability_scores = np.abs(bit_reliabilities)
        
        return reliability_scores
    
    def compute_syndrome(self, codeword: np.ndarray) -> np.ndarray:
        """Compute syndrome for given codeword."""
        return (self.H @ codeword) % 2
    
    def is_valid_codeword(self, codeword: np.ndarray) -> bool:
        """Check if codeword satisfies all parity check constraints."""
        syndrome = self.compute_syndrome(codeword)
        return np.sum(syndrome) == 0
    
    def get_algorithm_info(self) -> dict:
        """Return information about the decoder configuration."""
        return {
            'algorithm': 'Min-Sum',
            'matrix_size': f"{self.m}x{self.n}",
            'max_iterations': self.max_iterations,
            'damping_factor': self.damping_factor,
            'convergence_threshold': self.convergence_threshold,
            'variable_degrees': [len(checks) for checks in self.var_to_checks],
            'check_degrees': [len(vars) for vars in self.check_to_vars]
        }


class MinSumDecoder:
    """
    Edge-vectorized Min-Sum decoder.

    Same algorithm as :class:`ReferenceMinSumDecoder`, different data layout. Messages
    live in flat arrays indexed by Tanner-graph edge rather than in per-node Python
    lists, so a half-iteration is a few array operations instead of a loop over every
    node and every edge.

    Two index structures make that work, both built once:

    - Edges arrive from ``np.nonzero(H)`` in row-major order, so they are already
      grouped by check. Reshaping to (checks x max_degree) exposes each check's
      incoming messages as one row. Checks of unequal degree are padded with neutral
      entries: magnitude infinity, which can never win a minimum, and sign +1, which
      cannot change a sign product.
    - A stable argsort of the column indices groups the same edges by variable, so
      each variable's incoming messages are one ``np.add.reduceat`` away.

    Checks of degree below two are skipped exactly as the reference does: their
    outgoing messages keep the value they already hold.
    """

    def __init__(self,
                 H_matrix: np.ndarray,
                 max_iterations: int = 20,
                 damping_factor: float = 0.8,
                 convergence_threshold: float = 1e-6,
                 min_magnitude_threshold: float = 1e-8,
                 max_magnitude_threshold: float = 50.0,
                 llr_scale: float = 2.0):
        """
        Initialize Min-Sum decoder.

        Args:
            H_matrix: Parity check matrix (m x n)
            max_iterations: Maximum number of message passing iterations
            damping_factor: Damping factor for message updates (0.0 = no damping, 1.0 = full damping)
            convergence_threshold: Threshold for convergence detection
            min_magnitude_threshold: Minimum magnitude threshold for numerical stability
            max_magnitude_threshold: Maximum magnitude threshold to prevent overflow
            llr_scale: Constant magnitude for hard-decision LLR mapping (0 -> +llr_scale, 1 -> -llr_scale)
        """
        self.H = H_matrix.astype(np.int8)
        self.m, self.n = self.H.shape
        self.max_iterations = max_iterations
        self.damping_factor = damping_factor
        self.convergence_threshold = convergence_threshold
        self.min_magnitude_threshold = min_magnitude_threshold
        self.max_magnitude_threshold = max_magnitude_threshold
        self.llr_scale = llr_scale

        self._build_edge_index()

        logger.debug(f"MinSumDecoder initialized: {self.m}x{self.n} matrix, "
                     f"{self.num_edges} edges, max_iter={max_iterations}, "
                     f"damping={damping_factor}")

    def _build_edge_index(self):
        """Precompute the check-grouped and variable-grouped views of the edge set."""
        check_of_edge, var_of_edge = np.nonzero(self.H)
        self.edge_var = var_of_edge
        self.num_edges = int(check_of_edge.size)

        self.check_degrees = np.bincount(check_of_edge, minlength=self.m)
        self.variable_degrees = np.bincount(var_of_edge, minlength=self.n)

        # Check-grouped view, padded to the largest check degree. Edges are already
        # ordered by check, so an edge's position within its group is its rank there.
        self._max_check_degree = int(self.check_degrees.max()) if self.num_edges else 0
        if self._max_check_degree:
            position_in_check = np.concatenate(
                [np.arange(degree) for degree in self.check_degrees if degree > 0]
            ) if self.num_edges else np.zeros(0, dtype=int)
            self._slot = check_of_edge * self._max_check_degree + position_in_check
            occupied = np.zeros(self.m * self._max_check_degree, dtype=bool)
            occupied[self._slot] = True
            self._occupied = occupied.reshape(self.m, self._max_check_degree)
            self._slot_buffer = np.zeros(self.m * self._max_check_degree, dtype=np.float64)
            self._column_index = np.arange(self._max_check_degree)[None, :]

        # A check with fewer than two edges has no "other" message to pass on, so the
        # reference skips it entirely and its outgoing messages stay where they were.
        self._edge_active = self.check_degrees[check_of_edge] >= 2
        self._any_inactive = not bool(np.all(self._edge_active))
        self._can_vectorize_checks = self._max_check_degree >= 2

        # Variable-grouped view of the same edges.
        var_order = np.argsort(var_of_edge, kind='stable')
        self._var_order = var_order
        starts = np.concatenate(([0], np.cumsum(self.variable_degrees)[:-1]))

        # reduceat is given starts only for variables that actually have edges. A
        # variable with no edges shares the start of whatever follows it, and a
        # trailing one has a start equal to the edge count, which reduceat rejects
        # outright rather than returning a value that could be masked away after.
        self._var_has_edges = self.variable_degrees > 0
        self._connected_vars = np.nonzero(self._var_has_edges)[0]
        self._connected_var_starts = starts[self._var_has_edges]

        logger.debug(f"Edge index built: {self.num_edges} edges, "
                     f"max check degree {self._max_check_degree}")

    def _sum_incoming_per_variable(self, check_to_var: np.ndarray) -> np.ndarray:
        """Sum of the check messages arriving at each variable.

        A variable with no edges keeps a sum of zero, which is what summing an
        empty set of messages gives and what the per-node reference produces.
        """
        sums = np.zeros(self.n)
        if self._connected_var_starts.size:
            sums[self._connected_vars] = np.add.reduceat(
                check_to_var[self._var_order], self._connected_var_starts)
        return sums

    def decode(self,
               received_codeword: np.ndarray,
               syndrome: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Run Min-Sum decoding algorithm.

        Args:
            received_codeword: Received codeword (hard decisions)
            syndrome: Known syndrome vector (computed if not provided)

        Returns:
            Tuple of (bit_reliabilities, decoded_codeword, converged)
            - bit_reliabilities: Log-likelihood ratios for each bit
            - decoded_codeword: Hard decision decoded codeword
            - converged: Whether algorithm converged
        """
        if syndrome is None:
            syndrome = (self.H @ received_codeword) % 2

        # Already a valid codeword: nothing to correct, full confidence everywhere.
        if np.sum(syndrome) == 0:
            return np.ones(self.n) * 5.0, received_codeword.copy(), True

        channel_llr = self._compute_initial_llr(received_codeword)

        var_to_check = channel_llr[self.edge_var].astype(np.float64)
        check_to_var = np.zeros(self.num_edges, dtype=np.float64)

        damping = self.damping_factor
        limit = self.max_magnitude_threshold

        converged = False
        found_valid_codeword = False

        for iteration in range(self.max_iterations):
            previous_check_to_var = check_to_var.copy()

            check_to_var = self._update_check_to_var(var_to_check, previous_check_to_var)

            incoming_sums = self._sum_incoming_per_variable(check_to_var)
            var_to_check = np.clip(
                channel_llr[self.edge_var] + (incoming_sums[self.edge_var] - check_to_var),
                -limit, limit)

            # Early termination: tentative hard decisions satisfy parity checks
            bit_reliabilities = np.clip(channel_llr + incoming_sums, -limit, limit)
            decoded_codeword = (bit_reliabilities < 0).astype(np.int8)
            if not np.any((self.H @ decoded_codeword) % 2):
                converged = True
                found_valid_codeword = True
                logger.debug(f"Min-Sum reached valid codeword at iteration {iteration + 1}")
                break

            # Secondary convergence check based on message stability
            if np.mean(np.abs(check_to_var - previous_check_to_var)) < self.convergence_threshold:
                converged = True
                logger.debug(f"Min-Sum converged at iteration {iteration + 1}")
                break

        if not converged:
            logger.debug(f"Min-Sum did not converge after {self.max_iterations} iterations")

        if not found_valid_codeword:
            bit_reliabilities = np.clip(
                channel_llr + self._sum_incoming_per_variable(check_to_var), -limit, limit)
            decoded_codeword = (bit_reliabilities < 0).astype(np.int8)

        return bit_reliabilities, decoded_codeword, converged

    def _update_check_to_var(self, var_to_check: np.ndarray,
                             previous_check_to_var: np.ndarray) -> np.ndarray:
        """
        One check-to-variable half-iteration under the Min-Sum rule.

        Each outgoing message is the product of the other incoming signs times the
        smallest other incoming magnitude, which is the second smallest of the group
        for whichever edge holds the smallest and the smallest otherwise.
        """
        if not self._can_vectorize_checks:
            return previous_check_to_var

        buffer = self._slot_buffer
        buffer[:] = 0.0
        buffer[self._slot] = var_to_check
        grouped = buffer.reshape(self.m, self._max_check_degree)

        # Padding is neutral: sign +1 leaves a product unchanged, magnitude infinity
        # can never be a minimum.
        signs = np.where(self._occupied, np.where(grouped >= 0, 1.0, -1.0), 1.0)
        magnitudes = np.where(self._occupied,
                              np.maximum(np.abs(grouped), self.min_magnitude_threshold),
                              np.inf)

        total_sign = signs.prod(axis=1, keepdims=True)

        two_smallest = np.argpartition(magnitudes, 1, axis=1)
        smallest = np.take_along_axis(magnitudes, two_smallest[:, :1], axis=1)
        second_smallest = np.take_along_axis(magnitudes, two_smallest[:, 1:2], axis=1)
        magnitude_of_others = np.where(self._column_index == two_smallest[:, :1],
                                       second_smallest, smallest)

        # Signs are +/-1, so multiplying by an edge's own sign removes it from the product.
        new_messages = (total_sign * signs * magnitude_of_others).reshape(-1)[self._slot]

        if self._any_inactive:
            # A check of degree 1 has no other edges, so its "magnitude of the
            # others" is the padding infinity. The reference implementation skips
            # such checks outright and the line at the end of this method restores
            # their previous message, but the blend below would reach 0.0 * inf
            # first and put NaN in the array whenever damping is 1.0. Neutralise
            # them before the arithmetic rather than after it. The value chosen
            # here never survives; only its finiteness matters.
            new_messages = np.where(self._edge_active, new_messages, 0.0)

        damped = (self.damping_factor * previous_check_to_var
                  + (1.0 - self.damping_factor) * new_messages)
        updated = np.clip(damped, -self.max_magnitude_threshold, self.max_magnitude_threshold)

        if self._any_inactive:
            updated = np.where(self._edge_active, updated, previous_check_to_var)
        return updated

    def _compute_initial_llr(self, received_codeword: np.ndarray) -> np.ndarray:
        """
        Compute initial channel log-likelihood ratios.

        Args:
            received_codeword: Received hard decisions

        Returns:
            Initial log-likelihood ratios
        """
        # Deterministic hard-decision LLR mapping for speed and reproducibility
        # Positive = confidence in 0, Negative = confidence in 1
        return np.where(received_codeword == 0, self.llr_scale, -self.llr_scale).astype(np.float64)

    def get_bit_reliabilities(self,
                              received_codeword: np.ndarray,
                              syndrome: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Convenience method to get bit reliabilities without full decode output.

        Args:
            received_codeword: Received codeword
            syndrome: Known syndrome (computed if not provided)

        Returns:
            Bit reliability scores (lower = less reliable, higher = more reliable)
        """
        bit_reliabilities, _, _ = self.decode(received_codeword, syndrome)
        return np.abs(bit_reliabilities)

    def compute_syndrome(self, codeword: np.ndarray) -> np.ndarray:
        """Compute syndrome for given codeword."""
        return (self.H @ codeword) % 2

    def is_valid_codeword(self, codeword: np.ndarray) -> bool:
        """Check if codeword satisfies all parity check constraints."""
        return np.sum(self.compute_syndrome(codeword)) == 0

    def get_algorithm_info(self) -> dict:
        """Return information about the decoder configuration."""
        return {
            'algorithm': 'Min-Sum',
            'matrix_size': f"{self.m}x{self.n}",
            'max_iterations': self.max_iterations,
            'damping_factor': self.damping_factor,
            'convergence_threshold': self.convergence_threshold,
            'variable_degrees': self.variable_degrees.tolist(),
            'check_degrees': self.check_degrees.tolist()
        }
