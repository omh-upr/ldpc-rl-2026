"""
Min-Sum Algorithm Core Implementation for LDPC Decoding

This module implements the Min-Sum algorithm, a simplified version of Belief Propagation
for LDPC decoding. The Min-Sum algorithm approximates the sum-product operations with
min-sum operations, providing near-optimal performance with reduced computational complexity.

The implementation follows standard Min-Sum message passing between variable nodes
and check nodes in the LDPC bipartite graph.
"""

import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class MinSumDecoder:
    """
    Core Min-Sum algorithm implementation for LDPC decoding.
    
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
        
        logger.debug(f"MinSumDecoder initialized: {self.m}x{self.n} matrix, "
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
