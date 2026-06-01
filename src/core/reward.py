"""
Reward Functions for LDPC Decoding RL Environment

This module contains reward function implementations for training RL agents
on LDPC decoding tasks. The reward functions are designed to encourage
proper decoding behavior while discouraging reward hacking.

Design Philosophy:
==================

The ImprovedRewardFunction implements a multi-layered incentive structure:

1. **Asymmetric Hit/Miss Penalties**: Misses are punished harder than hits are rewarded
   (1.5x asymmetry) to discourage random exploration and encourage careful action
   selection in the structured LDPC problem space.

2. **Clear Success Signal**: Large terminal reward (5x base) for achieving zero
   syndrome emphasizes the primary goal of error correction.

3. **Proper STOP Training**: Significant rewards/penalties (±2x base) for STOP action
   usage trains the agent to terminate episodes appropriately, preventing both
   premature stopping and inefficient continuation.

4. **Efficiency Pressure**: Modest step penalty (0.25x base) encourages shorter
   solution paths without overwhelming the learning signal.

5. **Mixed Outcome Handling**: Moderate penalty (1x base) for actions that improve
   one metric while worsening another, providing clear feedback without being
   overly harsh on complex decisions.

The reward structure creates a hierarchy: Terminal outcomes > Individual actions > 
Efficiency, which is appropriate for error correction where correctness is paramount
but efficiency still matters.

Expected Behaviors:
==================
- Methodical, goal-oriented exploration
- Proper episode termination
- Reduced reward hacking through asymmetric penalties
- Focus on achieving successful decoding rather than gaming intermediate rewards
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
from src.core.ldpc_code import LDPC_Code


class BaseRewardFunction(ABC):
    """
    Abstract base class for reward functions in LDPC decoding environments.
    """

    # Phase 1.2 Optimization: Flag indicating if this reward function needs prev_state
    # Default to True for safety (backward compatibility with custom reward functions)
    uses_prev_state: bool = True

    @abstractmethod
    def compute_reward(self, state: np.ndarray, action: int, next_state: np.ndarray,
                      done: bool) -> float:
        """
        Computes reward R(s, a, s') according to RL theory.

        Args:
            state: Pre-action state s_t. When uses_prev_state=False, this is a
                   copy of next_state (not the true s_t) to avoid extra computation.
            action: Action taken a_t
            next_state: Post-action state s_{t+1}
            done: Terminal state indicator

        Returns:
            float: Reward value
        """
        pass

class ImprovedRewardFunction(BaseRewardFunction):
    """
    Improved reward function with minimal but effective incentive structure.

    This reward function focuses on four core components:
    1. Success bonus for achieving zero syndrome (terminal goal)
    2. STOP action training to encourage proper episode termination
    3. Step penalty for efficiency (encourages shorter episodes)
    4. Exponential flip penalty to discourage excessive bit flipping

    This removes intermediate action feedback (hit/miss rewards for individual
    bit flips), creating a sparser but cleaner learning signal that relies
    primarily on terminal success while still providing essential STOP action
    guidance.

    Design Philosophy:
    - Avoids syndrome reduction bias (greedy strategy isn't always optimal)
    - Sparse rewards allow agent to discover non-greedy optimal strategies
    - STOP feedback prevents "give up early" convergence on difficult cases
    - Gentle flip penalty (negligible within correction capability)

    Reward Components:
    - Success bonus: 5*base for achieving zero syndrome (terminal only)
    - STOP bonus: +1*base for correct termination (syndrome=0)
    - STOP penalty: -1.5*base for premature termination (syndrome>0)
    - Step penalty: -0.15*base per step for efficiency (softened for exploration)
    - Flip penalty: -0.75*exp(flip_count - correction_capability) for excessive flips

    For 64x128 code (correction_capability=6, d_min=14) with 1-6 error training:
    - Flip penalty is small in operating range (~0.005 to ~0.75 for 1-6 flips)
    - Success bonus dominates (12-18 points for typical successful episodes)
    - STOP training prevents premature termination on correctable cases
    """

    # Phase 1.2 Optimization: This reward function only uses next_state, not prev_state
    uses_prev_state: bool = False

    def __init__(self, code: LDPC_Code, base: float = 5.0, correction_capability: Optional[int] = None):
        """
        Initialize the simplified reward function.

        Args:
            code: LDPC code instance for dynamic dimensions (required)
            base: Base reward value that scales all other rewards
            correction_capability: Maximum allowed flips before penalty escalates exponentially.
                                 If None (default), uses the code's theoretical correction capability.
        """
        self.code = code
        self.base = base
        self.correction_capability = correction_capability or self.code.correction_capability

        # Reward multipliers
        self.success_multiplier = 5.0        # 5*base for success
        self.stop_success_multiplier = 1.0   # +1*base for correct STOP
        self.stop_failure_multiplier = -1.5  # -1.5*base for premature STOP
        self.step_penalty = -0.15            # -0.15*base per step (softened from -0.25)
        self.flip_penalty_multiplier = -0.75 # Exponential flip cost

        # Maximum exponent to prevent np.exp overflow (exp(20) ≈ 4.9e8, safely below float64 max)
        self._max_exp_arg = 20.0

    def compute_reward(self, state: np.ndarray, action: int, next_state: np.ndarray,
                      done: bool) -> float:
        """
        Computes simplified reward focusing on terminal success and STOP training.

        This reward function provides minimal intermediate feedback, relying on:
        - Large terminal success bonus to indicate goal achievement
        - STOP action feedback to prevent premature termination
        - Small per-step penalty to encourage efficiency (softened for exploration)
        - Exponential flip penalty (negligible within correction capability)

        Args:
            state: Current state s_t with metadata [syndrome_weight, penalty_sum, step_count]
            action: Action taken a_t (0-n-1: bit flips, n: STOP)
            next_state: Resulting state s_{t+1}
            done: Terminal state indicator

        Returns:
            float: Computed reward value
        """
        reward = 0.0

        # Extract syndrome weight from metadata using dynamic ranges
        ranges = self.code.state_component_ranges
        new_syndrome = next_state[ranges['syndrome_weight'][0]]

        # Success bonus (terminal success only)
        if done and (new_syndrome == 0):
            reward += self.success_multiplier * self.base

        # STOP action training
        if action >= self.code.n:  # STOP action
            if new_syndrome == 0:
                # Bonus for correct termination
                reward += self.stop_success_multiplier * self.base
            else:
                # Strong penalty for premature termination
                reward += self.stop_failure_multiplier * self.base

        # Step penalty (applied to every step)
        reward += self.step_penalty * self.base

        # Exponential flip penalty based on current flip count
        flip_range = ranges['flip_vector']
        current_flip_count = np.sum(next_state[flip_range[0]:flip_range[1]])
        exp_arg = min(current_flip_count - self.correction_capability, self._max_exp_arg)
        reward += self.flip_penalty_multiplier * np.exp(exp_arg)

        return reward

    def get_reward_breakdown(self, state: np.ndarray, action: int, next_state: np.ndarray,
                           done: bool) -> dict:
        """
        Returns detailed breakdown of reward components for analysis.

        Args:
            state: Current state s_t
            action: Action taken a_t
            next_state: Resulting state s_{t+1}
            done: Terminal state indicator

        Returns:
            dict: Breakdown of reward components including STOP action feedback
        """
        breakdown = {
            'success_bonus': 0.0,
            'stop_reward': 0.0,
            'stop_action': False,
            'step_penalty': self.step_penalty * self.base,
            'flip_penalty': 0.0,
            'flip_count': 0,
            'total_reward': 0.0
        }

        # Extract syndrome weight using dynamic ranges
        ranges = self.code.state_component_ranges
        new_syndrome = next_state[ranges['syndrome_weight'][0]]

        # Success bonus
        if done and new_syndrome == 0:
            breakdown['success_bonus'] = self.success_multiplier * self.base

        # STOP action feedback
        if action >= self.code.n:  # STOP action
            breakdown['stop_action'] = True
            if new_syndrome == 0:
                breakdown['stop_reward'] = self.stop_success_multiplier * self.base
            else:
                breakdown['stop_reward'] = self.stop_failure_multiplier * self.base

        # Flip penalty calculation using dynamic range
        flip_range = ranges['flip_vector']
        current_flip_count = np.sum(next_state[flip_range[0]:flip_range[1]])
        breakdown['flip_count'] = int(current_flip_count)
        exp_arg = min(current_flip_count - self.correction_capability, self._max_exp_arg)
        breakdown['flip_penalty'] = self.flip_penalty_multiplier * np.exp(exp_arg)

        breakdown['total_reward'] = (breakdown['success_bonus'] +
                                   breakdown['stop_reward'] +
                                   breakdown['step_penalty'] +
                                   breakdown['flip_penalty'])

        return breakdown