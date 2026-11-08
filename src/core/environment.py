"""
LDPC Environment for Reinforcement Learning
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict, Optional, Union
from src.core.ldpc_code import LDPC_Code
from src.core.reward import ImprovedRewardFunction


class BaseEnvironment(ABC):
    """
    Abstract base class defining the standard RL environment interface.
    """
    
    @abstractmethod
    def reset(self, *args, **kwargs) -> np.ndarray:
        """
        Resets the environment to an initial state.
        
        Returns:
            np.ndarray: Initial state observation
        """
        pass
    
    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Executes one time step within the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        pass
    
    @abstractmethod
    def get_state(self) -> np.ndarray:
        """
        Returns the current state observation.
        
        Returns:
            np.ndarray: Current state vector
        """
        pass
    
    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimensionality of the state space."""
        pass
    
    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimensionality of the action space."""
        pass
    
    def _check_constraints(self) -> bool:
        """
        Evaluates constraint violations that would terminate the episode.
        
        Returns:
            bool: True if any constraint is violated, False otherwise
        """
        return False


class LDPCEnvironment(BaseEnvironment):
    """
    The LDPC decoding environment for RL-based LDPC decoding.
    """
    def __init__(self, code: Optional[LDPC_Code] = None, max_steps=10, 
                 max_flips_vector=6,
                 reward_function=None,
                 forbid_immediate_reflip: bool = False):
        self.code = code or LDPC_Code.from_default()
        self.max_steps = max_steps
        self.max_flips_vector = max_flips_vector
        # Flipping the bit that was just flipped restores the codeword exactly, so
        # the pair costs two steps and can never change the syndrome. Off by default
        # because it changes the action set, and so the MDP: every configuration
        # written before this existed keeps the behaviour it was run with.
        self.forbid_immediate_reflip = forbid_immediate_reflip

        self.reward_function = reward_function or ImprovedRewardFunction(code=self.code)

        # State components
        self.codeword = None
        self.original_codeword = None
        self.syndrome = None
        self.flip_vector = None
        self.penalty_vector = None
        self.step_count = 0

        # NEW: Lightweight counters (avoid repeated np.sum calls)
        self._syndrome_weight = 0      # Count of 1s in syndrome
        self._flip_count = 0           # Count of 1s in flip_vector
        self._penalty_sum = 0          # Sum of penalty_vector values

        # NEW: Preallocated state buffer (avoid repeated allocations in get_state)
        self._state_buf = None  # Will be initialized in reset()       

        # NEW: Cached legal actions list (static for a given code)
        self._legal_actions = list(range(self.code.action_dim))

        # Bit flipped on the previous step, or None at the start of an episode and
        # whenever the restriction is off.
        self._last_flip = None

        # One list and one mask per bit, built once, so narrowing the action set
        # costs a lookup rather than an allocation on every step. Only built when
        # the restriction is on: for n=128 this is 128 lists of 128 entries.
        if self.forbid_immediate_reflip:
            n_actions = self.code.action_dim
            self._legal_without = [
                self._legal_actions[:k] + self._legal_actions[k + 1:]
                for k in range(self.code.n)
            ]
            self._mask_without = []
            for k in range(self.code.n):
                mask = np.ones(n_actions, dtype=bool)
                mask[k] = False
                self._mask_without.append(mask)
        else:
            self._legal_without = None
            self._mask_without = None

        # Incidence structure of H, cached once, for the incremental syndrome and
        # penalty updates in step(). _h_col_rows[j] holds the parity checks that bit j
        # participates in; _h_row_cols[i] holds the bits that check i covers.
        H = self.code.H
        self._h_col_rows = [np.flatnonzero(H[:, j]) for j in range(self.code.n)]
        self._h_row_cols = [np.flatnonzero(H[i, :]) for i in range(self.code.m)]

    def reset(self, noisy_codeword, original_codeword):
        """
        Resets the environment with noisy and original codewords.
        
        Args:
            noisy_codeword: The received noisy codeword to decode
            original_codeword: The original codeword for BER calculation
        """
        self.codeword = np.copy(noisy_codeword)
        self.original_codeword = np.copy(original_codeword)
        self.syndrome = self.code.syndrome(self.codeword)
        self.flip_vector = np.zeros(self.code.n, dtype=int)
        self.penalty_vector = self.code.penalty_vector(self.syndrome)
        self.step_count = 0

        # NEW: Initialize counters from computed vectors
        self._syndrome_weight = int(np.sum(self.syndrome))
        self._flip_count = 0  # flip_vector is all zeros
        self._penalty_sum = int(np.sum(self.penalty_vector))
        self._last_flip = None  # No action has been taken yet in this episode

        # NEW: Preallocate state buffer once per episode
        if self._state_buf is None or len(self._state_buf) != self.state_dim:
            self._state_buf = np.empty(self.state_dim, dtype=np.float32)

        return self.get_state()

    def compute_ber(self) -> float:
        """
        Compute Bit Error Rate between current and original codeword.
        
        Returns:
            float: BER value (bit_errors / total_bits)
        """
        bit_errors = np.sum(self.codeword != self.original_codeword)
        return bit_errors / len(self.codeword)

    def step(self, action):
        """
        Executes one time step within the environment.
        """
        # 1. Store state BEFORE action (Phase 1.2: only if reward function needs it)
        prev_state = None
        if self.reward_function.uses_prev_state:
            prev_state = self.get_state()

        # The restriction is a property of the environment, not a convention the
        # callers keep. Accepting the action here would let a caller that forgets
        # the mask train against a larger action set than evaluation offers, and
        # nothing downstream would say so. Every maintained caller consults
        # get_action_mask() first, so this cannot fire in the pipeline.
        if self.forbid_immediate_reflip and action == self._last_flip:
            raise ValueError(
                f"bit {action} was flipped on the previous step; flipping it back "
                f"restores the codeword exactly, and forbid_immediate_reflip "
                f"refuses it. Consult get_action_mask() or get_legal_actions() "
                f"before choosing an action.")

        # 2. Apply action and update environment
        self.step_count += 1

        if action < self.code.n:  # Bit flip action
            # Track OLD value before flip
            old_flip_value = self.flip_vector[action]

            self.codeword[action] ^= 1
            self.flip_vector[action] ^= 1

            # NEW: Incrementally update flip_count
            # If bit went 0→1: increment, if 1→0: decrement
            self._flip_count += (1 if old_flip_value == 0 else -1)

            # 3. Update syndrome and penalty incrementally.
            #
            # s = H @ c (mod 2), so flipping bit `action` toggles exactly the checks
            # in column `action` of H: s ^= H[:, action]. penalty = H.T @ s, so each
            # toggled check i shifts penalty by +/-1 on the bits it covers. Both are
            # exact integer identities -- bit-for-bit what a full recompute produces --
            # but touch ~4 checks and ~32 penalty entries instead of running two
            # sparse mat-vecs over all 512 nonzeros of H.
            syndrome = self.syndrome
            penalty = self.penalty_vector
            for check in self._h_col_rows[action]:
                delta = 1 if syndrome[check] == 0 else -1
                syndrome[check] ^= 1
                covered = self._h_row_cols[check]
                penalty[covered] += delta
                self._syndrome_weight += delta
                self._penalty_sum += delta * covered.size

            # Remember it so the next step can refuse to undo it.
            self._last_flip = action
        # STOP action leaves the codeword untouched, so the syndrome, penalty vector
        # and their counters are all unchanged -- no recompute needed.

        # OLD: Compute weights via np.sum
        # new_syndrome_weight = np.sum(self.syndrome)
        # new_flip_vector_weight = np.sum(self.flip_vector)

        # NEW: Use counters instead
        new_syndrome_weight = self._syndrome_weight
        new_flip_vector_weight = self._flip_count

        #success = (new_syndrome_weight == 0) and (new_flip_vector_weight <= self.max_flips_vector)
        if action >= self.code.n:  # STOP action
            success = (new_syndrome_weight == 0) and (new_flip_vector_weight <= self.max_flips_vector)
        else:  # Bit flip action
            success = False

        # 4. Determine termination based on new state and constraints
        #done = success or (action >= self.code.n) or self._check_constraints()
        done = (action >= self.code.n) or self._check_constraints()

        # 5. Get state AFTER action
        next_state = self.get_state()

        # 6. Compute reward R(s_t, a_t, s_{t+1}) (Phase 1.2: conditional prev_state usage)
        if self.reward_function.uses_prev_state:
            reward = self.reward_function.compute_reward(prev_state, action, next_state, done)
        else:
            # Reward function doesn't use prev_state, pass next_state as first arg for compatibility
            reward = self.reward_function.compute_reward(next_state, action, next_state, done)

        info = {
            'success': success,
            'steps_taken': self.step_count,
            'terminal_reason': self._get_terminal_reason(action, success, done),
            'ber': self.compute_ber() if done else None
        }
        
        return next_state, reward, done, info

    def get_state(self):
        """
        Constructs the state vector.
        Components: codeword + syndrome + flip_vector + penalty_vector + metadata.
        Metadata: [syndrome_weight, penalty_sum, step_count].
        """

        # OLD: Compute weights via np.sum
        # syndrome_weight = np.sum(self.syndrome)
        # penalty_sum = np.sum(self.penalty_vector)

        # OLD: Multiple allocations
        # meta = np.array([syndrome_weight, penalty_sum, self.step_count])
        # return np.concatenate([
        #     self.codeword,
        #     self.syndrome,
        #     self.flip_vector,
        #     self.penalty_vector,
        #     meta
        # ]).astype(np.float32)

        # NEW: Fill preallocated buffer in-place
        n = self.code.n
        m = self.code.m        
        # Fill buffer slices (no allocations, just memory copy)
        self._state_buf[0:n] = self.codeword
        self._state_buf[n:n+m] = self.syndrome
        self._state_buf[n+m:2*n+m] = self.flip_vector
        self._state_buf[2*n+m:3*n+m] = self.penalty_vector
        self._state_buf[-3] = self._syndrome_weight  # NEW: Use cached counters (O(1) instead of O(n))
        self._state_buf[-2] = self._penalty_sum      # NEW: Use cached counters (O(1) instead of O(n))
        self._state_buf[-1] = self.step_count
        
        # Return copy to maintain immutability
        return self._state_buf.copy()


    def extract_state_components(self, state: np.ndarray) -> Dict[str, Union[np.ndarray, int]]:
        """
        Extracts individual components from a state vector.
        
        Returns proper data types: integers for metadata components (syndrome_weight, 
        penalty_sum, step_count) and numpy arrays for vector components (codeword, 
        syndrome, flip_vector, penalty_vector).
        
        Args:
            state: State vector of length state_dimensions['total']
            
        Returns:
            Dictionary with scalar integers for metadata and arrays for vectors
        """
        if state.shape[-1] != self.code.state_dimensions['total']:
            raise ValueError(f"State length must be {self.code.state_dimensions['total']}, but got {state.shape[-1]}.")
        
        ranges = self.code.state_component_ranges
        components = {}
        
        for component, (start, end) in ranges.items():
            if component == 'metadata':
                continue  # Skip metadata as it's handled by individual components
            
            # Return scalars for size-1 components, arrays for larger components
            if end - start == 1:
                # Force int conversion for metadata components (syndrome_weight, penalty_sum, step_count)
                if component in ['syndrome_weight', 'penalty_sum', 'step_count']:
                    components[component] = int(state[start])
                else:
                    components[component] = state[start].item()  # True scalar extraction
            else:
                components[component] = state[..., start:end]  # Array extraction
        
        return components

    @property
    def state_dim(self) -> int:
        """Dimensionality of the state space."""
        return self.code.state_dimensions['total']

    @property  
    def action_dim(self) -> int:
        """Dimensionality of the action space."""
        return self.code.action_dim

    def get_state_component_dims(self) -> Dict[str, int]:
        """
        Extract state component dimensions.

        Returns:
            Dictionary mapping component names to their dimensions
        """
        return {
            'codeword': self.code.state_dimensions['codeword'],
            'syndrome': self.code.state_dimensions['syndrome'], 
            'flip_vector': self.code.state_dimensions['flip_vector'],
            'penalty_vector': self.code.state_dimensions['penalty_vector'],
            'metadata': self.code.state_dimensions['metadata']
        }

    def _check_constraints(self) -> bool:
        """
        Evaluates constraint violations that would terminate the episode.

        Returns:
            bool: True if any constraint is violated, False otherwise
        """
        # Current constraints (Phase 1.2: use cached _flip_count)
        if (self.step_count >= self.max_steps) or (self._flip_count > self.max_flips_vector):
            return True
            
        # Future extensible constraints can be added here:
        # - Maximum number of flips per bit
        # - Maximum total flips
        # - Stuck in local minimum detection
        # - Time-based constraints
        # - Memory constraints
        
        return False
    
    def get_legal_actions(self):
        """
        Returns list of legal actions for current state.

        The action space is n bit flips plus STOP. With
        `forbid_immediate_reflip` the bit flipped on the previous step is
        removed, because flipping it back restores the codeword exactly: the
        pair spends two of `max_steps` and leaves the syndrome untouched. STOP
        is never removed, so at least n actions always remain.

        Returns:
            list: Legal action indices. Cached, so callers must not mutate it.
        """
        if self._last_flip is None or not self.forbid_immediate_reflip:
            return self._legal_actions
        return self._legal_without[self._last_flip]

    def get_action_mask(self):
        """
        Boolean mask over the action space, or None when nothing is excluded.

        Carries the same restriction as `get_legal_actions` in the form a
        Q-value based agent can apply directly. None means every action is
        available, which lets a caller skip the masking work entirely.

        Returns:
            np.ndarray of bool with one entry per action, or None.
        """
        if self._last_flip is None or not self.forbid_immediate_reflip:
            return None
        return self._mask_without[self._last_flip]
    
    @property
    def state_components(self) -> Dict[str, Any]:
        """
        Read-only access to environment state components.
        
        Returns:
            Dict[str, Any]: Dictionary containing all state components
        """
        return {
            'original_codeword': self.original_codeword.copy() if self.original_codeword is not None else None,

            'codeword': self.codeword.copy() if self.codeword is not None else None,
            'syndrome': self.syndrome.copy() if self.syndrome is not None else None,
            'flip_vector': self.flip_vector.copy() if self.flip_vector is not None else None,
            'penalty_vector': self.penalty_vector.copy() if self.penalty_vector is not None else None,
            # Use the incrementally-maintained counters rather than re-summing, for
            # consistency with get_state() and step(), which already do.
            'syndrome_weight': self._syndrome_weight if self.syndrome is not None else None,
            'penalty_sum': self._penalty_sum if self.penalty_vector is not None else None,
            'step_count': self.step_count,            
        }

    def _get_terminal_reason(self, action, success, done):
        if not done:
            return 'ONGOING'
        elif action >= self.code.n:
            return 'SUCCESS_STOP' if success else 'FAILED_STOP'
        elif self.step_count >= self.max_steps:
            return 'MAX_STEPS'
        elif self._flip_count > self.max_flips_vector:  # Phase 1.2: use cached counter
            return 'MAX_FLIPS'
        else:
            return 'UNKNOWN'


