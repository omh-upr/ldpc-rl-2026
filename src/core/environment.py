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
                 reward_function=None):
        self.code = code or LDPC_Code.from_default()
        self.max_steps = max_steps
        self.max_flips_vector = max_flips_vector

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

        # 3. Update state components after action
        self.syndrome = self.code.syndrome(self.codeword)
        self.penalty_vector = self.code.penalty_vector(self.syndrome)

        # NEW: Update counters from recomputed vectors
        self._syndrome_weight = int(np.sum(self.syndrome))
        self._penalty_sum = int(np.sum(self.penalty_vector))

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
        
        Returns:
            list: Legal action indices (cached, never changes for a given code)
        """
        # Action space is static: n bit flips + 1 STOP action
        # Cached once in __init__() to avoid repeated list allocations
        return self._legal_actions
    
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
            'syndrome_weight': np.sum(self.syndrome) if self.syndrome is not None else None,
            'penalty_sum': np.sum(self.penalty_vector) if self.penalty_vector is not None else None,
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


