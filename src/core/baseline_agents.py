"""
Baseline Agents for LDPC Decoding

Non-learning baseline agents for LDPC decoding evaluation.
All baseline agents inherit from the unified BaseAgent interface.
"""

import numpy as np
from typing import Optional, List
from src.advanced.gallager_b import GallagerBDecoder
from src.advanced.min_sum_core import MinSumDecoder
from src.core.ldpc_code import LDPC_Code
from src.core.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """
    Baseline agent that selects actions randomly.
    """
    
    def __init__(self, code: Optional[LDPC_Code] = None):
        """
        Initialize RandomAgent.
        
        Args:
            code: LDPC code instance for dynamic dimensions
        """
        self.code = code or LDPC_Code.from_default()
        self._action_dim = self.code.action_dim
        
        # Cache action range for performance - avoids repeated list/array creation
        self._action_range = np.arange(self._action_dim)
    
    @property
    def action_dim(self) -> int:
        """The dimensionality of the action space."""
        return self._action_dim

    def select_action(self, state: np.ndarray, training: bool = False, **kwargs) -> int:
        """
        Select a random action from available actions.
        
        Args:
            state: Environment state vector (unused)
            training: Training mode flag (unused for random selection)
            **kwargs: Additional parameters (legal_actions, etc.)
            
        Returns:
            Randomly selected action index
        """
        legal_actions = kwargs.get('legal_actions', None)
        # Fast path: unconstrained random selection (most common case)
        if legal_actions is None:
            # Use np.random.randint for maximum speed - avoids array creation
            return np.random.randint(0, self._action_dim)
        
        # Slower path: constrained selection
        # Convert to numpy array if it's a list for better performance
        if isinstance(legal_actions, list):
            if len(legal_actions) == 0:
                # No legal actions - this shouldn't happen but handle gracefully
                return np.random.randint(0, self._action_dim)
            legal_actions = np.array(legal_actions)
        
        return np.random.choice(legal_actions)


class GreedyAgent(BaseAgent):
    """
    Baseline agent that greedily flips the bit with highest penalty.
    """
    
    def __init__(self, code: Optional[LDPC_Code] = None):
        """
        Initialize GreedyAgent.
        
        Args:
            code: LDPC code instance for dynamic dimensions
        """
        self.code = code or LDPC_Code.from_default()
        self._action_dim = self.code.action_dim
        self.num_bits = self.code.n
        
        # Cache frequently accessed values for performance
        self._penalty_range = self.code.state_component_ranges['penalty_vector']
        self._penalty_start = self._penalty_range[0]
        self._penalty_end = self._penalty_range[1]
    
    @property
    def action_dim(self) -> int:
        """The dimensionality of the action space."""
        return self._action_dim

    def select_action(self, state: np.ndarray, training: bool = False, **kwargs) -> int:
        """
        Select the bit with highest penalty value to flip.
        
        Args:
            state: Environment state vector
            training: Training mode flag (unused for greedy selection)
            **kwargs: Additional parameters (legal_actions, etc.)
            
        Returns:
            Action index for bit with highest penalty, or STOP if no penalties
        """
        legal_actions = kwargs.get('legal_actions', None)
        # Extract penalty vector using cached ranges for performance
        penalty_vector = state[self._penalty_start:self._penalty_end]
        
        # Fast path: no legal actions constraint
        if legal_actions is None:
            # If no penalties, return STOP
            if np.sum(penalty_vector) == 0:
                return self.num_bits  # STOP action
            return np.argmax(penalty_vector)
        
        # Slower path: apply legal actions mask
        # Make a copy only when we need to modify the penalty vector
        penalty_vector = penalty_vector.copy()
        
        # Vectorized legal action masking for performance
        if len(legal_actions) > 0:
            # Filter legal actions: separate bit flips from STOP action
            legal_actions_array = np.array(legal_actions)
            bit_flip_actions = legal_actions_array[legal_actions_array < self.num_bits]
            stop_action_allowed = self.num_bits in legal_actions_array
            
            if len(bit_flip_actions) > 0:
                # Create boolean mask for bit flip actions only
                legal_mask = np.zeros(self.num_bits, dtype=bool)
                legal_mask[bit_flip_actions] = True
                # Set illegal bit flip actions to -1 using vectorized indexing
                penalty_vector[~legal_mask] = -1
            else:
                # No legal bit flip actions, only STOP allowed
                penalty_vector[:] = -1
            
            # If only STOP action is legal or no positive penalties, return STOP
            if not stop_action_allowed and np.max(penalty_vector) <= 0:
                return self.num_bits  # STOP action
            elif len(bit_flip_actions) == 0:  # Only STOP action allowed
                return self.num_bits  # STOP action
        else:
            # No legal actions available
            return self.num_bits  # STOP action
        
        # If no legal actions have positive penalties, return STOP
        if np.max(penalty_vector) <= 0:
            return self.num_bits  # STOP action

        return np.argmax(penalty_vector)


class SimpleClassicalAgent(BaseAgent):
    """
    Simple classical LDPC decoder using basic iterative approach.
    
    This agent provides a simplified version of classical LDPC decoding
    without the complexity of full message passing algorithms.
    """
    
    def __init__(self, code: Optional[LDPC_Code] = None, max_iterations: int = 3):
        """
        Initialize SimpleClassicalAgent.
        
        Args:
            code: LDPC code instance for dynamic dimensions
            max_iterations: Maximum iterations for decoding attempt
        """
        self.code = code or LDPC_Code.from_default()
        self._action_dim = self.code.action_dim
        self.num_bits = self.code.n
        self.max_iterations = max_iterations
    
    @property
    def action_dim(self) -> int:
        """The dimensionality of the action space."""
        return self._action_dim

    def select_action(self, state: np.ndarray, training: bool = False, **kwargs) -> int:
        """
        Select action using simple classical decoding heuristic.
        
        Combines penalty information with syndrome pattern to make
        more informed decisions than pure greedy approach.
        
        Args:
            state: Environment state vector
            training: Training mode flag (unused for classical decoding)
            **kwargs: Additional parameters (legal_actions, etc.)
            
        Returns:
            Action index for selected bit flip, or STOP
        """
        legal_actions = kwargs.get('legal_actions', None)
        # Extract state components using dynamic ranges
        ranges = self.code.state_component_ranges
        syndrome = state[ranges['syndrome'][0]:ranges['syndrome'][1]].astype(int)
        penalty_vector = state[ranges['penalty_vector'][0]:ranges['penalty_vector'][1]].copy()
        
        # If no syndrome violations, return STOP
        if np.sum(syndrome) == 0:
            return self.num_bits  # STOP action
            
        # Compute enhanced bit scores combining penalty and syndrome information
        bit_scores = self._compute_bit_scores(syndrome, penalty_vector)
        
        # Apply legal actions mask if provided
        if legal_actions is not None:
            for bit in range(self.num_bits):
                if bit not in legal_actions:
                    bit_scores[bit] = -1
        
        # If no legal actions have positive scores, return STOP
        if np.max(bit_scores) <= 0:
            return self.num_bits  # STOP action

        return np.argmax(bit_scores)
    
    def _compute_bit_scores(self, syndrome: np.ndarray, penalty_vector: np.ndarray) -> np.ndarray:
        """
        Compute enhanced bit scores for action selection.
        
        Combines penalty information with syndrome pattern analysis
        to identify most promising bit flips.
        
        Args:
            syndrome: Syndrome vector indicating violated checks
            penalty_vector: Penalty values for each bit
            
        Returns:
            Score vector for each bit (higher = more likely to be error)
        """
        # Start with penalty-based scores
        bit_scores = penalty_vector.copy()
        
        # Enhance scores for bits involved in multiple violations
        # This simple heuristic favors bits that appear in many violated checks
        violation_bonus = 0.5
        for i in range(len(bit_scores)):
            if bit_scores[i] > 0:  # Only boost bits that already have penalties
                bit_scores[i] += violation_bonus * min(bit_scores[i], 3)  # Cap the bonus
        
        return bit_scores


class MinSumAgent(BaseAgent):
    """
    Min-Sum algorithm agent for LDPC decoding.
    
    Implements the Min-Sum algorithm, a simplified version of Belief Propagation
    that provides near-optimal performance for LDPC decoding. Uses iterative
    message passing to estimate bit reliabilities and selects actions based
    on least reliable bits.
    """
    
    def __init__(self, 
                 code: Optional[LDPC_Code] = None,
                 max_iterations: int = 20,
                 damping_factor: float = 0.5,
                 reliability_threshold: float = 0.5):
        """
        Initialize MinSumAgent.
        
        Args:
            code: LDPC code instance for dynamic dimensions
            max_iterations: Maximum iterations for Min-Sum algorithm
            damping_factor: Damping factor for message passing stability
            reliability_threshold: Threshold for determining unreliable bits
        """
        self.code = code or LDPC_Code.from_default()
        self._action_dim = self.code.action_dim
        self.num_bits = self.code.n
        self.reliability_threshold = reliability_threshold
        
        # Initialize Min-Sum decoder
        self.decoder = MinSumDecoder(
            H_matrix=self.code.H,
            max_iterations=max_iterations,
            damping_factor=damping_factor
        )
    
    @property
    def action_dim(self) -> int:
        """The dimensionality of the action space."""
        return self._action_dim
        
    def select_action(self, state: np.ndarray, training: bool = False, **kwargs) -> int:
        """
        Select action using Min-Sum algorithm.
        
        Runs Min-Sum message passing to estimate bit reliabilities and selects
        the least reliable bit for flipping. Returns STOP action if syndrome
        is already zero or all bits are sufficiently reliable.
        
        Args:
            state: Environment state vector  
            training: Training mode flag (unused for Min-Sum algorithm)
            **kwargs: Additional parameters (legal_actions, etc.)
            
        Returns:
            Action index for least reliable bit, or STOP action
        """
        legal_actions = kwargs.get('legal_actions', None)
        # Extract state components using dynamic ranges
        ranges = self.code.state_component_ranges
        codeword = state[ranges['codeword'][0]:ranges['codeword'][1]].astype(np.int8)
        syndrome = state[ranges['syndrome'][0]:ranges['syndrome'][1]].astype(np.int8)
        
        # If syndrome is already zero, return STOP
        if np.sum(syndrome) == 0:
            return self.num_bits  # STOP action
        
        # Run Min-Sum algorithm to get bit reliabilities
        try:
            bit_reliabilities = self.decoder.get_bit_reliabilities(codeword, syndrome)
        except Exception as e:
            # Fallback to penalty-based selection if Min-Sum fails
            penalty_vector = state[ranges['penalty_vector'][0]:ranges['penalty_vector'][1]]
            if np.sum(penalty_vector) == 0:
                return self.num_bits  # STOP action
            
            # Apply legal actions mask if provided
            if legal_actions is not None:
                masked_penalties = penalty_vector.copy()
                for bit in range(self.num_bits):
                    if bit not in legal_actions:
                        masked_penalties[bit] = -1
                if np.max(masked_penalties) <= 0:
                    return self.num_bits
                return np.argmax(masked_penalties)
            
            return np.argmax(penalty_vector)
        
        # Select least reliable bit (lowest reliability score)
        unreliable_bits = np.where(bit_reliabilities < self.reliability_threshold)[0]
        
        if len(unreliable_bits) == 0:
            # All bits are reliable, return STOP
            return self.num_bits
        
        # Apply legal actions mask if provided
        if legal_actions is not None:
            # Filter unreliable bits by legal actions
            legal_unreliable = [bit for bit in unreliable_bits if bit in legal_actions]
            if len(legal_unreliable) == 0:
                return self.num_bits  # No legal unreliable bits
            unreliable_bits = legal_unreliable
        
        # Among unreliable bits, select the least reliable one
        least_reliable_idx = np.argmin(bit_reliabilities[unreliable_bits])
        return unreliable_bits[least_reliable_idx]
    
    def get_algorithm_info(self) -> dict:
        """Return information about the Min-Sum configuration."""
        base_info = self.decoder.get_algorithm_info()
        base_info.update({
            'agent_type': 'MinSumAgent',
            'reliability_threshold': self.reliability_threshold,
            'action_dim': self._action_dim
        })
        return base_info
    
    def diagnose_performance(self, 
                           state: np.ndarray, 
                           true_errors: Optional[np.ndarray] = None) -> dict:
        """
        Diagnose Min-Sum performance on given state.
        
        Args:
            state: Environment state vector
            true_errors: Known error positions for analysis
            
        Returns:
            Dictionary with diagnostic information
        """
        ranges = self.code.state_component_ranges
        codeword = state[ranges['codeword'][0]:ranges['codeword'][1]].astype(np.int8)
        syndrome = state[ranges['syndrome'][0]:ranges['syndrome'][1]].astype(np.int8)
        
        # Run Min-Sum decoding
        bit_reliabilities, decoded_codeword, converged = self.decoder.decode(codeword, syndrome)
        
        # Compute action that would be selected
        selected_action = self.select_action(state)
        
        diagnostics = {
            'converged': converged,
            'syndrome_weight': np.sum(syndrome),
            'avg_reliability': np.mean(bit_reliabilities),
            'min_reliability': np.min(bit_reliabilities),
            'max_reliability': np.max(bit_reliabilities),
            'selected_action': selected_action,
            'decoded_syndrome_weight': np.sum(self.decoder.compute_syndrome(decoded_codeword))
        }
        
        if true_errors is not None:
            error_positions = np.where(true_errors == 1)[0]
            if len(error_positions) > 0:
                error_reliabilities = bit_reliabilities[error_positions]
                diagnostics.update({
                    'avg_error_reliability': np.mean(error_reliabilities),
                    'min_error_reliability': np.min(error_reliabilities),
                    'error_positions': error_positions.tolist(),
                    'selected_action_is_error': selected_action in error_positions if selected_action < self.num_bits else False
                })
        
        return diagnostics


class MinSumExpertAgent(BaseAgent):
    """
    Min-Sum Expert Agent for episode-level planning.
    
    Unlike MinSumAgent which runs Min-Sum at every step, this agent:
    1. Runs Min-Sum once per episode when receiving first state
    2. Creates action sequence based on bit reliabilities (least reliable first)
    3. Executes actions sequentially by popping from internal list
    4. Always adds STOP action for proper episode termination
    
    Agent is expected to be reset/recreated between episodes by the caller.
    """
    
    def __init__(self, 
                 code: Optional[LDPC_Code] = None,
                 max_iterations: int = 20,
                 damping_factor: float = 0.8,
                 reliability_threshold: float = 0.5):
        """
        Initialize MinSumExpertAgent.
        
        Args:
            code: LDPC code instance for dynamic dimensions
            max_iterations: Maximum iterations for Min-Sum algorithm
            damping_factor: Damping factor for message passing
            reliability_threshold: Threshold for unreliable bits
        """
        self.code = code or LDPC_Code.from_default()
        self.num_bits = self.code.n
        self._action_dim = self.code.action_dim
        
        # Min-Sum decoder configuration with proper saturation
        self.decoder = MinSumDecoder(
            H_matrix=self.code.H,
            max_iterations=max_iterations,
            damping_factor=damping_factor,
            convergence_threshold=1e-4,  # More relaxed threshold
            max_magnitude_threshold=50.0  # Prevent numerical overflow
        )
        self.reliability_threshold = reliability_threshold
        
        # Episode state management
        self._action_list = []
        self._episode_planned = False
    
    @property
    def action_dim(self) -> int:
        """The dimensionality of the action space."""
        return self._action_dim
    
    def select_action(self, state: np.ndarray, training: bool = False, **kwargs) -> int:
        """
        Select action using episode-level Min-Sum planning.
        
        On first call, runs Min-Sum algorithm and plans entire episode.
        Subsequent calls pop actions from the planned sequence.
        
        Args:
            state: Environment state vector
            training: Training mode flag (unused)
            **kwargs: Additional parameters
            
        Returns:
            Selected action (bit index or STOP action)
        """
        # Plan episode on first call
        if not self._episode_planned:
            self._plan_episode_actions(state)
            self._episode_planned = True
        
        # Pop next action from list (FIFO)
        if len(self._action_list) > 0:
            return self._action_list.pop(0)
        else:
            # Fallback: return STOP if list is empty
            return self.num_bits
    
    def _plan_episode_actions(self, state: np.ndarray):
        """
        Plan all actions for the episode using Min-Sum algorithm.
        
        Args:
            state: Initial state vector for the episode
        """
        # Extract codeword and syndrome from state using robust component ranges
        ranges = self.code.state_component_ranges
        codeword = state[ranges['codeword'][0]:ranges['codeword'][1]].astype(np.int8)
        syndrome = state[ranges['syndrome'][0]:ranges['syndrome'][1]].astype(np.int8)
        
        self._action_list = []  # Start with empty list
        
        # Only run Min-Sum if syndrome is non-zero (has errors)
        if np.sum(syndrome) > 0:
            try:
                # Run Min-Sum algorithm
                bit_reliabilities, decoded_codeword, converged = self.decoder.decode(codeword, syndrome)
                
                if converged:
                    # Verify decoded codeword satisfies parity checks
                    decoded_syndrome = self.decoder.compute_syndrome(decoded_codeword)
                    if np.sum(decoded_syndrome) == 0:
                        # Compare input codeword with decoded result to find flipped bits
                        flipped_bits = np.where(codeword != decoded_codeword)[0]
                        if len(flipped_bits) > 0:
                            # Sort by least reliable first (ascending |LLR|)
                            order = np.argsort(np.abs(bit_reliabilities[flipped_bits]))
                            self._action_list = flipped_bits[order].tolist()
                
                # If Min-Sum doesn't converge, provide no actions - teacher is not confident
                
            except Exception:
                # Min-Sum failed, keep empty action list
                pass
        
        # Always add STOP action at the end, regardless of what happened above
        self._action_list.append(self.num_bits)
    
    def reset(self):
        """
        Reset the agent for a new episode.
        
        Clears the planned action list and episode state.
        """
        self._action_list = []
        self._episode_planned = False
    
    def get_action_list(self) -> List[int]:
        """
        Get the current action list (for debugging/analysis).
        
        Returns:
            Copy of the current action list
        """
        return self._action_list.copy()


class GallagerBAgent(BaseAgent):
    """
    Gallager's hard-decision algorithm B, scored through the environment.

    Decodes the received word once with ``GallagerBDecoder`` (see
    ``src/advanced/gallager_b.py`` for the algorithm, the majority threshold and the
    decision rule) and plans the episode from the result, exactly as
    ``MinSumExpertAgent`` does: when the decision has a zero syndrome, flip every bit
    where it differs from the received word, in ascending index order, then STOP;
    when it does not converge, STOP at once. Flips are of distinct bits, so the
    re-flip restriction never binds, and the environment's step and flip budgets
    apply as to every other decoder.

    ``last_iterations`` holds the iterations the native decoder ran on the latest
    episode, for cost accounting. Agent is expected to be reset between episodes by
    the caller.
    """

    def __init__(self,
                 code: Optional[LDPC_Code] = None,
                 max_iterations: int = 20,
                 thresholds: Optional[dict] = None):
        """
        Args:
            code: LDPC code instance for dynamic dimensions
            max_iterations: Iteration limit of the native decoder
            thresholds: Optional {variable degree: b}; the default is the majority
                threshold ceil(degree / 2)
        """
        self.code = code or LDPC_Code.from_default()
        self.num_bits = self.code.n
        self._action_dim = self.code.action_dim
        self.decoder = GallagerBDecoder(self.code.H, max_iterations=max_iterations,
                                        thresholds=thresholds)
        self._action_list = []
        self._episode_planned = False
        self.last_iterations = 0
        self.last_converged = False

    @property
    def action_dim(self) -> int:
        """The dimensionality of the action space."""
        return self._action_dim

    def select_action(self, state: np.ndarray, training: bool = False, **kwargs) -> int:
        """Plan on the first call of an episode, then pop the planned actions."""
        if not self._episode_planned:
            self._plan_episode_actions(state)
            self._episode_planned = True
        if self._action_list:
            return self._action_list.pop(0)
        return self.num_bits

    def _plan_episode_actions(self, state: np.ndarray):
        ranges = self.code.state_component_ranges
        codeword = state[ranges['codeword'][0]:ranges['codeword'][1]].astype(np.int64)
        decision, converged, iterations = self.decoder.decode(codeword)
        self.last_iterations = iterations
        self.last_converged = converged
        self._action_list = []
        if converged:
            self._action_list = np.flatnonzero(decision != codeword).tolist()
        self._action_list.append(self.num_bits)

    def reset(self):
        """Reset the agent for a new episode."""
        self._action_list = []
        self._episode_planned = False


class OmniscientAgent(BaseAgent):
    """
    Omniscient Expert Agent for episode-level planning with oracle knowledge.
    
    This agent provides optimal teaching demonstrations by leveraging knowledge
    of the original (error-free) codeword. It demonstrates diverse solution paths
    by shuffling the order of error corrections, helping RL agents learn pattern
    recognition beyond greedy strategies.
    
    Key features:
    1. Requires original codeword via reset_with_oracle() method before each episode
    2. Plans episode actions by comparing current state with oracle knowledge
    3. Shuffles error bit corrections for diverse teaching demonstrations
    4. Always terminates with STOP action for proper episode completion
    
    Agent must be reset with original codeword between episodes by the caller.
    """
    
    def __init__(self, code: Optional[LDPC_Code] = None):
        """
        Initialize OmniscientAgent.
        
        Args:
            code: LDPC code instance for dynamic dimensions
        """
        self.code = code or LDPC_Code.from_default()
        self.num_bits = self.code.n
        self._action_dim = self.code.action_dim
        
        # Episode state management
        self._action_list = []
        self._episode_planned = False
        self._original_codeword = None
    
    @property
    def action_dim(self) -> int:
        """The dimensionality of the action space."""
        return self._action_dim
    
    def select_action(self, state: np.ndarray, training: bool = False, **kwargs) -> int:
        """
        Select action using omniscient episode-level planning.
        
        On first call, compares current state with oracle knowledge to plan
        optimal correction sequence. Subsequent calls pop actions from the
        randomized planned sequence.
        
        Args:
            state: Environment state vector
            training: Training mode flag (unused)
            **kwargs: Additional parameters
            
        Returns:
            Selected action (bit index or STOP action)
            
        Raises:
            RuntimeError: If original codeword not provided via reset_with_oracle()
        """
        if self._original_codeword is None:
            raise RuntimeError(
                "OmniscientAgent requires original codeword. "
                "Call reset_with_oracle(original_codeword) before using agent."
            )
        
        # Plan episode on first call
        if not self._episode_planned:
            self._plan_episode_actions(state)
            self._episode_planned = True
        
        # Pop next action from list (FIFO)
        if len(self._action_list) > 0:
            return self._action_list.pop(0)
        else:
            # Fallback: return STOP if list is empty
            return self.num_bits
    
    def _plan_episode_actions(self, state: np.ndarray):
        """
        Plan all actions for the episode using oracle knowledge.
        
        Compares current codeword with original (error-free) codeword to
        identify error positions, then creates a randomized correction
        sequence for diverse teaching demonstrations.
        
        Args:
            state: Initial state vector for the episode
        """
        # Extract current codeword from state
        current_codeword = self._extract_codeword(state)
        
        self._action_list = []  # Start with empty list
        
        # Find error positions by comparing with oracle knowledge
        error_positions = np.where(current_codeword != self._original_codeword)[0]
        
        if len(error_positions) > 0:
            # Shuffle error positions for diverse teaching demonstrations
            # This helps RL agents learn pattern recognition beyond greedy approaches
            error_positions_shuffled = error_positions.copy()
            np.random.shuffle(error_positions_shuffled)
            
            # Add shuffled error corrections to action list
            self._action_list = error_positions_shuffled.tolist()
        
        # Always add STOP action at the end for proper episode termination
        self._action_list.append(self.num_bits)
    
    def _extract_codeword(self, state: np.ndarray) -> np.ndarray:
        """
        Extract codeword from state vector.
        
        Args:
            state: Environment state vector
            
        Returns:
            Current codeword as numpy array
        """
        return state[:self.num_bits].astype(np.int8)
    
    def reset_with_oracle(self, original_codeword: np.ndarray):
        """
        Reset the agent for a new episode with oracle knowledge.
        
        This method MUST be called with the original (error-free) codeword
        before each episode to enable omniscient planning.
        
        Args:
            original_codeword: Original error-free codeword for oracle knowledge.
                             Must be provided for proper agent functionality.
                             
        Raises:
            ValueError: If original_codeword has incorrect dimensions
        """
        # Validate codeword dimensions
        if len(original_codeword) != self.num_bits:
            raise ValueError(
                f"Original codeword length {len(original_codeword)} does not match "
                f"expected length {self.num_bits}"
            )
        
        # Store oracle knowledge and reset episode state
        self._original_codeword = original_codeword.astype(np.int8)
        self._action_list = []
        self._episode_planned = False
    
    def get_action_list(self) -> List[int]:
        """
        Get the current action list (for debugging/analysis).
        
        Returns:
            Copy of the current action list
        """
        return self._action_list.copy()
    
    def has_oracle_knowledge(self) -> bool:
        """
        Check if agent has been provided with oracle knowledge.
        
        Returns:
            True if original codeword has been set, False otherwise
        """
        return self._original_codeword is not None
