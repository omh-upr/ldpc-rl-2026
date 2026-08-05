"""
Learning Agents for LDPC Decoding

This module provides learning agents that can be trained via experience replay.
All learning agents inherit from the root BaseAgent interface.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import warnings
from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Optional
from src.core.networks import SimpleMLP, LDPCOptimizedMLP
from src.core.network_factory import create_network
from src.core.base_agent import BaseAgent
from src.evaluation.metrics_callbacks import compute_q_value_stats


class LearningAgent(BaseAgent):
    """
    Abstract base class for agents that learn from experience.
    
    This class extends the basic BaseAgent interface with learning-specific
    methods for training, model persistence, and introspection.
    """
    
    @abstractmethod
    def update(self, batch: Any) -> Dict[str, float]:
        """
        Updates the agent using a batch of experiences.
        
        Args:
            batch: Batch of experiences (format depends on agent type)
            
        Returns:
            Dict[str, float]: Training metrics (loss, etc.)
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Saves the agent's model and configuration.
        
        Args:
            path: Path to save the model
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Loads the agent's model and configuration.
        
        Args:
            path: Path to load the model from
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
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Returns Q-values for all actions given a state (for analysis).
        
        Args:
            state: Current state observation
            
        Returns:
            np.ndarray: Q-values for all actions
        """
        raise NotImplementedError("Q-values not available for this agent type")
    
    def set_training_mode(self, training: bool) -> None:
        """
        Sets the agent's training mode.
        
        Args:
            training: Whether to set training mode (True) or evaluation mode (False)
        """
        pass
    
    def get_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        """
        Returns action probabilities for analysis.
        
        Args:
            state: Current state observation
            
        Returns:
            np.ndarray: Probability distribution over actions
        """
        raise NotImplementedError("Action probabilities not available for this agent type")

class DQNAgent(LearningAgent):
    """
    A DQN agent for LDPC decoding with configurable network architecture.
    """
    def __init__(self, state_dim: int, action_dim: int,
                 learning_rate=1e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_decay=0.999995, epsilon_min=0.05,
                 epsilon_decay_type="exponential",
                 network_config: Optional[Dict[str, Any]] = None,
                 weight_decay: float = 0.0,
                 device: Optional[str] = None,
                 strict_device: bool = True):
        
        self._state_dim = state_dim
        self._action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start  # Store for potential resets
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_decay_type = epsilon_decay_type.lower()
        
        # Validate decay type
        if self.epsilon_decay_type not in ["exponential", "linear"]:
            raise ValueError(f"epsilon_decay_type must be 'exponential' or 'linear', got '{epsilon_decay_type}'")

        # Validate decay value makes sense for the chosen type
        if self.epsilon_decay_type == "linear" and not isinstance(self.epsilon_decay, list) and self.epsilon_decay > 0.1:
            warnings.warn(
                f"Linear epsilon_decay={self.epsilon_decay} is unusually large "
                f"(epsilon drops by this amount per step). "
                f"Typical linear decay values are 1e-4 to 1e-2. "
                f"Did you mean to use epsilon_decay_type='exponential'?",
                UserWarning,
                stacklevel=2
            )

        # Training step counter for linear decay
        self.training_step_count = 0

        if device is None:
            # No device requested: auto-select, never an error.
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            requested_device = torch.device(device)
            if requested_device.type == "cuda" and not torch.cuda.is_available():
                # A CUDA device was explicitly requested but is unavailable. Silently
                # continuing on CPU produces results indistinguishable from a GPU run
                # while taking several times longer, so surface it loudly.
                detail = (
                    f"Requested device '{device}' is unavailable: "
                    f"torch.cuda.is_available() is False "
                    f"(torch {torch.__version__}, built for CUDA {torch.version.cuda})."
                )
                if strict_device:
                    raise RuntimeError(
                        detail + " Refusing to fall back to CPU silently. Either fix the "
                        "environment (see README section 6.1), set device: \"cpu\" "
                        "explicitly, or set strict_device: false in the experiment config "
                        "to allow the fallback."
                    )
                warnings.warn(
                    detail + " Falling back to CPU; run times are NOT comparable to "
                    "GPU runs.",
                    RuntimeWarning,
                    stacklevel=2
                )
                self.device = torch.device("cpu")
            else:
                self.device = requested_device

        # Use network factory for configurable architecture
        if network_config is None:
            # Default to SimpleMLP for backward compatibility
            network_config = {'type': 'SimpleMLP'}

        self.network_config = network_config  # Persist for checkpoint reproducibility

        network_type = network_config.get('type', 'SimpleMLP')
        network_params = {k: v for k, v in network_config.items() if k != 'type'}
        
        self.network = create_network(
            network_type=network_type,
            network_config=network_params,
            state_dim=state_dim,
            action_dim=action_dim
        ).to(self.device)

        self.target_network = create_network(
            network_type=network_type,
            network_config=network_params,
            state_dim=state_dim,
            action_dim=action_dim
        ).to(self.device)

        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.loss_fn = nn.MSELoss()
        
        # For storing training updates to be integrated into step data
        self.last_training_update = None

    def select_action(self, state: np.ndarray, training: bool = True, **kwargs) -> int:
        """
        Selects an action using an epsilon-greedy policy.
        
        Args:
            state: Current state observation
            training: Whether agent is in training mode (affects exploration)
            **kwargs: Additional parameters (ignored by DQN agent)
            
        Returns:
            int: Selected action
        """
        if training and np.random.rand() <= self.epsilon:
            return np.random.randint(self._action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.network(state_tensor)
            return q_values.argmax().item()

    def update(self, batch) -> Dict[str, float]:
        """
        Updates the agent's network using a batch of experiences.

        Args:
            batch: Batch of experiences (states, actions, rewards, next_states, dones)
                   Can be either tuple of lists (old format) or tuple of arrays (Phase 2 optimized)

        Returns:
            Dict[str, float]: Training metrics
        """
        states, actions, rewards, next_states, dones = batch

        # Phase 2 Optimization: Handle both numpy arrays and lists
        # If already numpy arrays (from sample_arrays), avoid np.array() call
        if isinstance(states, np.ndarray):
            # Optimized path: already numpy arrays, use torch.from_numpy for zero-copy
            states = torch.from_numpy(states).to(self.device)
            next_states = torch.from_numpy(next_states).to(self.device)
        else:
            # Legacy path: convert from lists
            states = torch.FloatTensor(np.array(states)).to(self.device)
            next_states = torch.FloatTensor(np.array(next_states)).to(self.device)

        # Actions, rewards, dones - handle both formats
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).unsqueeze(1).to(self.device)
            rewards = torch.from_numpy(rewards).unsqueeze(1).to(self.device)
            dones = torch.from_numpy(dones).unsqueeze(1).to(self.device)
        else:
            actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Get current Q values
        q_values = self.network(states).gather(1, actions)

        # Get next Q values from target network
        next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
        
        # Compute the expected Q values
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        loss = self.loss_fn(q_values, expected_q_values.detach())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon based on decay type
        self.training_step_count += 1
        
        if self.epsilon_decay_type == "linear":
            # Linear decay: epsilon decreases by fixed amount per step
            decay_amount = self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon - decay_amount)
        else:
            # Exponential decay (current behavior)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        
        # Store training update for step-level integration
        self.last_training_update = {
            "loss": float(loss.item()),
            "epsilon": self.epsilon,
            "q_values": compute_q_value_stats(q_values.detach().cpu().numpy().flatten()),
            "target_q_values": compute_q_value_stats(expected_q_values.detach().cpu().numpy().flatten())
        }
            
        return {
            "loss": loss.item(),
            "epsilon": self.epsilon,
            "q_values_mean": q_values.mean().item(),
            "target_q_mean": expected_q_values.mean().item()
        }

    def update_target_network(self):
        """
        Copies the weights from the main network to the target network.
        """
        self.target_network.load_state_dict(self.network.state_dict())

    @property
    def state_dim(self) -> int:
        """Dimensionality of the state space."""
        return self._state_dim

    @property  
    def action_dim(self) -> int:
        """Dimensionality of the action space."""
        return self._action_dim

    def save(self, path: str) -> None:
        """
        Saves the agent's model and configuration.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'epsilon_start': self.epsilon_start,
            'state_dim': self._state_dim,
            'action_dim': self._action_dim,
            'gamma': self.gamma,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay_type': self.epsilon_decay_type,
            'training_step_count': self.training_step_count,
            'network_config': self.network_config,
        }, path)

    def load(self, path: str) -> None:
        """
        Loads the agent's model and configuration.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        
        # Load new parameters with backward compatibility
        self.epsilon_decay_type = checkpoint.get('epsilon_decay_type', 'exponential')
        self.training_step_count = checkpoint.get('training_step_count', 0)

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Returns Q-values for all actions given a state (for analysis).
        
        Args:
            state: Current state observation
            
        Returns:
            np.ndarray: Q-values for all actions
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.network(state_tensor)
            return q_values.detach().cpu().numpy().flatten()

    def set_training_mode(self, training: bool) -> None:
        """
        Sets the agent's training mode.
        
        Args:
            training: Whether to set training mode (True) or evaluation mode (False)
        """
        self.network.train(training)
        self.target_network.train(False)  # Target network always in eval mode
    
    def set_epsilon_decay(self, epsilon_decay: float) -> None:
        """
        Set epsilon decay rate externally (used for curriculum learning).
        
        Args:
            epsilon_decay: New epsilon decay rate
        """
        self.epsilon_decay = epsilon_decay
    
    def set_epsilon_start(self, epsilon_start: float) -> None:
        """
        Set starting epsilon value externally (used for curriculum learning).
        
        Args:
            epsilon_start: New starting epsilon value for phase resets
        """
        self.epsilon_start = epsilon_start
    
    def set_epsilon_min(self, epsilon_min: float) -> None:
        """
        Set minimum epsilon value externally (used for curriculum learning).
        
        Args:
            epsilon_min: New minimum epsilon threshold
        """
        self.epsilon_min = epsilon_min
    
    def set_epsilon_decay_type(self, epsilon_decay_type: str) -> None:
        """
        Set epsilon decay type externally (used for curriculum learning).
        
        Args:
            epsilon_decay_type: New decay type ("exponential" or "linear")
        """
        epsilon_decay_type = epsilon_decay_type.lower()
        if epsilon_decay_type not in ["exponential", "linear"]:
            raise ValueError(f"epsilon_decay_type must be 'exponential' or 'linear', got '{epsilon_decay_type}'")
        self.epsilon_decay_type = epsilon_decay_type
    
    def reset_training_step_count(self) -> None:
        """
        Reset training step counter (useful for phase transitions with linear decay).
        """
        self.training_step_count = 0
