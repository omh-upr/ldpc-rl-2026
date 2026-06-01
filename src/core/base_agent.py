"""
Base Agent Interface for LDPC Decoding

This module provides the fundamental interface that all agents must implement,
establishing the root of the agent hierarchy for unified polymorphic handling.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """
    The fundamental interface for any agent that can select an action.
    
    This is the root of the agent hierarchy, defining the absolute minimum
    interface common to every agent type in the system.
    """
    
    @abstractmethod
    def select_action(self, state: np.ndarray, training: bool = False) -> int:
        """
        Selects an action given a state.
        
        Args:
            state: Current state observation
            training: Whether agent is in training mode (affects exploration)
            
        Returns:
            int: Selected action
        """
        pass
    
    @property
    @abstractmethod  
    def action_dim(self) -> int:
        """The dimensionality of the action space."""
        pass