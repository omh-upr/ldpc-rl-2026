"""
Simple Agent Registry for Expert Experience Generation

Lightweight registry focused on instantiating agents for expert experience generation.
Unlike the main AgentRegistry, this is designed for simplicity and single-purpose use.
"""

from typing import Dict, Type, Any, Optional
from src.core.base_agent import BaseAgent
from src.core.baseline_agents import RandomAgent, GreedyAgent, MinSumAgent, MinSumExpertAgent, GallagerBAgent, OmniscientAgent
from src.core.ldpc_code import LDPC_Code


class SimpleAgentRegistry:
    """
    Lightweight agent registry for expert experience generation.
    
    Provides a simple interface for instantiating agents that conform to
    the BaseAgent interface for use in experience generation.
    """
    
    def __init__(self):
        """Initialize the registry with available agent classes."""
        self.agent_classes: Dict[str, Type[BaseAgent]] = {
            'random': RandomAgent,
            'greedy': GreedyAgent, 
            'min_sum': MinSumAgent,
            'min_sum_expert': MinSumExpertAgent,
            'gallager_b': GallagerBAgent,
            'omniscient': OmniscientAgent
        }
    
    def get_agent_class(self, agent_type: str) -> Optional[Type[BaseAgent]]:
        """
        Get agent class by type name.
        
        Args:
            agent_type: String identifier for agent type
            
        Returns:
            Agent class or None if not found
        """
        return self.agent_classes.get(agent_type)
    
    def create_agent(self, agent_type: str, code: LDPC_Code, **agent_params) -> BaseAgent:
        """
        Create and initialize an agent instance.
        
        Args:
            agent_type: String identifier for agent type
            code: LDPC code instance
            **agent_params: Additional parameters for agent initialization
            
        Returns:
            Initialized agent instance
            
        Raises:
            ValueError: If agent type is not found
            RuntimeError: If agent creation fails
        """
        agent_class = self.get_agent_class(agent_type)
        if agent_class is None:
            available_types = list(self.agent_classes.keys())
            raise ValueError(f"Agent type '{agent_type}' not found. Available types: {available_types}")
        
        try:
            agent = agent_class(code=code, **agent_params)
            return agent
        except Exception as e:
            raise RuntimeError(f"Failed to create agent '{agent_type}': {e}") from e
    
    def list_available_agents(self) -> list[str]:
        """
        Get list of available agent types.
        
        Returns:
            List of agent type strings
        """
        return list(self.agent_classes.keys())
    
    def validate_agent_type(self, agent_type: str) -> bool:
        """
        Check if agent type is valid.
        
        Args:
            agent_type: String identifier to validate
            
        Returns:
            True if valid, False otherwise
        """
        return agent_type in self.agent_classes