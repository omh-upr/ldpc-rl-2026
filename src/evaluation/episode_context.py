"""
Episode Context Management

Provides clean context management for episode collection per agent,
ensuring proper collector reset and data isolation between agents.
"""

from typing import Dict, Any, Optional
from contextlib import contextmanager


class EpisodeContext:
    """
    Context manager for clean episode collection per agent.
    
    Handles collector lifecycle management, ensuring each agent's
    evaluation starts with clean collectors and proper cleanup.
    """
    
    def __init__(self, agent_name: str, collectors: Dict[str, Any]):
        """
        Initialize episode context for a specific agent.
        
        Args:
            agent_name: Name of the agent being evaluated
            collectors: Dictionary of metric collectors
        """
        self.agent_name = agent_name
        self.collectors = collectors
    
    def __enter__(self):
        """Enter context and prepare collectors for new agent."""
        self._reset_collectors_for_new_agent()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and perform final cleanup."""
        # No explicit cleanup needed as progressive saver handles data
        # This could be extended for additional cleanup if needed
        pass
    
    def collect_episode(self, episode: Dict[str, Any]):
        """
        Collect an episode through all configured collectors.
        
        Args:
            episode: Episode data dictionary
        """
        # Ensure episode has agent context
        episode.setdefault('agent_context', {})['agent_name'] = self.agent_name
        
        # Send episode to all collectors
        for collector in self.collectors.values():
            if hasattr(collector, 'on_episode_end'):
                collector.on_episode_end(episode)
    
    def _reset_collectors_for_new_agent(self):
        """Reset collectors for new agent evaluation."""
        for collector in self.collectors.values():
            if hasattr(collector, 'reset_for_new_agent'):
                collector.reset_for_new_agent(self.agent_name)
            elif hasattr(collector, 'reset'):
                collector.reset()
    
    def get_episode_count(self) -> int:
        """Get total number of episodes collected in this context."""
        total = 0
        for collector in self.collectors.values():
            if hasattr(collector, 'episodes'):
                total += len(collector.episodes)
        return total


@contextmanager
def agent_evaluation_context(agent_name: str, collectors: Dict[str, Any]):
    """
    Context manager factory for agent evaluation.
    
    Args:
        agent_name: Name of the agent being evaluated
        collectors: Dictionary of metric collectors
        
    Yields:
        EpisodeContext instance
    """
    context = EpisodeContext(agent_name, collectors)
    try:
        with context as ctx:
            yield ctx
    finally:
        # Additional cleanup if needed
        pass


class CollectorManager:
    """
    Helper class for managing multiple collectors across agent evaluations.
    
    Provides utilities for collector lifecycle management and data aggregation.
    """
    
    def __init__(self, collectors: Dict[str, Any]):
        """
        Initialize collector manager.
        
        Args:
            collectors: Dictionary of metric collectors
        """
        self.collectors = collectors
        self._agent_contexts = {}
    
    def create_agent_context(self, agent_name: str) -> EpisodeContext:
        """
        Create a new episode context for an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            EpisodeContext instance
        """
        context = EpisodeContext(agent_name, self.collectors)
        self._agent_contexts[agent_name] = context
        return context
    
    def get_agent_context(self, agent_name: str) -> Optional[EpisodeContext]:
        """Get existing context for an agent."""
        return self._agent_contexts.get(agent_name)
    
    def reset_all_collectors(self):
        """Reset all collectors to initial state."""
        for collector in self.collectors.values():
            if hasattr(collector, 'episodes'):
                collector.episodes.clear()
            if hasattr(collector, 'reset'):
                collector.reset()
    
    def get_all_episodes(self) -> Dict[str, Any]:
        """
        Get all episodes from all collectors.
        
        Returns:
            Dictionary with aggregated episode data
        """
        all_episodes = []
        
        for collector in self.collectors.values():
            if hasattr(collector, 'episodes'):
                all_episodes.extend(collector.episodes)
        
        return {'episodes': all_episodes}
    
    def get_collector_summary(self) -> Dict[str, int]:
        """
        Get summary of episode counts per collector.
        
        Returns:
            Dictionary with collector names and episode counts
        """
        summary = {}
        
        for name, collector in self.collectors.items():
            if hasattr(collector, 'episodes'):
                summary[name] = len(collector.episodes)
            else:
                summary[name] = 0
        
        return summary