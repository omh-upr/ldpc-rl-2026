"""
Modular Metrics Collection Framework for RL LDPC Decoder

This module implements a callback-based system for collecting training and evaluation
metrics. The framework provides separation of concerns, modularity, and extensibility
for metrics tracking in the LDPC decoding experiment.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import numpy as np
from collections import defaultdict


def compress_state(state_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert full state to compressed weight-based representation.
    
    Args:
        state_components: Dictionary containing state arrays (codeword, syndrome, etc.)
        
    Returns:
        Dictionary with compressed state weights
    """
    # Handle missing state components gracefully
    if not state_components:
        return {
            "codeword_weight": None,
            "syndrome_weight": None,
            "flip_weight": None,
            "penalty_sum": None,
            "step_count": None
        }
    
    return {
        "codeword_weight": int(np.sum(state_components["codeword"])) if "codeword" in state_components else None,
        "syndrome_weight": int(state_components["syndrome_weight"]) if "syndrome_weight" in state_components and state_components["syndrome_weight"] is not None else None,
        "flip_weight": int(np.sum(state_components["flip_vector"])) if "flip_vector" in state_components else None,
        "penalty_sum": int(state_components["penalty_sum"]) if "penalty_sum" in state_components and state_components["penalty_sum"] is not None else None,
        "step_count": int(state_components["step_count"]) if "step_count" in state_components and state_components["step_count"] is not None else None
    }


def compute_q_value_stats(q_values: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive Q-value statistics including min/max.
    
    Args:
        q_values: Array of Q-values
        
    Returns:
        Dictionary with mean, std, min, max statistics
    """
    return {
        "mean": float(np.mean(q_values)),
        "std": float(np.std(q_values)),
        "min": float(np.min(q_values)),
        "max": float(np.max(q_values))
    }


class MetricCollector(ABC):
    """
    Abstract base class for metrics collection callbacks.
    
    Defines the standard interface for collecting metrics at different points
    in the training/evaluation lifecycle. Inspired by callback patterns in
    modern ML frameworks like PyTorch Lightning and Keras.
    """
    
    def __init__(self, name: str = None):
        """
        Initialize the metric collector.
        
        Args:
            name: Optional name for the collector (defaults to class name)
        """
        self.name = name or self.__class__.__name__
        self.enabled = True
    
    def on_episode_begin(self, episode_data: Dict[str, Any]) -> None:
        """
        Called at the beginning of each episode.
        
        Args:
            episode_data: Dictionary containing episode metadata
                - episode_id: Episode number or identifier
                - phase: Curriculum phase (if applicable)
                - num_errors: Initial number of errors
                - error_model: Error model used
                - Any other episode initialization data
        """
        pass
    
    def on_step_end(self, step_data: Dict[str, Any]) -> None:
        """
        Called after each step within an episode.
        
        Args:
            step_data: Dictionary containing step information
                - state: Current state
                - action: Action taken
                - reward: Reward received
                - next_state: Next state
                - done: Whether episode is done
                - info: Additional environment info
                - step_number: Current step number
                - Any other step-specific data
        """
        pass
    
    def on_episode_end(self, episode_summary: Dict[str, Any]) -> None:
        """
        Called at the end of each episode.
        
        Args:
            episode_summary: Dictionary containing episode results
                - success: Whether episode was successful
                - reward: Total episode reward
                - length: Episode length in steps
                - ber: Bit error rate
                - syndrome_weight: Final syndrome weight
                - flip_weight: Final flip weight
                - terminal_reason: How episode ended
                - Any other episode outcome data
        """
        pass
    
    @abstractmethod
    def get_results(self) -> Dict[str, Any]:
        """
        Returns the collected metrics for this collector.
        
        Returns:
            Dictionary containing the aggregated metrics
        """
        pass
    
    def reset(self) -> None:
        """
        Reset the collector state for a new evaluation/training run.
        Should be called before starting a new set of episodes.
        """
        pass
    
    def enable(self) -> None:
        """Enable this collector."""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable this collector."""
        self.enabled = False


class EpisodeDataCollector(MetricCollector):
    """
    Collects raw, time-series data for each episode with memory-conscious features.
    """
    def __init__(self, name: str = "EpisodeDataCollector", max_episodes: int = None, 
                 flush_callback = None, memory_check_frequency: int = 1000):
        """
        Initialize the episode data collector with memory management features.
        
        Args:
            name: Name of the collector
            max_episodes: Maximum number of episodes to keep in memory (None = unlimited)
            flush_callback: Callback function to call when flushing episodes (episodes_data) -> None
            memory_check_frequency: Check memory limits every N episodes
        """
        super().__init__(name)
        self.max_episodes = max_episodes
        self.flush_callback = flush_callback
        self.memory_check_frequency = memory_check_frequency
        self.episode_count = 0
        self.reset()

    def on_episode_begin(self, episode_data: Dict[str, Any]) -> None:
        self.current_episode = {
            "episode_id": episode_data.get("episode_id"),
            "episode_type": episode_data.get("episode_type"),
            "agent_type": episode_data.get("agent_type"),
            "phase": episode_data.get("phase"),
            "original_codeword": episode_data.get("original_codeword"),
            "noisy_codeword": episode_data.get("noisy_codeword"),
            "error_info": episode_data.get("error_info"),
            "steps": [],
        }

    def on_step_end(self, step_data: Dict[str, Any]) -> None:
        # Compress states for efficient storage
        state_compressed = compress_state(step_data.get("state", {}))
        next_state_compressed = compress_state(step_data.get("next_state", {})) if step_data.get("next_state") else None
        
        step_record = {
            "step": step_data.get("step_number"),
            "state": state_compressed,
            "next_state": next_state_compressed,
            "action": step_data.get("action"),
            "reward": step_data.get("reward"),
            "done": step_data.get("done"),
            "success": step_data.get("info", {}).get("success")
        }
        
        # Add training update if present (nested within step)
        if "training_update" in step_data:
            step_record["training_update"] = step_data["training_update"]
        
        self.current_episode["steps"].append(step_record)

    def on_episode_end(self, episode_summary: Dict[str, Any]) -> None:
        self.current_episode["terminal_reason"] = episode_summary.get("terminal_reason")
        self.episodes.append(self.current_episode)
        self.episode_count += 1
        
        # Check memory limits and flush if needed
        if self.max_episodes is not None and self.episode_count % self.memory_check_frequency == 0:
            if len(self.episodes) >= self.max_episodes:
                self._flush_episodes()
    
    def _flush_episodes(self) -> None:
        """Flush episodes to free memory, calling the flush callback if provided."""
        if self.flush_callback and self.episodes:
            # Make a copy for the callback
            episodes_to_flush = self.episodes.copy()
            self.flush_callback(episodes_to_flush)
            print(f"   Flushed {len(episodes_to_flush)} episodes from {self.name}")
        
        # Clear the episodes to free memory
        self.episodes.clear()
        print(f"   Cleared {self.name} memory")
    
    def force_flush(self) -> List[Dict[str, Any]]:
        """Force flush all episodes and return them (for manual memory management)."""
        episodes_data = self.episodes.copy() if self.episodes else []
        if episodes_data:
            self._flush_episodes()
        return episodes_data

    def get_results(self) -> List[Dict[str, Any]]:
        return self.episodes

    def reset(self) -> None:
        self.episodes = []
        self.current_episode = {}
        self.episode_count = 0
    
    def __len__(self) -> int:
        """Return the number of completed episodes."""
        return len(self.episodes)



class MetricsCollectorManager:
    """
    Manages a collection of MetricCollector instances and coordinates their lifecycle.
    
    This class provides a unified interface for managing multiple collectors,
    invoking their callbacks, and aggregating their results.
    """
    
    def __init__(self, collectors: List[MetricCollector] = None):
        """
        Initialize the metrics manager.
        
        Args:
            collectors: List of MetricCollector instances to manage
        """
        self.collectors = collectors or []
        self.enabled = True
    
    def add_collector(self, collector: MetricCollector) -> None:
        """Add a collector to the manager."""
        self.collectors.append(collector)
    
    def remove_collector(self, collector_name: str) -> bool:
        """Remove a collector by name. Returns True if found and removed."""
        for i, collector in enumerate(self.collectors):
            if collector.name == collector_name:
                del self.collectors[i]
                return True
        return False
    
    def get_collector(self, collector_name: str) -> Optional[MetricCollector]:
        """Get a collector by name."""
        for collector in self.collectors:
            if collector.name == collector_name:
                return collector
        return None
    
    def enable_collector(self, collector_name: str) -> bool:
        """Enable a specific collector. Returns True if found."""
        collector = self.get_collector(collector_name)
        if collector:
            collector.enable()
            return True
        return False
    
    def disable_collector(self, collector_name: str) -> bool:
        """Disable a specific collector. Returns True if found."""
        collector = self.get_collector(collector_name)
        if collector:
            collector.disable()
            return True
        return False
    
    def on_episode_begin(self, episode_data: Dict[str, Any]) -> None:
        """Invoke on_episode_begin for all enabled collectors."""
        if not self.enabled:
            return
        for collector in self.collectors:
            if collector.enabled:
                collector.on_episode_begin(episode_data)
    
    def on_step_end(self, step_data: Dict[str, Any]) -> None:
        """Invoke on_step_end for all enabled collectors."""
        if not self.enabled:
            return
        for collector in self.collectors:
            if collector.enabled:
                collector.on_step_end(step_data)
    
    def on_episode_end(self, episode_summary: Dict[str, Any]) -> None:
        """Invoke on_episode_end for all enabled collectors."""
        if not self.enabled:
            return
        for collector in self.collectors:
            if collector.enabled:
                collector.on_episode_end(episode_summary)
    
    def get_all_results(self) -> Dict[str, Any]:
        """Get results from all collectors."""
        results = {}
        for collector in self.collectors:
            if collector.enabled:
                collector_results = collector.get_results()
                if collector_results:  # Only include non-empty results
                    results[collector.name] = collector_results
        return results
    
    def get_all_results_snapshot(self) -> Dict[str, Any]:
        """Get a deep copy snapshot of all collector results."""
        import copy
        results = {}
        for collector in self.collectors:
            if collector.enabled:
                collector_results = collector.get_results()
                if collector_results:  # Only include non-empty results
                    # Deep copy to prevent reference issues when collectors are reset
                    results[collector.name] = copy.deepcopy(collector_results)
        return results
    
    def reset_all(self) -> None:
        """Reset all collectors."""
        for collector in self.collectors:
            collector.reset()
    
    def enable_all(self) -> None:
        """Enable the manager and all collectors."""
        self.enabled = True
        for collector in self.collectors:
            collector.enable()
    
    def disable_all(self) -> None:
        """Disable the manager (stops all callback invocations)."""
        self.enabled = False
