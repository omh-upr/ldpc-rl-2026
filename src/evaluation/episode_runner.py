"""
Episode Runner Module for RL LDPC Decoder

This module provides a general-purpose episode execution engine that manages
the lifecycle of running episodes while invoking metrics collection callbacks.
This refactors the episode execution logic from LDPCEvaluator into a reusable component.
"""

from typing import Dict, Any, List, Iterator, Optional, Union
import torch
import numpy as np
from src.evaluation.metrics_callbacks import MetricCollector, MetricsCollectorManager


class EpisodeRunner:
    """
    General-purpose episode execution engine with metrics collection support.
    
    This class is responsible for running episodes while managing the lifecycle
    of metrics collection callbacks. It separates the concerns of episode execution
    from metrics aggregation, providing a clean, reusable interface.
    """
    
    def __init__(self, env, collectors: List[MetricCollector] = None, config: dict = None):
        """
        Initialize the episode runner.
        
        Args:
            env: The environment to run episodes in
            collectors: List of MetricCollector instances for metrics tracking
            config: The experiment configuration
        """
        self.env = env
        self.metrics_manager = MetricsCollectorManager(collectors or [])
        self.config = config or {}
    
    def add_collector(self, collector: MetricCollector) -> None:
        """Add a metrics collector."""
        self.metrics_manager.add_collector(collector)
    
    def remove_collector(self, collector_name: str) -> bool:
        """Remove a metrics collector by name."""
        return self.metrics_manager.remove_collector(collector_name)
    
    def enable_collector(self, collector_name: str) -> bool:
        """Enable a specific collector."""
        return self.metrics_manager.enable_collector(collector_name)
    
    def disable_collector(self, collector_name: str) -> bool:
        """Disable a specific collector."""
        return self.metrics_manager.disable_collector(collector_name)
    
    def run_episode(self, agent, original_codeword: np.ndarray, 
                   noisy_codeword: np.ndarray, error_info: Dict[str, Any], 
                   episode_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run a single episode and collect metrics.
        
        Args:
            agent: The agent to run the episode with
            original_codeword: The original error-free codeword
            noisy_codeword: The noisy received codeword
            error_info: Dictionary with detailed error information
            episode_metadata: Optional metadata about the episode (phase, etc.)
            
        Returns:
            Dictionary containing episode results
        """
        # Initialize episode counter if needed (1-indexed)
        if not hasattr(self, '_episode_counter'):
            self._episode_counter = 1
        else:
            self._episode_counter += 1
            
        # Prepare episode metadata
        episode_data = episode_metadata or {}
        episode_data.update({
            'episode_id': self._episode_counter,
            'original_codeword': original_codeword,
            'noisy_codeword': noisy_codeword,
            'error_info': error_info
        })
        
        # Notify collectors of episode start
        self.metrics_manager.on_episode_begin(episode_data)
        
        # Reset agent for new episode with appropriate method
        if hasattr(agent, 'reset_with_oracle'):
            agent.reset_with_oracle(original_codeword)
        elif hasattr(agent, 'reset'):
            agent.reset()
        
        # Reset environment
        state = self.env.reset(noisy_codeword, original_codeword)
        done = False
        step_number = 1  # Start at 1 for 1-indexed step numbering
        
        # Episode loop
        while not done:
            # All agents now use the unified interface
            legal_actions = self.env.get_legal_actions()
            action = agent.select_action(state, training=False, legal_actions=legal_actions)
            
            # Execute step
            next_state, reward, done, info = self.env.step(action)
            
            # Prepare step data for collectors
            step_data = {
                'state': self.env.extract_state_components(state),
                'action': action,
                'reward': reward,
                'next_state': self.env.extract_state_components(next_state),
                'done': done,
                'info': info,
                'step_number': step_number,
                'legal_actions': legal_actions
            }
            
            # Add training update if available (from agent training step)
            if hasattr(agent, 'last_training_update') and agent.last_training_update is not None:
                step_data['training_update'] = agent.last_training_update
                agent.last_training_update = None  # Clear after use
            
            # Notify collectors of step completion
            self.metrics_manager.on_step_end(step_data)
            
            # Update for next step
            state = next_state
            step_number += 1  # Increment after recording, so next step will be step_number + 1
        
        # Prepare episode summary
        episode_summary = {
            'success': info.get('success', False),
            'terminal_reason': info.get('terminal_reason', 'unknown'),
        }
        
        # Notify collectors of episode completion
        self.metrics_manager.on_episode_end(episode_summary)
        
        return episode_summary
    
    def run_episodes_from_dataset(self, agent, data_iterator: Iterator[Dict[str, Any]], 
                                 agent_type: str, max_episodes: Optional[int] = None,
                                 episode_type: str = 'unknown') -> List[Dict[str, Any]]:
        """
        Run multiple episodes from a dataset iterator.
        
        Args:
            agent: The agent to run episodes with
            data_iterator: Iterator providing episode data samples
            agent_type: The type of agent being evaluated
            max_episodes: Maximum number of episodes to run (None for unlimited)
            episode_type: Type of episode ('train', 'val', 'eval', or 'unknown' if not specified)
            
        Returns:
            List of episode results
        """
        episode_results = []
        episode_count = 0
        
        # Reset all collectors
        self.metrics_manager.reset_all()
        
        for sample in data_iterator:
            if max_episodes is not None and episode_count >= max_episodes:
                break
            
            # Extract episode data from sample
            original_codeword = sample['original']
            noisy_codeword = sample['noisy']
            error_pattern = sample['error']
            
            # Construct error_info
            error_model = sample['meta']['error_model'].decode() if isinstance(sample['meta']['error_model'], bytes) else sample['meta']['error_model']
            
            # Extract parameters from curriculum config for the specific error model
            curriculum = self.config.get('dataset_creation', {}).get('curriculum', [])
            phase_idx = sample['phase'] - 1  # Convert to 0-based index
            parameters = {'num_errors': sample['meta']['num_errors']}
            
            if 0 <= phase_idx < len(curriculum):
                error_models = curriculum[phase_idx].get('error_models', [])
                # Find the matching error model for this specific error_model type
                for model_config in error_models:
                    if model_config.get('type') == error_model:
                        model_params = model_config.get('params', {})
                        parameters.update(model_params)
                        break
            
            error_info = {
                'error_model': error_model,
                'parameters': parameters,
                'error_pattern': error_pattern
            }
            
            # Prepare episode metadata
            metadata = {
                'phase': sample.get('phase', 1),
                'agent_type': agent_type,
                'episode_type': episode_type
            }
            
            # Run episode
            episode_result = self.run_episode(agent, original_codeword, noisy_codeword, error_info, metadata)
            episode_results.append(episode_result)
            episode_count += 1
        
        return episode_results
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get aggregated metrics from all collectors.
        
        Returns:
            Dictionary containing results from all enabled collectors
        """
        return self.metrics_manager.get_all_results()
    
    def reset_metrics(self) -> None:
        """Reset all metrics collectors."""
        self.metrics_manager.reset_all()
