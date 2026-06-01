
"""
Training Infrastructure for the RL LDPC Decoder

This module contains the LDPCTrainer which orchestrates the training process,
including the curriculum learning strategy.

Updated to use the modular metrics collection framework.
"""

import numpy as np
import gc
import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from src.data.robust_data_generator import RobustDataGenerator
from src.evaluation.metrics_callbacks import (
    MetricsCollectorManager,
    EpisodeDataCollector
)


class BaseTrainer(ABC):
    """
    Abstract base class defining the standard RL trainer interface.
    """
    
    def __init__(self, env, agent, replay_buffer, batch_size: int = 32):
        """
        Initialize the base trainer.
        
        Args:
            env: RL environment
            agent: RL agent
            replay_buffer: Experience replay buffer
            batch_size: Batch size for training updates
        """
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
    
    @abstractmethod
    def train_episode(self, **kwargs) -> Dict[str, Any]:
        """
        Executes a single training episode.
        
        Returns:
            Dict containing episode metrics
        """
        pass
    
    @abstractmethod
    def train(self, num_episodes: int, **kwargs) -> Dict[str, Any]:
        """
        Executes the full training process.
        
        Args:
            num_episodes: Number of episodes to train
            
        Returns:
            Dict containing training history and metrics
        """
        pass
    
    def save_checkpoint(self, path: str) -> None:
        """
        Saves training checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        raise NotImplementedError(
            "BaseTrainer.save_checkpoint() is not implemented. "
            "Use experiment-level persistence (MetricsManager/model saves) or "
            "override this method in a concrete trainer."
        )
    
    def load_checkpoint(self, path: str) -> None:
        """
        Loads training checkpoint.
        
        Args:
            path: Path to load checkpoint from
        """
        raise NotImplementedError(
            "BaseTrainer.load_checkpoint() is not implemented. "
            "Use agent/replay-buffer specific loaders or override this method "
            "in a concrete trainer."
        )
    


class DQNTrainer(BaseTrainer):
    """
    Standard DQN trainer with experience replay and target network updates.
    """
    
    def __init__(self, env, agent, replay_buffer, batch_size: int = 32, 
                 target_update_freq: int = 1000, min_replay_size: int = None):
        """
        Initialize DQN trainer.
        
        Args:
            env: RL environment
            agent: DQN agent
            replay_buffer: Experience replay buffer
            batch_size: Batch size for training updates
            target_update_freq: Frequency of target network updates
            min_replay_size: Minimum replay buffer size before training starts
        """
        super().__init__(env, agent, replay_buffer, batch_size)
        self.target_update_freq = target_update_freq
        self.min_replay_size = min_replay_size or batch_size
        self.training_steps = 0
    
    def train_episode(self, training: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Executes a single DQN training episode.
        
        Args:
            training: Whether to use training mode (exploration)
            
        Returns:
            Dict containing episode metrics
        """
        state = self.env.reset(**kwargs)
        done = False
        total_reward = 0
        episode_length = 0
        episode_loss = 0.0
        loss_count = 0
        
        while not done:
            # Agent selects action
            action = self.agent.select_action(state, training=training)
            next_state, reward, done, info = self.env.step(action)
            
            # Store experience
            if training:
                self.replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            episode_length += 1
            
            # Training update
            if training and len(self.replay_buffer) >= self.min_replay_size:
                batch = self.replay_buffer.sample(self.batch_size)
                # Unzip the batch
                states, actions, rewards, next_states, dones = zip(*batch)
                metrics = self.agent.update((states, actions, rewards, next_states, dones))
                episode_loss += metrics['loss']
                loss_count += 1
                self.training_steps += 1
        
        # Calculate average loss for this episode
        avg_loss = episode_loss / loss_count if loss_count > 0 else 0.0
        
        # Extract info metrics
        success = info.get('success', False)
        terminal_reason = info.get('terminal_reason', 'unknown')
        
        episode_data = {
            'reward': total_reward,
            'length': episode_length,
            'loss': avg_loss,
            'success': success,
            'terminal_reason': terminal_reason,
            'steps_taken': info.get('steps_taken', episode_length),
            'epsilon': getattr(self.agent, 'epsilon', 0.0)
        }
        
        return episode_data
    
    def train(self, num_episodes: int, eval_frequency: int = 2000, 
              eval_episodes: int = 20, **kwargs) -> Dict[str, Any]:
        """
        Executes the full DQN training process.
        
        Args:
            num_episodes: Number of training episodes
            eval_frequency: Frequency of evaluation runs
            eval_episodes: Number of episodes per evaluation
            
        Returns:
            Dict containing training history and metrics
        """
        print(f"Starting DQN training for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            # Training episode
            episode_data = self.train_episode(training=True, **kwargs)
            
            # Target network update
            if self.training_steps % self.target_update_freq == 0 and self.training_steps > 0:
                self.agent.update_target_network()
            
            # Periodic evaluation
            if (episode + 1) % eval_frequency == 0:
                eval_metrics = self._evaluate(eval_episodes, **kwargs)
                print(f"Episode {episode + 1}/{num_episodes}: "
                      f"Success Rate: {eval_metrics['success_rate']:.1%}, "
                      f"Avg Reward: {eval_metrics['avg_reward']:.2f}, "
                      f"Epsilon: {episode_data['epsilon']:.3f}")
        
        # Final evaluation
        final_eval = self._evaluate(eval_episodes * 2, **kwargs)
        
        # Training summary
        summary = {
            'total_episodes': num_episodes,
            'total_training_steps': self.training_steps,
            'final_epsilon': self.agent.epsilon,
            'final_evaluation': final_eval
        }
        
        print(f"\nDQN training complete. Final success rate: {final_eval['success_rate']:.1%}")
        return summary
    
    def _evaluate(self, num_episodes: int, **kwargs) -> Dict[str, float]:
        """
        Evaluates the agent performance without training.
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Dict containing evaluation metrics
        """
        self.agent.set_training_mode(False)
        
        eval_rewards = []
        eval_success = []
        eval_lengths = []
        
        for _ in range(num_episodes):
            episode_data = self.train_episode(training=False, **kwargs)
            eval_rewards.append(episode_data['reward'])
            eval_success.append(episode_data['success'])
            eval_lengths.append(episode_data['length'])
        
        self.agent.set_training_mode(True)
        
        return {
            'success_rate': np.mean(eval_success),
            'avg_reward': np.mean(eval_rewards),
            'avg_length': np.mean(eval_lengths),
            'num_episodes': num_episodes
        }

class LDPCTrainer(DQNTrainer):
    """
    LDPC-specific DQN trainer with curriculum learning.
    """
    
    def __init__(self, env, agent, replay_buffer, data_generator: RobustDataGenerator,
                 batch_size: int = 32, target_update_freq: int = 1000,
                 epsilon_phase_reset: bool = False, epsilon_decay=None,
                 epsilon_start=None, epsilon_min=None, epsilon_decay_type=None,
                 collectors: list = None, training_flush_saver=None,
                 update_frequency: int = 1, **kwargs):
        """
        Initialize LDPC trainer with curriculum learning using RobustDataGenerator.

        Args:
            env: LDPC environment
            agent: DQN agent
            replay_buffer: Experience replay buffer
            data_generator: RobustDataGenerator instance
            batch_size: Batch size for training
            target_update_freq: Target network update frequency
            epsilon_phase_reset: Whether to reset epsilon at each curriculum phase
            epsilon_decay: Either float (uniform) or list of floats (one per curriculum phase)
            epsilon_start: Either float (uniform) or list of floats for starting epsilon per phase
            epsilon_min: Either float (uniform) or list of floats for minimum epsilon per phase
            epsilon_decay_type: Either string (uniform) or list of strings for decay type per phase
            collectors: List of MetricCollector instances (uses defaults if None)
            training_flush_saver: TrainingFlushSaver instance for immediate disk writes on memory limit
            update_frequency: How often to perform gradient updates (every N environment steps, default: 1)
        """
        # Filter out kwargs that parent class doesn't accept
        parent_kwargs = {k: v for k, v in kwargs.items()
                        if k in ['min_replay_size']}  # Only accepted parent kwargs
        super().__init__(env, agent, replay_buffer, batch_size, target_update_freq, **parent_kwargs)
        self.data_generator = data_generator
        self.epsilon_phase_reset = epsilon_phase_reset

        # Store training flush saver for phase coordination
        self.training_flush_saver = training_flush_saver

        # Phase 1.1 Optimization: Update frequency control
        self.update_frequency = update_frequency
        self.total_env_steps = 0  # Track environment steps separately from training_steps

        # Store memory config for cleanup decisions
        self.memory_config = kwargs.get('memory_config', {})
        
        # Initialize metrics collection system
        if collectors is None:
            # Create default collectors with memory config from kwargs
            memory_config = self.memory_config
            max_episodes = memory_config.get('max_episodes_per_phase', None)
            flush_frequency = memory_config.get('auto_flush_frequency', 1000)
            
            collectors = [
                EpisodeDataCollector(
                    max_episodes=max_episodes,
                    memory_check_frequency=flush_frequency
                )
            ]
            
            if max_episodes:
                print(f"   Training collector memory limit: {max_episodes:,} episodes")
                print(f"   Training collector check frequency: every {flush_frequency:,} episodes")
        
        self.metrics_manager = MetricsCollectorManager(collectors)
        
        # Store initial epsilon for phase resets
        self.initial_epsilon = getattr(agent, 'epsilon', 1.0)
        
        # Get number of phases from the dataset dynamically
        self.num_phases = self.data_generator.get_num_phases()
        print(f"Detected {self.num_phases} phases in dataset")
        
        # Process epsilon decay parameter for dynamic number of phases
        if epsilon_decay is not None:
            if isinstance(epsilon_decay, (list, tuple)):
                if len(epsilon_decay) != self.num_phases:
                    print(f"Adjusting epsilon_decay from {len(epsilon_decay)} to {self.num_phases} values")
                    if len(epsilon_decay) > self.num_phases:
                        # Truncate the list
                        self.epsilon_decay_schedule = list(epsilon_decay[:self.num_phases])
                    else:
                        # Extend the list by repeating the last value (or default if empty)
                        if len(epsilon_decay) > 0:
                            last_value = epsilon_decay[-1]
                            self.epsilon_decay_schedule = list(epsilon_decay) + [last_value] * (self.num_phases - len(epsilon_decay))
                        else:
                            # Handle empty list by using agent's default
                            agent_decay = getattr(agent, 'epsilon_decay', 0.999)
                            self.epsilon_decay_schedule = [agent_decay] * self.num_phases
                    print(f"   New epsilon_decay schedule: {self.epsilon_decay_schedule}")
                else:
                    self.epsilon_decay_schedule = list(epsilon_decay)
            else:
                # Single value - apply to all phases
                self.epsilon_decay_schedule = [epsilon_decay] * self.num_phases
        else:
            # Use agent's current epsilon_decay for all phases
            agent_decay = getattr(agent, 'epsilon_decay', 0.999)
            self.epsilon_decay_schedule = [agent_decay] * self.num_phases
        
        # Set initial phase epsilon decay
        self.current_phase_epsilon_decay = self.epsilon_decay_schedule[0]
        self.agent.set_epsilon_decay(self.current_phase_epsilon_decay)
        
        # Process epsilon_start parameter for dynamic number of phases
        if epsilon_start is not None:
            if isinstance(epsilon_start, (list, tuple)):
                if len(epsilon_start) != self.num_phases:
                    print(f"Adjusting epsilon_start from {len(epsilon_start)} to {self.num_phases} values")
                    if len(epsilon_start) > self.num_phases:
                        self.epsilon_start_schedule = list(epsilon_start[:self.num_phases])
                    else:
                        if len(epsilon_start) > 0:
                            last_value = epsilon_start[-1]
                            self.epsilon_start_schedule = list(epsilon_start) + [last_value] * (self.num_phases - len(epsilon_start))
                        else:
                            agent_start = getattr(agent, 'epsilon_start', 1.0)
                            self.epsilon_start_schedule = [agent_start] * self.num_phases
                    print(f"   New epsilon_start schedule: {self.epsilon_start_schedule}")
                else:
                    self.epsilon_start_schedule = list(epsilon_start)
            else:
                self.epsilon_start_schedule = [epsilon_start] * self.num_phases
        else:
            agent_start = getattr(agent, 'epsilon_start', 1.0)
            self.epsilon_start_schedule = [agent_start] * self.num_phases
        
        # Process epsilon_min parameter for dynamic number of phases
        if epsilon_min is not None:
            if isinstance(epsilon_min, (list, tuple)):
                if len(epsilon_min) != self.num_phases:
                    print(f"Adjusting epsilon_min from {len(epsilon_min)} to {self.num_phases} values")
                    if len(epsilon_min) > self.num_phases:
                        self.epsilon_min_schedule = list(epsilon_min[:self.num_phases])
                    else:
                        if len(epsilon_min) > 0:
                            last_value = epsilon_min[-1]
                            self.epsilon_min_schedule = list(epsilon_min) + [last_value] * (self.num_phases - len(epsilon_min))
                        else:
                            agent_min = getattr(agent, 'epsilon_min', 0.05)
                            self.epsilon_min_schedule = [agent_min] * self.num_phases
                    print(f"   New epsilon_min schedule: {self.epsilon_min_schedule}")
                else:
                    self.epsilon_min_schedule = list(epsilon_min)
            else:
                self.epsilon_min_schedule = [epsilon_min] * self.num_phases
        else:
            agent_min = getattr(agent, 'epsilon_min', 0.05)
            self.epsilon_min_schedule = [agent_min] * self.num_phases
        
        # Process epsilon_decay_type parameter for dynamic number of phases
        if epsilon_decay_type is not None:
            if isinstance(epsilon_decay_type, (list, tuple)):
                if len(epsilon_decay_type) != self.num_phases:
                    print(f"Adjusting epsilon_decay_type from {len(epsilon_decay_type)} to {self.num_phases} values")
                    if len(epsilon_decay_type) > self.num_phases:
                        self.epsilon_decay_type_schedule = list(epsilon_decay_type[:self.num_phases])
                    else:
                        if len(epsilon_decay_type) > 0:
                            last_value = epsilon_decay_type[-1]
                            self.epsilon_decay_type_schedule = list(epsilon_decay_type) + [last_value] * (self.num_phases - len(epsilon_decay_type))
                        else:
                            agent_decay_type = getattr(agent, 'epsilon_decay_type', 'exponential')
                            self.epsilon_decay_type_schedule = [agent_decay_type] * self.num_phases
                    print(f"   New epsilon_decay_type schedule: {self.epsilon_decay_type_schedule}")
                else:
                    self.epsilon_decay_type_schedule = list(epsilon_decay_type)
            else:
                self.epsilon_decay_type_schedule = [epsilon_decay_type] * self.num_phases
        else:
            agent_decay_type = getattr(agent, 'epsilon_decay_type', 'exponential')
            self.epsilon_decay_type_schedule = [agent_decay_type] * self.num_phases
        
        # Set initial phase epsilon parameters
        self.current_phase_epsilon_start = self.epsilon_start_schedule[0]
        self.current_phase_epsilon_min = self.epsilon_min_schedule[0]
        self.current_phase_epsilon_decay_type = self.epsilon_decay_type_schedule[0]
        self.agent.set_epsilon_start(self.current_phase_epsilon_start)
        self.agent.set_epsilon_min(self.current_phase_epsilon_min)
        self.agent.set_epsilon_decay_type(self.current_phase_epsilon_decay_type)
        
    
    # Note: _create_curriculum_schedule method removed - curriculum is now embedded in RobustDataGenerator
    
    def _handle_phase_transition(self, old_phase: int, new_phase: int, episode_count: int, config: dict = None, external_metrics_manager = None):
        """
        Handle curriculum phase transitions with epsilon management and incremental saving.
        
        Args:
            old_phase: Previous curriculum phase
            new_phase: New curriculum phase
            episode_count: Current episode count
            config: Experiment configuration for saving
            external_metrics_manager: External metrics manager for saving partial results
        """
        print(f"\nTransitioning from Phase {old_phase} to Phase {new_phase} at episode {episode_count}")

        # Update training flush saver's current phase for new flush files
        if self.training_flush_saver is not None:
            self.training_flush_saver.set_phase(new_phase)
            print(f"   Updated flush saver to Phase {new_phase}")

        # Save and clear accumulated training data for the completed phase to free memory
        if config is not None and external_metrics_manager is not None:
            episode_collector = self.metrics_manager.get_collector('EpisodeDataCollector')
            if episode_collector and episode_collector.episodes:
                phase_data = episode_collector.get_results().copy()  # Get a copy of the data
                if phase_data:
                    # Save phase data using external metrics manager
                    saved_path = external_metrics_manager.save_training_partial(
                        config=config, 
                        training_data=phase_data, 
                        phase=old_phase,
                        clear_memory=False  # We'll clear manually for more control
                    )
                    print(f"   Saved Phase {old_phase} data ({len(phase_data)} episodes) to {saved_path.name}")
                    
                    # Clear the episode collector to free memory
                    episode_collector.reset()
                    
                    # Force memory cleanup
                    del phase_data
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(f"   Cleared Phase {old_phase} data from memory")
        
        # Update epsilon parameters for new phase (with bounds checking)
        phase_index = max(0, min(new_phase - 1, len(self.epsilon_decay_schedule) - 1))
        
        # Update epsilon decay
        old_decay = self.current_phase_epsilon_decay
        self.current_phase_epsilon_decay = self.epsilon_decay_schedule[phase_index]
        self.agent.set_epsilon_decay(self.current_phase_epsilon_decay)
        print(f"   Epsilon decay: {old_decay:.6f} → {self.current_phase_epsilon_decay:.6f}")
        
        # Update epsilon start
        old_start = self.current_phase_epsilon_start
        self.current_phase_epsilon_start = self.epsilon_start_schedule[phase_index]
        self.agent.set_epsilon_start(self.current_phase_epsilon_start)
        print(f"   Epsilon start: {old_start:.4f} → {self.current_phase_epsilon_start:.4f}")
        
        # Update epsilon min
        old_min = self.current_phase_epsilon_min
        self.current_phase_epsilon_min = self.epsilon_min_schedule[phase_index]
        self.agent.set_epsilon_min(self.current_phase_epsilon_min)
        print(f"   Epsilon min: {old_min:.4f} → {self.current_phase_epsilon_min:.4f}")
        
        # Update epsilon decay type
        old_decay_type = self.current_phase_epsilon_decay_type
        self.current_phase_epsilon_decay_type = self.epsilon_decay_type_schedule[phase_index]
        self.agent.set_epsilon_decay_type(self.current_phase_epsilon_decay_type)
        print(f"   Epsilon decay type: {old_decay_type} → {self.current_phase_epsilon_decay_type}")
        
        # Reset training step count for linear decay when transitioning phases
        if self.current_phase_epsilon_decay_type == "linear":
            self.agent.reset_training_step_count()
            print(f"   Reset training step count for linear decay")
        
        # Reset epsilon if requested (use current phase's epsilon_start)
        if self.epsilon_phase_reset and hasattr(self.agent, 'epsilon'):
            old_epsilon = self.agent.epsilon
            self.agent.epsilon = self.current_phase_epsilon_start
            print(f"   Epsilon reset: {old_epsilon:.4f} → {self.agent.epsilon:.4f}")
        
    def train_episode(self, sample: Dict[str, Any] = None, training: bool = True,
                    episode_type: str = None, **kwargs) -> Dict[str, Any]:
        """
        Executes a single LDPC training episode using RobustDataGenerator sample with metrics collection.

        Overrides DQNTrainer.train_episode with LDPC-specific sample requirement.

        Args:
            sample: Sample dictionary from RobustDataGenerator (contains 'original', 'noisy', 'error', 'meta').
                    Required for LDPC training — raises ValueError if None.
            training: Whether to use training mode
            episode_type: Override episode type ('train', 'val', or 'eval').
                        If None, auto-determined from training flag ('train' if True, 'val' if False).
            **kwargs: Additional arguments (for parent compatibility)

        Returns:
            Dict containing episode metrics
        """
        if sample is None:
            raise ValueError(
                "LDPCTrainer.train_episode requires a 'sample' argument from RobustDataGenerator"
            )
        # Extract data from RobustDataGenerator sample
        original_codeword = sample['original']
        noisy_codeword = sample['noisy']
        error_pattern = sample['error']
        metadata = sample['meta']
        num_errors = metadata['num_errors']
        error_model = metadata['error_model'].decode() if isinstance(metadata['error_model'], bytes) else metadata['error_model']
        phase = sample.get('phase', 1)
        
        # Extract parameters from curriculum config for the specific error model
        parameters = {'num_errors': num_errors}
        curriculum = self.data_generator.config.get('curriculum', [])
        phase_idx = phase - 1  # Convert to 0-based index
        
        if 0 <= phase_idx < len(curriculum):
            error_models = curriculum[phase_idx].get('error_models', [])
            # Find the matching error model for this specific error_model type
            for model_config in error_models:
                if model_config.get('type') == error_model:
                    model_params = model_config.get('params', {})
                    parameters.update(model_params)
                    break

        # Auto-determine episode_type if not provided based on training flag
        if episode_type is None:
            episode_type = 'train' if training else 'val'

        # Prepare episode metadata for collectors
        episode_metadata = {
            'episode_id': self.metrics_manager.get_collector('EpisodeDataCollector').__len__(),
            'episode_type': episode_type,
            'agent_type': 'agent',
            'phase': phase,
            'original_codeword': original_codeword,
            'noisy_codeword': noisy_codeword,
            'error_info': {
                'error_model': error_model,
                'parameters': parameters,
                'error_pattern': error_pattern
            }
        }
        
        # Only notify collectors for training episodes (not evaluation episodes)
        if training:
            self.metrics_manager.on_episode_begin(episode_metadata)
        
        # Reset environment with LDPC-specific data
        state = self.env.reset(noisy_codeword, original_codeword)
        done = False
        total_reward = 0
        episode_length = 0
        episode_loss = 0.0
        loss_count = 0
        training_steps = 0
        step_number = 0
        
        while not done:
            # Agent selects action
            action = self.agent.select_action(state, training=training)
            next_state, reward, done, info = self.env.step(action)
            
            # Prepare step data for collectors
            step_data = {
                'step_number': step_number,
                'state': self.env.extract_state_components(state),
                'next_state': self.env.extract_state_components(next_state),
                'action': action,
                'reward': reward,
                'done': done,
                'info': info,
            }
            
            # Add training update if available (from agent training step)
            if hasattr(self.agent, 'last_training_update') and self.agent.last_training_update is not None:
                step_data['training_update'] = self.agent.last_training_update
                self.agent.last_training_update = None  # Clear after use
            
            # Only notify collectors for training episodes (not evaluation episodes)
            if training:
                self.metrics_manager.on_step_end(step_data)
            
            # Store experience
            if training:
                self.replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            episode_length += 1
            step_number += 1
            
            # Training update (Phase 1.1: gated by update_frequency)
            if training and len(self.replay_buffer) >= self.min_replay_size:
                self.total_env_steps += 1

                # Only perform update every update_frequency steps
                if self.total_env_steps % self.update_frequency == 0:
                    # Phase 2 Optimization: Use optimized array sampling if available
                    if hasattr(self.replay_buffer, 'sample_arrays'):
                        # Use fast array sampling
                        batch_arrays = self.replay_buffer.sample_arrays(self.batch_size)
                        metrics = self.agent.update(batch_arrays)
                    else:
                        # Fallback to old method for compatibility
                        batch = self.replay_buffer.sample(self.batch_size)
                        states, actions, rewards, next_states, dones = zip(*batch)
                        metrics = self.agent.update((states, actions, rewards, next_states, dones))
                    # Training metrics are now automatically stored in agent.last_training_update

                    episode_loss += metrics['loss']
                    loss_count += 1
                    training_steps += 1
        
        # Calculate average loss
        avg_loss = episode_loss / loss_count if loss_count > 0 else 0.0
        
        # Extract episode metrics
        success = info.get('success', False)
        terminal_reason = info.get('terminal_reason', 'unknown')
        
        # Prepare episode summary for collectors
        episode_summary = {
            'reward': total_reward,
            'length': episode_length,
            'loss': avg_loss,
            'success': success,
            'terminal_reason': terminal_reason,
            'steps_taken': info.get('steps_taken', episode_length),
            'epsilon': getattr(self.agent, 'epsilon', 0.0),
            'num_errors': num_errors,
            'error_model': error_model,
            'phase': phase,
            'syndrome_weight': np.sum(self.env.syndrome),
            'flip_weight': np.sum(self.env.flip_vector),
            'hamming_weight': np.sum(self.env.codeword != self.env.original_codeword),
            'training_steps': training_steps,
            'ber': info.get('ber', 0.0)
        }
        
        # Only notify collectors for training episodes (not evaluation episodes)
        if training:
            self.metrics_manager.on_episode_end(episode_summary)
        
        return episode_summary
    
    def train(self, num_epochs: int, eval_frequency: int = 2000, 
              eval_episodes: int = 25, config: dict = None, external_metrics_manager = None, **kwargs) -> Dict[str, Any]:
        """
        Executes LDPC training using RobustDataGenerator with curriculum learning.
        
        Args:
            num_epochs: Number of training epochs (full passes through dataset)
            eval_frequency: Frequency of evaluation runs (in episodes)
            eval_episodes: Number of episodes per evaluation
            config: Experiment configuration for incremental saving
            external_metrics_manager: External metrics manager for saving partial results
            
        Returns:
            Dict containing training history and metrics
        """
        print(f"Starting LDPC training for {num_epochs} epochs using RobustDataGenerator...")
        
        # Initialize metrics collection for training
        self.metrics_manager.reset_all()
        
        # Track phases and basic counters (still needed for phase transitions)
        current_phase = 1
        total_training_steps = 0
        episode_count = 0
        
        for epoch in range(num_epochs):
            print(f"\nStarting epoch {epoch + 1}/{num_epochs}")
            
            # Iterate through training data
            for sample in self.data_generator.get_iterator('train'):
                # Get actual phase from dataset
                phase = sample['phase']
                
                # Check for phase transition
                if phase > current_phase:
                    self._handle_phase_transition(current_phase, phase, episode_count, config, external_metrics_manager)
                    current_phase = phase
                
                # Training episode with dataset sample (now handles metrics automatically)
                episode_data = self.train_episode(sample, training=True)
                
                total_training_steps += episode_data['training_steps']
                episode_count += 1
                
                # Target network update
                if total_training_steps % self.target_update_freq == 0 and total_training_steps > 0:
                    self.agent.update_target_network()
                
                # Periodic evaluation
                if episode_count % eval_frequency == 0:
                    eval_metrics = self._evaluate_dataset(eval_episodes, current_phase)
                    print(f"Episode {episode_count} (Epoch {epoch + 1}, Phase {current_phase}): "
                          f"Success Rate: {eval_metrics['success_rate']:.1%}, "
                          f"Avg Reward: {eval_metrics['avg_reward']:.2f}, "
                          f"Avg Steps: {eval_metrics['avg_length']:.1f}, "
                          f"Epsilon: {episode_data['epsilon']:.3f}")
        
        # Save any remaining training data from the final phase and cleanup memory
        if config is not None and external_metrics_manager is not None:
            episode_collector = self.metrics_manager.get_collector('EpisodeDataCollector')
            if episode_collector and episode_collector.episodes:
                final_phase_data = episode_collector.get_results().copy()
                if final_phase_data:
                    saved_path = external_metrics_manager.save_training_partial(
                        config=config, 
                        training_data=final_phase_data, 
                        phase=current_phase,
                        clear_memory=False
                    )
                    print(f"Saved final Phase {current_phase} data ({len(final_phase_data)} episodes) to {saved_path.name}")
                    
                    # Clear the final phase data but keep collectors for final evaluation
                    del final_phase_data
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print("Phase data cleanup completed")
        
        # Final evaluation (use all phases for comprehensive assessment) BEFORE getting collector results
        final_eval = self._evaluate_dataset(eval_episodes * 2, current_phase=None)
        
        # Get final metrics from collectors AFTER final evaluation
        collector_results = self.metrics_manager.get_all_results()
        
        # NOW do the final memory cleanup after everything is collected
        # Check if progressive saving is enabled (handle both old and new config structures)
        enable_progressive = False
        if hasattr(self, 'memory_config'):
            if 'training' in self.memory_config:
                # New hierarchical structure
                enable_progressive = self.memory_config.get('training', {}).get('enable_progressive_saving', False)
            else:
                # Legacy flat structure
                enable_progressive = self.memory_config.get('enable_progressive_saving', False)
        
        if enable_progressive:
            # With progressive saving, clear collectors completely after final eval
            episode_collector = self.metrics_manager.get_collector('EpisodeDataCollector')
            if episode_collector:
                episode_collector.reset()
                print(f"   Cleared {episode_collector.__class__.__name__} after final evaluation")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Final memory cleanup completed (collectors cleared)")
        
        # Training summary using collector results
        summary = {
            'total_epochs': num_epochs,
            'total_episodes': episode_count,
            'total_training_steps': total_training_steps,
            'final_epsilon': self.agent.epsilon,
            'final_evaluation': final_eval,
            'collector_results': collector_results
        }
        
        return summary
    
    def _evaluate_dataset(self, num_episodes: int, current_phase: int = None) -> Dict[str, float]:
        """
        Evaluates agent using test dataset from RobustDataGenerator (VALIDATION during training).
        
        Args:
            num_episodes: Number of evaluation episodes
            current_phase: If specified, evaluate only on this phase (1-indexed)
            
        Returns:
            Dict containing evaluation metrics
        """
        self.agent.set_training_mode(False)
        
        eval_rewards = []
        eval_success = []
        eval_lengths = []
        
        # Get phase-aware iterator
        if current_phase is not None:
            test_iterator = self.data_generator.get_phase_iterator(
                'test', phases=current_phase, max_samples=num_episodes
            )
        else:
            test_iterator = self.data_generator.get_iterator('test')
        
        episode_count = 0
        for sample in test_iterator:
            # Apply episode limit for non-phase-specific evaluation
            if current_phase is None and episode_count >= num_episodes:
                break
            episode_count += 1
                
            episode_data = self.train_episode(sample, training=False, episode_type='val')
            eval_rewards.append(episode_data['reward'])
            eval_success.append(episode_data['success'])
            eval_lengths.append(episode_data['length'])
        
        self.agent.set_training_mode(True)
        
        return {
            'success_rate': np.mean(eval_success) if eval_success else 0.0,
            'avg_reward': np.mean(eval_rewards) if eval_rewards else 0.0,
            'avg_length': np.mean(eval_lengths) if eval_lengths else 0.0,
            'num_episodes': len(eval_success)
        }
