
"""
Training Infrastructure for the RL LDPC Decoder

This module contains the LDPCTrainer which orchestrates the training process,
including the curriculum learning strategy.

Updated to use the modular metrics collection framework.
"""

import numpy as np
import gc
import sys
import torch
from abc import ABC, abstractmethod
from collections import defaultdict

from src.training import epsilon_schedule
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
            # Agent selects action. The mask is None unless the environment
            # restricts the action set, in which case the agent honours it in
            # both the greedy and the exploring branch.
            action = self.agent.select_action(
                state, training=training,
                action_mask=self.env.get_action_mask())
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

                # Synchronize on the update that reaches the interval, not on the
                # episode that happens to end there. Episodes have variable length,
                # so a boundary test can step straight over a multiple and miss the
                # synchronization entirely.
                self.training_steps += 1
                if self.training_steps % self.target_update_freq == 0:
                    self.agent.update_target_network()

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

            # Target network synchronization happens inside train_episode, on the
            # update that reaches the interval.

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
                 update_frequency: int = 1, epsilon_episode_schedule=None,
                 **kwargs):
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
            epsilon_episode_schedule: Optional callable (epoch, phase, index_in_phase) -> epsilon.
                When given, epsilon is set from progress through the phase at the
                start of every episode instead of decaying per gradient update.
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

        # Episode-progress epsilon scheduling. Counted per (epoch, phase), the
        # same key the flush saver partitions by, so every epoch walks its own
        # schedule from the phase's first episode and what a later epoch does is
        # decided by its configured row rather than by a carried-over count.
        self.epsilon_episode_schedule = epsilon_episode_schedule
        self._phase_episode_counts = defaultdict(int)

        # Current epoch, and a run-global episode counter. The counter makes
        # every training episode identifiable on its own, which is what lets
        # the analysis layer order episodes without depending on file names.
        self.current_epoch = 1
        self._episode_seq = 0

        # Store memory config for cleanup decisions
        self.memory_config = kwargs.get('memory_config', {})

        # One persistence policy per run. With progressive saving on, each
        # (epoch, phase) segment is written as it completes and memory limits flush
        # to disk. With it off, the experiment writes a single file at the end, so
        # the trainer must not also write and clear segment files: doing both left
        # that single file holding only the final phase's remainder.
        self.progressive_saving = self._progressive_saving_enabled(self.memory_config)
        
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
        
        # Epsilon fields are normalised to one row per epoch, each holding one
        # value per phase. A flat config keeps its old meaning and every epoch
        # reuses that single row; a config written as a list of rows gives each
        # epoch its own, with the last row covering any epochs beyond it. Both
        # schedule modes read the same grid, so a config means the same thing
        # whichever mode it selects.
        #
        # The raw values are kept and the grid is built inside `train`, which is
        # where the epoch count is authoritative. Nothing reads the grid before
        # then, so there is no second place for the two counts to disagree.
        self._epsilon_fields = (
            ('epsilon_decay', epsilon_decay, 0.999, float),
            ('epsilon_start', epsilon_start, 1.0, float),
            ('epsilon_min', epsilon_min, 0.05, float),
            ('epsilon_decay_type', epsilon_decay_type, 'exponential', str),
        )
        self._epsilon_grid = {}

    def _build_epsilon_grid(self, num_epochs: int):
        """
        Normalise every epsilon field to `num_epochs` rows of per-phase values.

        A field left unset falls back to the agent's own attribute. A field
        written for a different curriculum length is reshaped rather than
        rejected, which the trainer has always done, and the adjustment is
        reported once per field rather than once per epoch row.

        Args:
            num_epochs: Epochs this run will perform.
        """
        for name, raw, default, cast in self._epsilon_fields:
            value = raw if raw is not None else getattr(self.agent, name, default)
            if epsilon_schedule.needs_fitting(value, self.num_phases):
                print(f"Adjusting {name} to {self.num_phases} phases")
            self._epsilon_grid[name] = epsilon_schedule.as_per_epoch(
                value, num_epochs, self.num_phases, name, cast=cast, fit=True)

    def _apply_epoch_schedules(self, epoch: int):
        """
        Install one epoch's epsilon schedule as the active per-phase schedule.

        Called once per epoch so that the phase-transition handler, which reads
        the per-phase lists, picks up the row belonging to the epoch it is in.
        Epochs past the configured rows clamp to the last one, which is the
        same reuse-last rule the config normalisation applies.

        Epsilon itself is reset to the opening value of the epoch's first phase,
        independently of `epsilon_phase_reset`. That flag governs transitions
        within an epoch; an epoch's own starting value has to take effect or a
        per-epoch schedule row would have no way to be honoured.

        Args:
            epoch: 1-based epoch number.
        """
        grid = self._epsilon_grid
        e = max(0, min(epoch - 1, len(grid['epsilon_start']) - 1))
        self.epsilon_decay_schedule = list(grid['epsilon_decay'][e])
        self.epsilon_start_schedule = list(grid['epsilon_start'][e])
        self.epsilon_min_schedule = list(grid['epsilon_min'][e])
        self.epsilon_decay_type_schedule = list(grid['epsilon_decay_type'][e])

        # Re-seed the agent from this epoch's first phase. The phase-transition
        # handler overwrites these as soon as the curriculum moves, so this only
        # has to be right for the phase the epoch opens on.
        self.current_phase_epsilon_decay = self.epsilon_decay_schedule[0]
        self.current_phase_epsilon_start = self.epsilon_start_schedule[0]
        self.current_phase_epsilon_min = self.epsilon_min_schedule[0]
        self.current_phase_epsilon_decay_type = self.epsilon_decay_type_schedule[0]
        self.agent.set_epsilon_decay(self.current_phase_epsilon_decay)
        self.agent.set_epsilon_start(self.current_phase_epsilon_start)
        self.agent.set_epsilon_min(self.current_phase_epsilon_min)
        self.agent.set_epsilon_decay_type(self.current_phase_epsilon_decay_type)
        self.agent.epsilon = self.current_phase_epsilon_start


    # Note: _create_curriculum_schedule method removed - curriculum is now embedded in RobustDataGenerator
    
    @staticmethod
    def _progressive_saving_enabled(memory_config: dict) -> bool:
        """
        Read enable_progressive_saving, accepting either config structure.

        The current form nests settings under 'training' and 'evaluation'; the
        older flat form keeps them at the top level. Default is True, matching
        every shipped configuration.
        """
        if not memory_config:
            return True
        if 'training' in memory_config:
            return memory_config.get('training', {}).get('enable_progressive_saving', True)
        return memory_config.get('enable_progressive_saving', True)

    def _handle_segment_transition(self, closing_epoch: int, old_phase: int,
                                   new_epoch: int, new_phase: int, episode_count: int,
                                   config: dict = None, external_metrics_manager = None):
        """
        Close one (epoch, phase) segment and open the next.

        Keyed on the pair rather than on the phase alone. A curriculum of a single
        phase keeps the same phase number across epochs, so a phase-only test never
        fired: every flush file stayed labelled with the first epoch and the closing
        file mixed epochs together.

        Args:
            closing_epoch: Epoch of the segment being closed
            old_phase: Curriculum phase of the segment being closed
            new_epoch: Epoch of the segment being opened
            new_phase: Curriculum phase of the segment being opened
            episode_count: Current episode count
            config: Experiment configuration for saving
            external_metrics_manager: External metrics manager for saving partial results
        """
        print(f"\nTransitioning from Epoch {closing_epoch} Phase {old_phase} to "
              f"Epoch {new_epoch} Phase {new_phase} at episode {episode_count}")

        # Update training flush saver's current segment for new flush files
        if self.training_flush_saver is not None:
            self.training_flush_saver.set_segment(new_epoch, new_phase)
            print(f"   Updated flush saver to Epoch {new_epoch} Phase {new_phase}")

        # Save and clear accumulated training data for the completed phase to free memory
        if self.progressive_saving and config is not None and external_metrics_manager is not None:
            episode_collector = self.metrics_manager.get_collector('EpisodeDataCollector')
            if episode_collector is not None and episode_collector.episodes:
                phase_data = episode_collector.get_results().copy()  # Get a copy of the data
                if phase_data:
                    # Save phase data using external metrics manager
                    saved_path = external_metrics_manager.save_training_partial(
                        config=config,
                        training_data=phase_data,
                        epoch=closing_epoch,
                        phase=old_phase,
                        clear_memory=False  # We'll clear manually for more control
                    )
                    print(f"   Saved Epoch {closing_epoch} Phase {old_phase} data "
                          f"({len(phase_data)} episodes) to {saved_path.name}")
                    
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

        # Under the episode schedule these fields are inert: the decay rate is
        # zero and epsilon is rewritten at the start of the next episode. They
        # are still applied so the agent's own state stays consistent, but
        # reporting them would announce values that never take effect, so the
        # log reports the schedule's own numbers instead.
        scheduled = self.epsilon_episode_schedule is not None
        previous = (self.current_phase_epsilon_decay,
                    self.current_phase_epsilon_start,
                    self.current_phase_epsilon_min,
                    self.current_phase_epsilon_decay_type)

        self.current_phase_epsilon_decay = self.epsilon_decay_schedule[phase_index]
        self.current_phase_epsilon_start = self.epsilon_start_schedule[phase_index]
        self.current_phase_epsilon_min = self.epsilon_min_schedule[phase_index]
        self.current_phase_epsilon_decay_type = self.epsilon_decay_type_schedule[phase_index]
        self.agent.set_epsilon_decay(self.current_phase_epsilon_decay)
        self.agent.set_epsilon_start(self.current_phase_epsilon_start)
        self.agent.set_epsilon_min(self.current_phase_epsilon_min)
        self.agent.set_epsilon_decay_type(self.current_phase_epsilon_decay_type)

        # Reset training step count for linear decay when transitioning phases
        linear = self.current_phase_epsilon_decay_type == "linear"
        if linear:
            self.agent.reset_training_step_count()

        # Reset epsilon if requested (use current phase's epsilon_start)
        reset = self.epsilon_phase_reset and hasattr(self.agent, 'epsilon')
        if reset:
            self.agent.epsilon = self.current_phase_epsilon_start

        if scheduled:
            # Index 0 is the phase's opening value and a large index is past
            # every anneal span, so both ends come from the schedule itself
            # rather than from a second copy of the same numbers.
            opening = self.epsilon_episode_schedule(self.current_epoch, new_phase, 0)
            floor = self.epsilon_episode_schedule(self.current_epoch, new_phase, sys.maxsize)
            detail = (f"{opening:.4f} annealing to {floor:.4f}"
                      if opening > floor else f"held at {opening:.4f}")
            print(f"   Epsilon (epoch {self.current_epoch} schedule): {detail}")
        else:
            old_decay, old_start, old_min, old_decay_type = previous
            print(f"   Epsilon decay: {old_decay:.6f} → {self.current_phase_epsilon_decay:.6f}")
            print(f"   Epsilon start: {old_start:.4f} → {self.current_phase_epsilon_start:.4f}")
            print(f"   Epsilon min: {old_min:.4f} → {self.current_phase_epsilon_min:.4f}")
            print(f"   Epsilon decay type: {old_decay_type} → {self.current_phase_epsilon_decay_type}")
            if linear:
                print(f"   Reset training step count for linear decay")
            if reset:
                print(f"   Epsilon reset to {self.agent.epsilon:.4f}")
        
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

        # Prepare episode metadata for collectors. episode_id counts the whole
        # run rather than the current collector, which resets on every flush.
        episode_metadata = {
            'episode_id': self._episode_seq,
            'episode_type': episode_type,
            'agent_type': 'agent',
            'epoch': self.current_epoch,
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
            self._episode_seq += 1
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
        
        # Nothing may carry over from the previous episode: a pending update
        # belongs to the step that produced it, and that step has been recorded.
        self.agent.last_training_update = None

        while not done:
            # Agent selects action. The mask is None unless the environment
            # restricts the action set, in which case the agent honours it in
            # both the greedy and the exploring branch.
            action = self.agent.select_action(
                state, training=training,
                action_mask=self.env.get_action_mask())
            next_state, reward, done, info = self.env.step(action)

            # Store experience before this step's update, so the transition just
            # taken is available to the batch it is about to be sampled into.
            if training:
                self.replay_buffer.push(state, action, reward, next_state, done)

            # This step's gradient update runs before the step is recorded, so the
            # loss, epsilon and Q-value statistics it produces are attached to the
            # step that produced them rather than to the following one.
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

                    # Synchronize on the update that reaches the interval. Testing
                    # this at the episode boundary steps over most multiples,
                    # because episode length varies and rarely divides the interval.
                    self.training_steps += 1
                    if self.training_steps % self.target_update_freq == 0:
                        self.agent.update_target_network()

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

            # Attach this step's own training update, if it performed one
            if getattr(self.agent, 'last_training_update', None) is not None:
                step_data['training_update'] = self.agent.last_training_update
                self.agent.last_training_update = None  # Clear after use

            # Only notify collectors for training episodes (not evaluation episodes)
            if training:
                self.metrics_manager.on_step_end(step_data)

            state = next_state
            total_reward += reward
            episode_length += 1
            step_number += 1

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
              eval_episodes: int = 25, config: dict = None, external_metrics_manager = None,
              on_epoch_end = None, **kwargs) -> Dict[str, Any]:
        """
        Executes LDPC training using RobustDataGenerator with curriculum learning.
        
        Args:
            num_epochs: Number of training epochs (full passes through dataset)
            eval_frequency: Frequency of evaluation runs (in episodes)
            eval_episodes: Number of episodes per evaluation
            on_epoch_end: Called with the 1-based epoch number once that epoch's
                last episode has been trained on. An epoch boundary is the only
                point at which the weights represent a complete pass over the
                curriculum, and it is the only chance to capture them: training
                overwrites them on the next pass. Nothing here knows where models
                are stored, so the caller decides what to do with the moment.
            config: Experiment configuration for incremental saving
            external_metrics_manager: External metrics manager for saving partial results
            
        Returns:
            Dict containing training history and metrics
        """
        print(f"Starting LDPC training for {num_epochs} epochs using RobustDataGenerator...")
        
        # Initialize metrics collection for training
        self.metrics_manager.reset_all()
        
        # Track the (epoch, phase) segment and basic counters. Gradient updates
        # are counted by self.training_steps, incremented inside train_episode
        # where the target network is also synchronized, so there is one
        # authoritative counter rather than a loop copy that can disagree.
        current_phase = 1
        current_segment = (1, 1)
        episode_count = 0

        self._build_epsilon_grid(num_epochs)

        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            self._apply_epoch_schedules(self.current_epoch)
            print(f"\nStarting epoch {self.current_epoch}/{num_epochs}")

            # Iterate through training data
            for sample in self.data_generator.get_iterator('train'):
                # Get actual phase from dataset
                phase = sample['phase']

                # A segment is one (epoch, phase) pair, so a transition is any
                # change to that pair: a new phase within an epoch, the drop back
                # to phase 1 when a new epoch restarts the curriculum, and the
                # epoch boundary itself when the curriculum has only one phase.
                segment = (self.current_epoch, phase)
                if segment != current_segment:
                    self._handle_segment_transition(
                        current_segment[0], current_segment[1], self.current_epoch, phase,
                        episode_count, config, external_metrics_manager)
                    current_segment = segment
                    current_phase = phase
                
                # Epsilon from progress through this phase, when scheduled that
                # way. Set before the episode so every step within it explores
                # at the same rate; the agent's own per-update decay is a no-op
                # here because the resolved decay rate is zero.
                if self.epsilon_episode_schedule is not None:
                    key = (self.current_epoch, phase)
                    index = self._phase_episode_counts[key]
                    self._phase_episode_counts[key] = index + 1
                    self.agent.epsilon = self.epsilon_episode_schedule(
                        self.current_epoch, phase, index)

                # Training episode with dataset sample (now handles metrics automatically)
                episode_data = self.train_episode(sample, training=True)

                episode_count += 1

                # Periodic evaluation
                if episode_count % eval_frequency == 0:
                    eval_metrics = self._evaluate_dataset(eval_episodes, current_phase)
                    print(f"Episode {episode_count} (Epoch {epoch + 1}, Phase {current_phase}): "
                          f"Success Rate: {eval_metrics['success_rate']:.1%}, "
                          f"Avg Reward: {eval_metrics['avg_reward']:.2f}, "
                          f"Avg Steps: {eval_metrics['avg_length']:.1f}, "
                          f"Epsilon: {episode_data['epsilon']:.3f}")

            if on_epoch_end is not None:
                on_epoch_end(self.current_epoch)

        # Save any remaining training data from the final phase and cleanup memory
        if self.progressive_saving and config is not None and external_metrics_manager is not None:
            episode_collector = self.metrics_manager.get_collector('EpisodeDataCollector')
            if episode_collector is not None and episode_collector.episodes:
                final_phase_data = episode_collector.get_results().copy()
                if final_phase_data:
                    saved_path = external_metrics_manager.save_training_partial(
                        config=config,
                        training_data=final_phase_data,
                        epoch=self.current_epoch,
                        phase=current_phase,
                        clear_memory=False
                    )
                    print(f"Saved final Epoch {self.current_epoch} Phase {current_phase} data "
                          f"({len(final_phase_data)} episodes) to {saved_path.name}")
                    
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
        
        # NOW do the final memory cleanup after everything is collected. The policy
        # was resolved once in the constructor; recomputing it here had drifted to
        # the opposite default and could disagree with the flag the run acted on.
        #
        # The policy alone is not enough to authorize clearing. Segments are only
        # written when a config and an external metrics manager were supplied, so
        # without them nothing reached disk and clearing would discard the whole run.
        segments_were_saved = config is not None and external_metrics_manager is not None
        if self.progressive_saving and segments_were_saved:
            # With progressive saving, clear collectors completely after final eval
            episode_collector = self.metrics_manager.get_collector('EpisodeDataCollector')
            # An empty collector is falsey, so test for the lookup, not the contents.
            if episode_collector is not None:
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
            'total_training_steps': self.training_steps,
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
