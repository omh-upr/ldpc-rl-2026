"""
Experience Generator for Expert Experience Collection

Runs an agent in an environment to generate experiences and saves them
to a replay buffer file. Supports both episode-based and experience-based
generation targets with progress tracking.
"""

import numpy as np
from tqdm import tqdm
from typing import Optional

from src.core.environment import LDPCEnvironment
from src.core.base_agent import BaseAgent
from src.training.replay_buffer import ReplayBuffer
from src.data.robust_data_generator import RobustDataGenerator


class ExperienceGenerator:
    """
    Runs an agent in an environment to generate experiences and saves them
    to a replay buffer file.
    
    Provides flexible generation targets (episodes or experience count) with
    comprehensive progress tracking and error handling.
    """
    
    def __init__(self, env: LDPCEnvironment, agent: BaseAgent, buffer_size: int, data_generator: RobustDataGenerator):
        """
        Initialize the experience generator.
        
        Args:
            env: LDPC environment for running episodes
            agent: Agent that conforms to BaseAgent interface
            buffer_size: Maximum capacity for replay buffer
            data_generator: Data generator for creating noisy/original codeword pairs
        """
        self.env = env
        self.agent = agent
        self.replay_buffer = ReplayBuffer(capacity=buffer_size, code=env.code)
        self.data_generator = data_generator
        
        # Setup data iterator for entire dataset (no train/test split needed)
        self.data_iterator = self.data_generator.get_iterator('all')

    def _next_data_sample_or_fail(
        self,
        mode: str,
        target: int,
        episodes_done: int,
        experiences_collected: int,
    ) -> dict:
        """
        Get next sample from iterator or fail fast with actionable guidance.

        Args:
            mode: Generation mode ('experience_count' or 'episode_count')
            target: Requested target value for the mode
            episodes_done: Number of episodes completed so far
            experiences_collected: Number of experiences collected so far

        Returns:
            Next data sample dictionary

        Raises:
            RuntimeError: If the dataset iterator is exhausted
        """
        try:
            return next(self.data_iterator)
        except StopIteration as exc:
            if mode == "experience_count":
                target_desc = f"target experiences={target}"
            else:
                target_desc = f"target episodes={target}"

            raise RuntimeError(
                "Dataset iterator exhausted during expert experience generation.\n"
                f"Progress at failure: episodes_done={episodes_done}, experiences_collected={experiences_collected}, {target_desc}.\n"
                "Suggested actions:\n"
                "  1. Increase dataset size (regenerate or use a larger dataset file).\n"
                "  2. Reduce requested generation target (num_episodes or num_experiences).\n"
                "  3. If cycling is intended, implement an explicit iterator restart policy."
            ) from exc
    
    def generate(self, num_episodes: Optional[int] = None, num_experiences: Optional[int] = None) -> int:
        """
        Generate experiences by running the agent in the environment.
        
        Generation stops when either the episode count or experience count is met.
        num_experiences has priority over num_episodes if both are specified.
        
        Args:
            num_episodes: The number of episodes to run
            num_experiences: The target number of experiences (steps) to collect
            
        Returns:
            The total number of experiences collected
            
        Raises:
            ValueError: If neither num_episodes nor num_experiences is specified
        """
        if num_experiences is not None:
            return self._generate_by_experience_count(num_experiences)
        elif num_episodes is not None:
            return self._generate_by_episode_count(num_episodes)
        else:
            raise ValueError("Either num_episodes or num_experiences must be specified")
    
    def _generate_by_experience_count(self, num_experiences: int) -> int:
        """Generate experiences until target experience count is reached.

        Progress bar updates every 25000 experiences to reduce logging overhead.
        """
        UPDATE_INTERVAL = 25000

        with tqdm(total=num_experiences, desc="Generating Experiences") as pbar:
            episode_count = 0
            last_update_position = 0

            while len(self.replay_buffer) < num_experiences:
                # Get data sample for episode
                data_sample = self._next_data_sample_or_fail(
                    mode="experience_count",
                    target=num_experiences,
                    episodes_done=episode_count,
                    experiences_collected=len(self.replay_buffer),
                )
                noisy_codeword = data_sample['noisy']
                original_codeword = data_sample['original']

                # Reset agent for new episode with appropriate method
                if hasattr(self.agent, 'reset_with_oracle'):
                    self.agent.reset_with_oracle(original_codeword)
                elif hasattr(self.agent, 'reset'):
                    self.agent.reset()

                state = self.env.reset(noisy_codeword, original_codeword)
                done = False
                episode_count += 1

                while not done:
                    action = self.agent.select_action(state)
                    next_state, reward, done, _ = self.env.step(action)

                    self.replay_buffer.push(state, action, reward, next_state, float(done))

                    # Update progress bar only every 5000 experiences
                    current_position = len(self.replay_buffer)
                    if current_position - last_update_position >= UPDATE_INTERVAL:
                        experiences_to_update = current_position - last_update_position
                        pbar.update(experiences_to_update)
                        pbar.set_postfix(episodes=episode_count, experiences=current_position)
                        last_update_position = current_position

                    state = next_state

                    if current_position >= num_experiences:
                        break

            # Final update to ensure progress bar reaches 100%
            final_experiences = len(self.replay_buffer) - last_update_position
            if final_experiences > 0:
                pbar.update(final_experiences)
            pbar.set_postfix(episodes=episode_count, experiences=len(self.replay_buffer))

        return len(self.replay_buffer)
    
    def _generate_by_episode_count(self, num_episodes: int) -> int:
        """Generate experiences for a fixed number of episodes."""
        with tqdm(total=num_episodes, desc="Generating Episodes") as pbar:
            total_experiences = 0
            
            for episode_idx in range(num_episodes):
                # Get data sample for episode
                data_sample = self._next_data_sample_or_fail(
                    mode="episode_count",
                    target=num_episodes,
                    episodes_done=episode_idx,
                    experiences_collected=total_experiences,
                )
                noisy_codeword = data_sample['noisy']
                original_codeword = data_sample['original']
                
                # Reset agent for new episode with appropriate method
                if hasattr(self.agent, 'reset_with_oracle'):
                    self.agent.reset_with_oracle(original_codeword)
                elif hasattr(self.agent, 'reset'):
                    self.agent.reset()
                
                state = self.env.reset(noisy_codeword, original_codeword)
                done = False
                episode_experiences = 0
                
                while not done:
                    action = self.agent.select_action(state)
                    next_state, reward, done, _ = self.env.step(action)
                    
                    self.replay_buffer.push(state, action, reward, next_state, float(done))
                    episode_experiences += 1
                    total_experiences += 1
                    
                    state = next_state
                
                pbar.update(1)
                pbar.set_postfix(total_experiences=total_experiences, avg_steps=total_experiences/(episode_idx+1))
        
        return len(self.replay_buffer)
    
    def save(self, file_path: str) -> bool:
        """
        Save the replay buffer to a file.
        
        Args:
            file_path: Path where to save the replay buffer
            
        Returns:
            True if save was successful, False otherwise
            
        Raises:
            RuntimeError: If save operation fails
        """
        success = self.replay_buffer.save_to_file(file_path)
        
        if not success:
            raise RuntimeError(f"Failed to save replay buffer to {file_path}")
        
        print(f"Successfully saved {len(self.replay_buffer)} experiences to {file_path}")
        return True
    
    def get_buffer_info(self) -> dict:
        """
        Get information about the current buffer state.
        
        Returns:
            Dictionary with buffer statistics
        """
        return {
            'experiences_collected': len(self.replay_buffer),
            'buffer_capacity': self.replay_buffer.capacity,
            'buffer_utilization': len(self.replay_buffer) / self.replay_buffer.capacity,
            'environment_info': {
                'action_dim': self.env.action_dim,
                'state_dim': self.env.state_dim,
                'max_steps': self.env.max_steps
            }
        }
