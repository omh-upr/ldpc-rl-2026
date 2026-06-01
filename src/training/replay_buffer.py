"""
Replay Buffer for Experience Replay

This module provides a simple, efficient circular buffer for storing and 
sampling experiences (state, action, reward, next_state, done) for training
the DQN agent. Includes optional file persistence for training continuation
and experience sharing across sessions.
"""

import random
import pickle
import numpy as np
import os
import warnings
from collections import deque, namedtuple
from typing import Optional
from src.core.ldpc_code import LDPC_Code

# Define the structure of an experience
Experience = namedtuple('Experience', 
                        ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    """
    A fixed-size buffer to store experience tuples with optional file persistence.
    """
    def __init__(self, capacity=10000, replay_file=None, code: Optional[LDPC_Code] = None):
        """
        Initializes the ReplayBuffer.

        Args:
            capacity (int): The maximum number of experiences to store.
            replay_file (str, optional): Path to file for loading/saving experiences.
                                       If provided, will attempt to load on initialization.
            code (LDPC_Code, optional): LDPC code instance for dynamic validation.
        """
        self.capacity = capacity
        self.replay_file = replay_file
        self.code = code or LDPC_Code.from_default()
        self.buffer = deque([], maxlen=capacity)
        
        # Load from file if specified
        if replay_file is not None:
            self.load_from_file(replay_file)

    def push(self, *args):
        """
        Saves an experience to the buffer.

        Args:
            *args: The experience tuple (state, action, reward, next_state, done).
        """
        self.buffer.append(Experience(*args))

    def sample(self, batch_size=32):
        """
        Samples a batch of experiences from the buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            A list of randomly selected experience tuples.
        """
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} experiences from buffer of size {len(self.buffer)}"
            )
        return random.sample(self.buffer, batch_size)

    def sample_arrays(self, batch_size=32):
        """
        Phase 2 Optimization: Samples a batch and returns pre-allocated numpy arrays.

        This method is faster than sample() because it:
        - Uses numpy indexing instead of random.sample
        - Returns contiguous arrays instead of list of tuples
        - Avoids intermediate data structures

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            Tuple of numpy arrays: (states, actions, rewards, next_states, dones)
            - states: (batch_size, state_dim) float32
            - actions: (batch_size,) int64
            - rewards: (batch_size,) float32
            - next_states: (batch_size, state_dim) float32
            - dones: (batch_size,) float32
        """
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} experiences from buffer of size {len(self.buffer)}"
            )

        # Random sample indices
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)

        # Pre-allocate arrays
        # Note: We need to determine state_dim from first experience
        first_exp = self.buffer[0]
        state_dim = len(first_exp.state)

        states = np.empty((batch_size, state_dim), dtype=np.float32)
        actions = np.empty(batch_size, dtype=np.int64)
        rewards = np.empty(batch_size, dtype=np.float32)
        next_states = np.empty((batch_size, state_dim), dtype=np.float32)
        dones = np.empty(batch_size, dtype=np.float32)

        # Fill arrays using indexing
        for i, idx in enumerate(indices):
            exp = self.buffer[idx]
            states[i] = exp.state
            actions[i] = exp.action
            rewards[i] = exp.reward
            next_states[i] = exp.next_state
            dones[i] = float(exp.done)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """
        Returns the current number of experiences in the buffer.
        """
        return len(self.buffer)

    def save_to_file(self, filename=None):
        """
        Saves the buffer content to a pickle file.
        
        Args:
            filename (str, optional): Path to save file. If None, uses self.replay_file.
        
        Returns:
            bool: True if save successful, False otherwise.
        """
        if filename is None:
            filename = self.replay_file
            
        if filename is None:
            warnings.warn("No filename provided for save operation")
            return False
            
        try:
            # Convert deque to list for serialization
            buffer_data = {
                'experiences': list(self.buffer),
                'capacity': self.capacity,
                'version': '1.0'
            }
            
            # Create directory if it doesn't exist
            dirname = os.path.dirname(filename)
            if dirname:  # Only create directory if there's a path component
                os.makedirs(dirname, exist_ok=True)
            
            with open(filename, 'wb') as f:
                pickle.dump(buffer_data, f)
                
            print(f"Successfully saved {len(self.buffer)} experiences to {filename}")
            return True
            
        except Exception as e:
            warnings.warn(f"Failed to save replay buffer to {filename}: {str(e)}")
            return False

    def load_from_file(self, filename):
        """
        Loads experiences from a pickle file.
        
        Args:
            filename (str): Path to load file.
        
        Returns:
            bool: True if load successful, False otherwise.
        """
        if not os.path.exists(filename):
            warnings.warn(f"Replay file {filename} not found. Starting with empty buffer.")
            return False
            
        try:
            with open(filename, 'rb') as f:
                buffer_data = pickle.load(f)
                
            # Validate loaded data structure
            if not isinstance(buffer_data, dict) or 'experiences' not in buffer_data:
                warnings.warn(f"Invalid replay file format: {filename}")
                return False
                
            experiences = buffer_data['experiences']
            if not isinstance(experiences, list):
                warnings.warn(f"Invalid experiences format in {filename}")
                return False
                
            # Clear current buffer and recreate with proper capacity
            self.buffer = deque([], maxlen=self.capacity)
            
            # Load and validate experiences
            loaded_count = 0
            for exp in experiences:
                if self._validate_experience(exp):
                    self.buffer.append(exp)
                    loaded_count += 1
                else:
                    warnings.warn(f"Skipping invalid experience during load")
                    
            print(f"Successfully loaded {loaded_count} experiences from {filename}")
            return True
            
        except Exception as e:
            warnings.warn(f"Failed to load replay buffer from {filename}: {str(e)}")
            return False

    def _validate_experience(self, exp):
        """
        Validates that an experience has the correct structure.
        
        Args:
            exp: Experience object to validate.
            
        Returns:
            bool: True if valid, False otherwise.
        """
        try:
            # Check if it's an Experience namedtuple
            if not isinstance(exp, Experience):
                return False
                
            state, action, reward, next_state, done = exp
            
            # Validate state and next_state using dynamic dimensions
            expected_state_dim = self.code.state_dimensions['total']
            if not isinstance(state, np.ndarray) or state.shape != (expected_state_dim,):
                return False
            if not isinstance(next_state, np.ndarray) or next_state.shape != (expected_state_dim,):
                return False
                
            # Validate action using dynamic action space
            if not isinstance(action, (int, np.integer)) or not (0 <= action < self.code.action_dim):
                return False
                
            # Validate reward (numeric)
            if not isinstance(reward, (int, float, np.number)):
                return False
                
            # Validate done (boolean or float)
            if not isinstance(done, (bool, np.bool_, float, np.float32, np.float64)):
                return False
                
            return True
            
        except Exception:
            return False

    def auto_save(self):
        """
        Automatically saves to the default replay_file if specified.
        
        Returns:
            bool: True if save successful, False otherwise.
        """
        if self.replay_file is None:
            return False
        return self.save_to_file()

    def get_buffer_info(self):
        """
        Returns information about the buffer state.
        
        Returns:
            dict: Buffer information including size, capacity, and file status.
        """
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'utilization': len(self.buffer) / self.capacity,
            'replay_file': self.replay_file,
            'file_exists': os.path.exists(self.replay_file) if self.replay_file else False
        }

    def clear_and_reload(self):
        """
        Clears the current buffer and reloads from the replay file.
        
        Returns:
            bool: True if reload successful, False otherwise.
        """
        if self.replay_file is None:
            warnings.warn("No replay file specified for reload")
            return False
            
        # Clear buffer
        self.buffer.clear()
        
        # Reload from file
        return self.load_from_file(self.replay_file)