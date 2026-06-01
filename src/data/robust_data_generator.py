import copy
import numpy as np
import h5py
import json
from typing import Dict, List, Tuple, Iterator, Any, Union, Optional
from src.core.ldpc_code import LDPC_Code

class RobustDataGenerator:
    """
    Generates, saves, and serves robust LDPC datasets for training and evaluation.
    """
    def __init__(self, config: Dict[str, Any], code: Optional[LDPC_Code] = None):
        """
        Initializes the generator with a configuration dictionary.

        Args:
            config: A dictionary containing the data generation parameters,
                    including seed, curriculum, and uniqueness constraints.
            code: LDPC code instance for dynamic dimensions
        """
        self.config = copy.deepcopy(config)
        self.h5_filepath = None
        self.h5_file = None

        # If total_samples is defined, recalculate phase samples to be proportional
        if 'total_samples' in self.config and self.config['total_samples'] is not None:
            self._recalculate_phase_samples()

        # Initialize LDPC code
        self.code = code or LDPC_Code.from_default()
        self.k = self.code.k
        self.n = self.code.n

        # Initialize a random number generator for reproducibility
        self.rng = np.random.default_rng(self.config.get('seed'))

        # Uniqueness tracking sets
        self._global_message_cache = set()
        self._global_error_pattern_cache = set()

        # Placeholder for split indices
        self.train_indices = None
        self.test_indices = None

        self._error_model_registry = {
            'uniform_count': self._add_uniform_errors_by_count,
            'uniform_ber': self._add_uniform_errors_by_ber,
            'burst': self._add_burst_errors,
            'cluster': self._add_cluster_errors,
            'asymmetric': self._add_asymmetric_errors,
            'systematic': self._add_systematic_errors,
        }

    def _recalculate_phase_samples(self):
        """Recalculates num_samples for each phase based on total_samples."""
        total_samples = self.config['total_samples']
        curriculum = self.config['curriculum']
        
        # Treat num_samples as weights and calculate total weight
        total_weight = sum(phase.get('num_samples', 0) for phase in curriculum)
        if total_weight == 0:
            # If all weights are zero, distribute samples evenly
            num_phases = len(curriculum)
            if num_phases > 0:
                samples_per_phase = total_samples // num_phases
                for phase in curriculum:
                    phase['num_samples'] = samples_per_phase
        else:
            # Calculate proportional samples
            running_total = 0
            for phase in curriculum:
                weight = phase.get('num_samples', 0)
                proportional_samples = int((weight / total_weight) * total_samples)
                phase['num_samples'] = proportional_samples
                running_total += proportional_samples
        
        # Distribute remainder to ensure total_samples is met exactly
        remainder = total_samples - sum(p['num_samples'] for p in curriculum)
        for i in range(remainder):
            curriculum[i % len(curriculum)]['num_samples'] += 1

    def _add_cluster_errors(self, num_patterns: int, params: dict) -> np.ndarray:
        """Generates errors clustered around a central point."""
        num_errors = params['num_errors']
        radius = params['radius']
        error_patterns = np.zeros((num_patterns, self.n), dtype=np.int8)

        for i in range(num_patterns):
            center = self.rng.integers(0, self.n)
            # Define the pool of indices within the cluster, wrapping around the codeword
            indices_pool = np.arange(center - radius, center + radius + 1) % self.n
            # Ensure indices are unique, as modulo can cause duplicates
            unique_indices = np.unique(indices_pool)
            
            # If the number of unique indices is less than num_errors, we take all of them
            actual_num_errors = min(num_errors, len(unique_indices))
            
            error_indices = self.rng.choice(unique_indices, size=actual_num_errors, replace=False)
            error_patterns[i, error_indices] = 1
        return error_patterns

    def _add_asymmetric_errors(self, num_patterns: int, params: dict) -> np.ndarray:
        """Generates errors with different probabilities for 0->1 and 1->0 flips."""
        # This model is unique as it depends on the original codeword
        # It will be handled specially in the main generation loop
        # This function will simply return a placeholder or raise an error
        raise NotImplementedError("Asymmetric errors are applied directly in the main loop, not pre-generated.")

    def _add_systematic_errors(self, num_patterns: int, params: dict) -> np.ndarray:
        """Generates errors with a systematic, periodic pattern."""
        step = params['step']
        offset = params.get('offset', 0)
        error_patterns = np.zeros((num_patterns, self.n), dtype=np.int8)
        for i in range(num_patterns):
            # Each pattern can have a slightly different offset if specified as a list
            current_offset = self.rng.choice(offset) if isinstance(offset, list) else offset
            error_indices = np.arange(current_offset, self.n, step)
            error_patterns[i, error_indices] = 1
        return error_patterns

    def _add_uniform_errors_by_count(self, num_patterns: int, params: dict) -> np.ndarray:
        """Generates error patterns with a fixed number of errors."""
        num_errors = params['num_errors']
        if isinstance(num_errors, list):
            # If a list is provided, choose randomly from it for each pattern
            error_counts = self.rng.choice(num_errors, size=num_patterns)
        else:
            error_counts = np.full(num_patterns, num_errors)

        error_patterns = np.zeros((num_patterns, self.n), dtype=np.int8)
        for i in range(num_patterns):
            error_indices = self.rng.choice(self.n, size=error_counts[i], replace=False)
            error_patterns[i, error_indices] = 1
        return error_patterns

    def _add_uniform_errors_by_ber(self, num_patterns: int, params: dict) -> np.ndarray:
        """Generates error patterns based on a bit error rate."""
        ber = params['ber']
        error_patterns = self.rng.binomial(1, ber, size=(num_patterns, self.n)).astype(np.int8)
        return error_patterns

    def _add_burst_errors(self, num_patterns: int, params: dict) -> np.ndarray:
        """Generates error patterns with a consecutive burst of errors."""
        length = params['length']
        if isinstance(length, list):
            burst_lengths = self.rng.choice(length, size=num_patterns)
        else:
            burst_lengths = np.full(num_patterns, length)

        error_patterns = np.zeros((num_patterns, self.n), dtype=np.int8)
        for i in range(num_patterns):
            start_pos = self.rng.integers(0, self.n - burst_lengths[i] + 1)
            error_patterns[i, start_pos:start_pos + burst_lengths[i]] = 1
        return error_patterns

    def _generate_unique_error_patterns(self, n_patterns: int, model_func, params: dict, uniqueness_config: dict, phase_cache: set) -> np.ndarray:
        """Generates a batch of unique error patterns using a specified model."""
        if not uniqueness_config.get('ensure_unique_error_patterns', False):
            return model_func(n_patterns, params)

        patterns = set()
        max_attempts = n_patterns * 100
        attempts = 0
        use_global_cache = uniqueness_config.get('scope') == 'dataset'

        while len(patterns) < n_patterns and attempts < max_attempts:
            # Generate a single pattern
            pattern_array = model_func(1, params)[0]
            pattern_tuple = tuple(pattern_array)

            is_globally_unique = pattern_tuple not in self._global_error_pattern_cache
            is_phase_unique = pattern_tuple not in phase_cache

            if is_phase_unique and (not use_global_cache or is_globally_unique):
                patterns.add(pattern_tuple)
                phase_cache.add(pattern_tuple)
                if use_global_cache:
                    self._global_error_pattern_cache.add(pattern_tuple)
            attempts += 1

        if len(patterns) < n_patterns:
            raise RuntimeError(f"Failed to generate {n_patterns} unique error patterns for model '{model_func.__name__}' after {max_attempts} attempts.")

        return np.array(list(patterns), dtype=np.int8)

    def _generate_unique_messages(self, n_messages: int, uniqueness_config: dict, phase_cache: set) -> np.ndarray:
        """Generates a batch of unique messages."""
        if not uniqueness_config.get('ensure_unique_messages', False):
            # For large k values (e.g., k=128), 2**k exceeds uint64 limits
            # Generate messages directly as binary arrays instead
            return self.rng.integers(0, 2, size=(n_messages, self.k), dtype=np.int8)

        messages = set()
        max_attempts = n_messages * 100  # Safety break
        attempts = 0
        
        use_global_cache = uniqueness_config.get('scope') == 'dataset'

        while len(messages) < n_messages and attempts < max_attempts:
            # Generate a potential message as a tuple of 0s and 1s
            # For large k values, generate directly as binary array to avoid uint64 overflow
            msg_bits = self.rng.integers(0, 2, size=self.k, dtype=np.int8)
            msg_tuple = tuple(msg_bits)

            # Check for uniqueness
            is_globally_unique = msg_tuple not in self._global_message_cache
            is_phase_unique = msg_tuple not in phase_cache

            if is_phase_unique and (not use_global_cache or is_globally_unique):
                messages.add(msg_tuple)
                phase_cache.add(msg_tuple)
                if use_global_cache:
                    self._global_message_cache.add(msg_tuple)
            attempts += 1
        
        if len(messages) < n_messages:
            raise RuntimeError(f"Failed to generate {n_messages} unique messages after {max_attempts} attempts.")

        return np.array(list(messages), dtype=np.int8)

    def generate_and_save_dataset(self, output_filepath: str):
        """
        The main generation method. Reads the curriculum from the config,
        generates all data phase by phase, and saves the complete dataset
        to the specified HDF5 file.
        """
        self.h5_filepath = output_filepath
        print(f"Starting dataset generation. Output will be saved to: {self.h5_filepath}")
        uniqueness_config = self.config.get('uniqueness', {})

        with h5py.File(self.h5_filepath, 'w') as f:
            # --- 1. Save top-level attributes ---
            f.attrs['dataset_name'] = self.config.get('dataset_name', 'Unknown')
            f.attrs['seed'] = self.config.get('seed', None)
            f.attrs['creation_date'] = np.bytes_(str(np.datetime64('now', 's')))
            f.attrs['source_config'] = json.dumps(self.config)

            total_samples = sum(phase['num_samples'] for phase in self.config['curriculum'])
            f.attrs['total_samples'] = total_samples

            # --- 2. Process each curriculum phase ---
            phase_message_cache = set()
            phase_error_pattern_cache = set()
            for i, phase_config in enumerate(self.config['curriculum']):
                phase_name = phase_config.get('phase_name', f'phase_{i}')
                num_samples = phase_config['num_samples']
                print(f"\nProcessing {phase_name} ({num_samples} samples)...")

                # Reset phase caches if uniqueness is per-phase
                if uniqueness_config.get('scope') == 'phase':
                    phase_message_cache.clear()
                    phase_error_pattern_cache.clear()

                # Create a group for the current phase
                phase_group = f.create_group(f'phase_{i}')
                phase_group.attrs['phase_name'] = phase_name
                phase_group.attrs['num_samples'] = num_samples

                # --- 3. Generate data for the phase ---
                messages = self._generate_unique_messages(num_samples, uniqueness_config, phase_message_cache)
                original_codewords = self.code.encode(messages)

                # Generate error patterns based on the weighted models in the config
                error_patterns = np.zeros((num_samples, self.n), dtype=np.int8)
                metadata = np.empty(num_samples, dtype=[('error_model', 'S16'), ('num_errors', 'i4')])
                
                model_weights = [model['weight'] for model in phase_config['error_models']]
                model_indices = self.rng.choice(len(phase_config['error_models']), size=num_samples, p=model_weights)

                for model_idx, model_config in enumerate(phase_config['error_models']):
                    target_indices = np.where(model_indices == model_idx)[0]
                    count = len(target_indices)
                    if count == 0:
                        continue
                    
                    model_type = model_config['type']
                    params = model_config['params']

                    # Handle the special case of asymmetric errors
                    if model_type == 'asymmetric':
                        p_01 = params['p_01']
                        p_10 = params['p_10']
                        # Generate errors directly based on the original codewords selected for this model
                        codeword_slice = original_codewords[target_indices]
                        is_zero = (codeword_slice == 0)
                        is_one = (codeword_slice == 1)
                        prob_flips = np.zeros_like(codeword_slice, dtype=float)
                        prob_flips[is_zero] = p_01
                        prob_flips[is_one] = p_10
                        patterns = self.rng.binomial(1, prob_flips).astype(np.int8)
                    else:
                        model_func = self._error_model_registry[model_type]
                        patterns = self._generate_unique_error_patterns(count, model_func, params, uniqueness_config, phase_error_pattern_cache)
                    
                    error_patterns[target_indices] = patterns
                    
                    metadata['error_model'][target_indices] = np.bytes_(model_type)
                    metadata['num_errors'][target_indices] = np.sum(patterns, axis=1)

                # Shuffle the generated data within the phase
                shuffle_indices = self.rng.permutation(num_samples)
                original_codewords = original_codewords[shuffle_indices]
                error_patterns = error_patterns[shuffle_indices]
                metadata = metadata[shuffle_indices]

                noisy_codewords = (original_codewords + error_patterns) % 2

                # --- 4. Save datasets to the HDF5 group ---
                phase_group.create_dataset('original_codewords', data=original_codewords, compression='gzip')
                phase_group.create_dataset('noisy_codewords', data=noisy_codewords, compression='gzip')
                phase_group.create_dataset('error_patterns', data=error_patterns, compression='gzip')
                phase_group.create_dataset('metadata', data=metadata, compression='gzip')

                print(f"Finished processing {phase_name}.")

        print(f"\nDataset successfully generated and saved to {self.h5_filepath}")

    @classmethod
    def from_h5(cls, h5_filepath: str, code: Optional[LDPC_Code] = None) -> 'RobustDataGenerator':
        """
        Class method to load a generator instance from a pre-generated HDF5 dataset file.
        """
        if not h5py.is_hdf5(h5_filepath):
            raise ValueError(f"File is not a valid HDF5 file: {h5_filepath}")

        with h5py.File(h5_filepath, 'r') as f:
            config_str = f.attrs.get('source_config')
            if not config_str:
                raise ValueError("HDF5 file does not contain a source_config attribute.")
            config = json.loads(config_str)

        # Create an instance of the class with the loaded config
        generator = cls(config, code)
        generator.h5_filepath = h5_filepath
        generator.h5_file = h5py.File(h5_filepath, 'r+') # Keep the file open for reading and writing
        
        print(f"Generator loaded from {h5_filepath}")
        return generator

    def create_split(self, test_size: float = 0.2, stratify: bool = True, shuffle: bool = True, seed: int = None):
        """
        Creates a train/test split of the data and saves it to the HDF5 file.
        
        Args:
            test_size: Fraction of data to use for testing
            stratify: Whether to maintain phase proportions in train/test splits
            shuffle: Maintained for compatibility, but cross-phase shuffling is disabled to preserve curriculum
            seed: Random seed for reproducible splits
        """
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_filepath, 'a')

        print(f"Creating {'stratified ' if stratify else ''}train/test split with test_size={test_size}...")
        split_rng = np.random.default_rng(seed)

        train_indices_list = []
        test_indices_list = []
        global_offset = 0

        # Use consistent phase sizes instead of HDF5 attributes to avoid discrepancies
        phase_sizes = self._get_consistent_phase_sizes()

        for i, num_samples_in_phase in enumerate(phase_sizes):
            phase_indices = np.arange(num_samples_in_phase)

            if stratify:
                # Shuffle indices within the phase before splitting
                split_rng.shuffle(phase_indices)
                split_point = int(num_samples_in_phase * (1 - test_size))
                
                train_phase_indices = phase_indices[:split_point]
                test_phase_indices = phase_indices[split_point:]
                
                train_indices_list.append(train_phase_indices + global_offset)
                test_indices_list.append(test_phase_indices + global_offset)
            
            global_offset += num_samples_in_phase

        if stratify:
            self.train_indices = np.concatenate(train_indices_list)
            self.test_indices = np.concatenate(test_indices_list)
        else:
            # Simple random split across the whole dataset
            total_samples = self.h5_file.attrs['total_samples']
            all_indices = np.arange(total_samples)
            split_rng.shuffle(all_indices)
            split_point = int(total_samples * (1 - test_size))
            self.train_indices = all_indices[:split_point]
            self.test_indices = all_indices[split_point:]

        # Note: We don't shuffle the final indices to preserve curriculum ordering
        # Individual phases are already shuffled internally during generation

        # Save splits to the HDF5 file
        if 'splits' in self.h5_file:
            del self.h5_file['splits']
        splits_group = self.h5_file.create_group('splits')
        splits_group.create_dataset('train_indices', data=self.train_indices)
        splits_group.create_dataset('test_indices', data=self.test_indices)
        splits_group.attrs['num_train_samples'] = len(self.train_indices)
        splits_group.attrs['num_test_samples'] = len(self.test_indices)
        
        print(f"Split created: {len(self.train_indices)} train samples, {len(self.test_indices)} test samples.")
        
        # Validate split consistency
        self._validate_split_consistency()

    def _get_consistent_phase_sizes(self):
        """
        Get phase sizes consistently used by both split creation and iterator.
        
        This ensures that both create_split() and _map_global_indices_to_phases()
        use identical phase size calculations, eliminating discrepancies caused
        by rounding errors or inconsistent assumptions.
        
        Returns:
            List of integers representing the number of samples in each phase
        """
        if hasattr(self, '_cached_phase_sizes'):
            return self._cached_phase_sizes
            
        if 'total_samples' in self.config:
            # Use weighted distribution (same logic as dataset creation)
            total_weight = sum(phase['num_samples'] for phase in self.config['curriculum'])
            total_samples = self.config['total_samples']
            
            phase_sizes = []
            remaining_samples = total_samples
            
            for i, phase in enumerate(self.config['curriculum']):
                if i == len(self.config['curriculum']) - 1:
                    # Last phase gets remaining samples to avoid rounding errors
                    phase_size = remaining_samples
                else:
                    phase_size = int(round(phase['num_samples'] * total_samples / total_weight))
                    remaining_samples -= phase_size
                phase_sizes.append(phase_size)
        else:
            # Use direct num_samples (no total_samples constraint)
            phase_sizes = [phase['num_samples'] for phase in self.config['curriculum']]
        
        self._cached_phase_sizes = phase_sizes
        return phase_sizes

    def _validate_split_consistency(self):
        """Validate that split creation and iterator use same assumptions."""
        expected_total = sum(self._get_consistent_phase_sizes())
        
        if hasattr(self, 'train_indices') and hasattr(self, 'test_indices'):
            actual_total = len(self.train_indices) + len(self.test_indices)
            if expected_total != actual_total:
                raise ValueError(
                    f"Split consistency error: {actual_total} total indices "
                    f"for {expected_total} expected samples. "
                    f"Train: {len(self.train_indices)}, Test: {len(self.test_indices)}"
                )
            print(f"   Split consistency validated: {actual_total} indices match expected samples")

    def _get_phase_boundaries_map(self):
        """Create mapping from phase to global index ranges.

        Uses _get_consistent_phase_sizes() to ensure consistency
        with split creation and iterator logic.
        """
        phase_map = {}
        global_offset = 0

        for phase_idx, num_samples in enumerate(self._get_consistent_phase_sizes()):
            phase_map[phase_idx + 1] = {  # 1-indexed phases
                'start': global_offset,
                'end': global_offset + num_samples,
                'size': num_samples
            }
            global_offset += num_samples

        return phase_map

    def _map_global_indices_to_phases(self):
        """Helper method to create a mapping from global index to (phase, local_index)."""
        self._phase_map = {}
        self._phase_boundaries = []
        global_offset = 0
        
        # Use consistent phase sizes instead of HDF5 attributes to match split creation
        phase_sizes = self._get_consistent_phase_sizes()
        
        for phase_size in phase_sizes:
            self._phase_boundaries.append(global_offset + phase_size)
            global_offset += phase_size

    def get_iterator(self, split_name: str) -> Iterator[Dict[str, Any]]:
        """
        Returns a memory-efficient iterator for the specified data split ('train', 'test', or 'all').
        """
        if split_name not in ['train', 'test', 'all']:
            raise ValueError("split_name must be 'train', 'test', or 'all'")

        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_filepath, 'r+')

        if split_name == 'all':
            total_samples = self.h5_file.attrs['total_samples']
            indices_to_iterate = np.arange(total_samples)
        else: # 'train' or 'test'
            if self.train_indices is None or self.test_indices is None:
                # Attempt to load splits from file if they exist
                if 'splits' in self.h5_file:
                    self.train_indices = self.h5_file['splits/train_indices'][:]
                    self.test_indices = self.h5_file['splits/test_indices'][:]
                    print("Loaded existing splits from HDF5 file.")
                else:
                    raise RuntimeError("You must call create_split() before getting an iterator for 'train' or 'test' splits.")
            indices_to_iterate = self.train_indices if split_name == 'train' else self.test_indices

        self._map_global_indices_to_phases()

        # print(f"Providing iterator for '{split_name}' split with {len(indices_to_iterate)} samples.")
        for global_idx in indices_to_iterate:
            # Find which phase this global index belongs to
            phase_idx = np.searchsorted(self._phase_boundaries, global_idx, side='right')
            
            # Calculate the local index within that phase
            phase_start_index = self._phase_boundaries[phase_idx-1] if phase_idx > 0 else 0
            local_idx = global_idx - phase_start_index

            phase_group = self.h5_file[f'phase_{phase_idx}']
            
            yield {
                'original': phase_group['original_codewords'][local_idx],
                'noisy': phase_group['noisy_codewords'][local_idx],
                'error': phase_group['error_patterns'][local_idx],
                'meta': phase_group['metadata'][local_idx],
                'phase': phase_idx + 1  # 1-indexed phase number
            }

    def get_num_phases(self) -> int:
        """
        Get the number of phases in this dataset.
        
        Returns:
            Number of curriculum phases in the dataset
        """
        return len(self.config['curriculum'])
    
    def get_phase_info(self) -> Dict[str, Any]:
        """
        Get detailed information about all phases in the dataset.
        
        Returns:
            Dictionary with phase information including names, sample counts, etc.
        """
        phase_info = {
            'num_phases': self.get_num_phases(),
            'phases': {}
        }
        
        for i, phase_config in enumerate(self.config['curriculum']):
            phase_num = i + 1  # 1-indexed
            phase_info['phases'][phase_num] = {
                'phase_name': phase_config.get('phase_name', f'phase_{i}'),
                'num_samples': phase_config['num_samples'],
                'error_models': [model['type'] for model in phase_config.get('error_models', [])]
            }
        
        return phase_info

    def get_phase_iterator(self, split_name: str, phases: Union[int, List[int]], 
                          max_samples: int = None) -> Iterator[Dict[str, Any]]:
        """
        Get iterator for specific phases from specified split.
        
        Args:
            split_name: 'train', 'test', or 'all'
            phases: Single phase number or list of phase numbers (1-indexed)
            max_samples: Maximum samples to return (None for all)
        
        Returns:
            Iterator yielding samples from specified phases
        """
        if split_name not in ['train', 'test', 'all']:
            raise ValueError("split_name must be 'train', 'test', or 'all'")
        
        if isinstance(phases, (int, np.integer)):
            phases = [int(phases)]
        
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_filepath, 'r+')
        
        # Get split indices
        if split_name == 'all':
            total_samples = self.h5_file.attrs['total_samples']
            target_indices = np.arange(total_samples)
        else:
            if self.train_indices is None or self.test_indices is None:
                if 'splits' in self.h5_file:
                    self.train_indices = self.h5_file['splits/train_indices'][:]
                    self.test_indices = self.h5_file['splits/test_indices'][:]
                else:
                    raise RuntimeError("Split not found. Call create_split() first.")
            target_indices = self.train_indices if split_name == 'train' else self.test_indices
        
        # Get phase boundaries
        phase_map = self._get_phase_boundaries_map()
        
        # Filter indices by requested phases
        phase_indices = []
        for phase in phases:
            if phase not in phase_map:
                continue
            
            phase_start = phase_map[phase]['start']
            phase_end = phase_map[phase]['end']
            
            # Find indices within this phase
            mask = (target_indices >= phase_start) & (target_indices < phase_end)
            phase_specific_indices = target_indices[mask]
            phase_indices.extend(phase_specific_indices)
        
        # Apply max_samples limit
        if max_samples is not None and len(phase_indices) > max_samples:
            phase_indices = phase_indices[:max_samples]
        
        # Use existing iterator logic with filtered indices
        self._map_global_indices_to_phases()
        
        for global_idx in phase_indices:
            # Find which phase this global index belongs to
            phase_idx = np.searchsorted(self._phase_boundaries, global_idx, side='right')
            
            # Calculate the local index within that phase
            phase_start_index = self._phase_boundaries[phase_idx-1] if phase_idx > 0 else 0
            local_idx = global_idx - phase_start_index
            
            phase_group = self.h5_file[f'phase_{phase_idx}']
            
            yield {
                'original': phase_group['original_codewords'][local_idx],
                'noisy': phase_group['noisy_codewords'][local_idx],
                'error': phase_group['error_patterns'][local_idx],
                'meta': phase_group['metadata'][local_idx],
                'phase': phase_idx + 1  # 1-indexed phase number
            }

    def close(self):
        """Close the HDF5 file if open."""
        if self.h5_file:
            self.h5_file.close()
            self.h5_file = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()

# Example Usage (for demonstration)
if __name__ == '__main__':
    # Define a simple configuration for testing
    example_config = {
      "seed": 42,
      "dataset_name": "Initial_Test_Dataset",
      "curriculum": [
        {
          "phase_name": "Phase1_Dummy",
          "num_samples": 1000,
          "error_models": [] # To be implemented
        },
        {
          "phase_name": "Phase2_Dummy",
          "num_samples": 2000,
          "error_models": [] # To be implemented
        }
      ]
    }

    # --- Generation Workflow ---
    print("--- 1. GENERATION ---")
    generator = RobustDataGenerator(config=example_config)
    dataset_path = 'test_dataset.h5'
    generator.generate_and_save_dataset(dataset_path)

    # --- Loading and Iterating Workflow ---
    print("\n--- 2. LOADING AND SPLITTING ---")
    loaded_generator = RobustDataGenerator.from_h5(dataset_path)
    loaded_generator.create_split(test_size=0.25, seed=123)

    print("\n--- 3. ITERATING (TRAIN) ---")
    train_iterator = loaded_generator.get_iterator('train')
    train_count = 0
    for i, sample in enumerate(train_iterator):
        if i < 3: # Print first 3 samples
            print(f"  Train Sample {i}: meta={sample['meta']}")
        train_count += 1
    print(f"Iterated through {train_count} training samples.")

    print("\n--- 4. ITERATING (TEST) ---")
    test_iterator = loaded_generator.get_iterator('test')
    test_count = 0
    for i, sample in enumerate(test_iterator):
        if i < 3: # Print first 3 samples
            print(f"  Test Sample {i}: meta={sample['meta']}")
        test_count += 1
    print(f"Iterated through {test_count} testing samples.")
