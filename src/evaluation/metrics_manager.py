"""
Metrics Management System for RL LDPC Decoder

Simplified metrics collection and saving for LDPC decoder experiments.
"""

import json
import gzip
import torch
from datetime import datetime
from pathlib import Path
import numpy as np

class MetricsManager:
    """
    Metrics saving and results management for LDPC experiments.
    """
    
    def __init__(self, experiment_name="ldpc_rl_experiment", output_dir=None, models_dir=None):
        """
        Initialize the MetricsManager.
        
        Args:
            experiment_name: Name prefix for saved files
            output_dir: Directory for results and metrics (defaults to current directory)
            models_dir: Directory for model checkpoints (defaults to output_dir/models)
        """
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up directories
        if output_dir:
            self.results_dir = Path(output_dir)
        else:
            self.results_dir = Path(__file__).parent
        
        if models_dir:
            self.models_dir = Path(models_dir)
        else:
            # Fallback: use models subdirectory in results_dir for backward compatibility
            self.models_dir = self.results_dir / "models"
        
        # Create directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def save_results(self, results: dict) -> Path:
        """
        Save the final results dictionary to a compressed JSON file.

        Args:
            results: The final results dictionary to save.

        Returns:
            The path to the saved results file.
        """
        results_path = self.results_dir / f"{self.experiment_name}_{self.timestamp}.json.gz"
        with gzip.open(results_path, 'wt') as f:
            processed_data = self._preprocess_for_json(results)
            json.dump(processed_data, f, default=self._json_serializer)
        return results_path

    def save_training_results(self, config: dict, training_data: list) -> Path:
        """
        Save training results to a separate compressed file with config.
        Filters out evaluation episodes to prevent data corruption.
        
        Args:
            config: Experiment configuration
            training_data: Training episodes data
            
        Returns:
            The path to the saved training results file.
        """
        # Filter out evaluation episodes that may have leaked into training data
        filtered_training_data = []
        eval_episodes_found = 0
        
        for episode in training_data:
            episode_type = episode.get('episode_type', 'unknown')
            if episode_type == 'train':
                filtered_training_data.append(episode)
            elif episode_type in ['val', 'eval']:  # Accept both validation and evaluation
                eval_episodes_found += 1
            else:
                # Unknown episode type - include with warning
                filtered_training_data.append(episode)
                print(f"Warning: Found episode with unknown type '{episode_type}' in training data")
        
        # Log filtering results
        if eval_episodes_found > 0:
            print(f"Filtered out {eval_episodes_found} evaluation episodes from training data")
            print(f"Keeping {len(filtered_training_data)} actual training episodes")
        
        training_results = {
            'experiment_configuration': config,
            'training_episodes': filtered_training_data
        }
        
        training_path = self.results_dir / f"{self.experiment_name}_{self.timestamp}_training.json.gz"
        with gzip.open(training_path, 'wt') as f:
            processed_data = self._preprocess_for_json(training_results)
            json.dump(processed_data, f, default=self._json_serializer)
        return training_path

    def save_evaluation_results(self, config: dict, evaluation_data: list) -> Path:
        """
        Save evaluation results to a separate compressed file with config.
        
        Args:
            config: Experiment configuration
            evaluation_data: Evaluation episodes data
            
        Returns:
            The path to the saved evaluation results file.
        """
        evaluation_results = {
            'experiment_configuration': config,
            'evaluation_episodes': evaluation_data
        }
        
        evaluation_path = self.results_dir / f"{self.experiment_name}_{self.timestamp}_evaluation.json.gz"
        with gzip.open(evaluation_path, 'wt') as f:
            processed_data = self._preprocess_for_json(evaluation_results)
            json.dump(processed_data, f, default=self._json_serializer)
        return evaluation_path

    def save_model(self, model_state_dict, model_info=None):
        """
        Save model weights with metadata.
        
        Args:
            model_state_dict: PyTorch model state dictionary
            model_info: Additional model information
        """
        model_data = {
            "state_dict": model_state_dict,
            "metadata": {
                "experiment_name": self.experiment_name,
                "timestamp": self.timestamp,
                "model_info": model_info or {}
            }
        }
        
        # Save model
        model_path = self.models_dir / f"ldpc_rl_agent_{self.timestamp}.pth"
        torch.save(model_data, model_path)
        
        return model_path
    
    def _preprocess_for_json(self, obj):
        """
        Recursively preprocess data to ensure JSON compatibility.
        Converts numpy types and ensures all dictionary keys are strings.
        """
        if isinstance(obj, dict):
            # Convert all keys to strings and recursively process values
            return {str(k): self._preprocess_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            # Recursively process list/tuple elements
            return [self._preprocess_for_json(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for remaining non-serializable objects."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable: {obj!r}")
    
    def save_training_partial(self, config: dict, training_data: list, phase: int, clear_memory: bool = False) -> Path:
        """
        Save partial training results for a specific phase to prevent memory accumulation.

        Args:
            config: Experiment configuration
            training_data: Training episodes data for this phase
            phase: Curriculum phase number
            clear_memory: Deprecated, ignored. Callers should manage their own cleanup.

        Returns:
            The path to the saved partial training results file.
        """
        partial_results = {
            'experiment_configuration': config,
            'curriculum_phase': phase,
            'training_episodes': training_data
        }

        # Use phase-specific filename to avoid conflicts
        partial_path = self.results_dir / f"{self.experiment_name}_{self.timestamp}_training_phase{phase}.json.gz"
        with gzip.open(partial_path, 'wt') as f:
            processed_data = self._preprocess_for_json(partial_results)
            json.dump(processed_data, f, default=self._json_serializer)

        return partial_path

    def save_training_flush(self, config: dict, training_data: list,
                            phase: int, flush_num: int) -> Path:
        """
        Save flush data immediately to disk within a training phase.

        Called when memory limits are hit during training to prevent data loss.
        Creates files with pattern: {experiment}_{timestamp}_training_phase{phase}_flush{N}.json.gz

        Args:
            config: Experiment configuration
            training_data: Training episodes data for this flush
            phase: Current curriculum phase number
            flush_num: Sequential flush number within the phase

        Returns:
            Path to the saved flush file
        """
        flush_results = {
            'experiment_configuration': config,
            'curriculum_phase': phase,
            'flush_number': flush_num,
            'training_episodes': training_data
        }

        # Use phase and flush-specific filename
        flush_path = self.results_dir / f"{self.experiment_name}_{self.timestamp}_training_phase{phase}_flush{flush_num}.json.gz"
        with gzip.open(flush_path, 'wt') as f:
            processed_data = self._preprocess_for_json(flush_results)
            json.dump(processed_data, f, default=self._json_serializer)

        return flush_path

    def append_to_training_results(self, training_data: list) -> None:
        """
        Append new training data to the main training results file.
        This allows incremental saving without losing existing data.
        
        Args:
            training_data: New training episodes to append
        """
        main_training_path = self.results_dir / f"{self.experiment_name}_{self.timestamp}_training.json.gz"
        
        # If main file exists, read it and append
        if main_training_path.exists():
            with gzip.open(main_training_path, 'rt') as f:
                existing_data = json.load(f)
            existing_data['training_episodes'].extend(training_data)
        else:
            # Create new structure
            existing_data = {
                'experiment_configuration': {},  # Will be set properly when config is available
                'training_episodes': training_data
            }
        
        # Save the combined data
        with gzip.open(main_training_path, 'wt') as f:
            processed_data = self._preprocess_for_json(existing_data)
            json.dump(processed_data, f, default=self._json_serializer)
    
