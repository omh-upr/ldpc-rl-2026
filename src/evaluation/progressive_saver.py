"""
Progressive Results Saving Abstraction

Unified system for saving training phases and evaluation agents progressively
to manage memory usage while preserving all data through intelligent file management.
"""

import os
import gzip
import json
from typing import Dict, Any, Callable, Optional
from pathlib import Path


class ProgressiveSaver:
    """
    Unified progressive saving for training phases and evaluation agents.
    
    Provides consistent memory management and file creation across different
    contexts while maintaining clean separation between training and evaluation data.
    """
    
    def __init__(self, base_filename: str, save_type: str = "training", output_dir: str = "results"):
        """
        Initialize progressive saver with base configuration.
        
        Args:
            base_filename: Base name for generated files (without extension)
            save_type: Type of saving - "training" or "evaluation"
            output_dir: Directory where files will be saved
        """
        self.base_filename = base_filename
        self.save_type = save_type
        self.output_dir = Path(output_dir)
        self.current_phase = None
        self.current_collectors = None
        self.saved_files = []
        self._temp_data = []
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def start_phase(self, phase_id: str, collectors: Dict[str, Any]) -> Callable:
        """
        Begin new phase/agent with collectors configured for progressive saving.
        
        Args:
            phase_id: Identifier for the phase (e.g., "phase1", "agent_rl")
            collectors: Dictionary of collectors to configure with flush callbacks
            
        Returns:
            Flush callback function for the collectors
        """
        self.current_phase = phase_id
        self.current_collectors = collectors
        
        # Create flush callback for this phase
        flush_callback = self._create_flush_callback(phase_id)
        
        # Configure collectors with the flush callback
        for collector in collectors.values():
            if hasattr(collector, 'flush_callback'):
                collector.flush_callback = flush_callback
        
        return flush_callback
    
    def end_phase(self) -> str:
        """
        End current phase and save all remaining data.
        
        Returns:
            Path to the saved file
        """
        if not self.current_phase or not self.current_collectors:
            raise ValueError("No active phase to end. Call start_phase() first.")
        
        # Generate filename for this phase
        filename = self._generate_filename(self.current_phase)
        filepath = self.output_dir / filename
        
        # Collect all data from collectors
        phase_data = self._collect_phase_data()
        
        # Save the data
        self._save_data(filepath, phase_data)
        
        # Clear current phase state
        self.current_phase = None
        self.current_collectors = None
        
        # Track saved file
        self.saved_files.append(str(filepath))
        
        return str(filepath)
    
    def _create_flush_callback(self, phase_id: str) -> Callable:
        """
        Create phase-specific flush callback for collectors.
        
        This callback will be called by collectors when they need to flush
        data due to memory limits, allowing the progressive saver to handle
        the data appropriately.
        """
        def flush_callback(collector_data: Dict[str, Any]):
            # Add the flushed data to our temporary storage
            if isinstance(collector_data, list):
                self._temp_data.extend(collector_data)
            elif isinstance(collector_data, dict) and 'episodes' in collector_data:
                self._temp_data.extend(collector_data['episodes'])
            
            print(f"   Temporarily stored {len(collector_data) if isinstance(collector_data, list) else 'unknown'} episodes for {phase_id}")
        
        return flush_callback
    
    def _generate_filename(self, phase_id: str) -> str:
        """Generate consistent filename for a phase."""
        return f"{self.base_filename}_{phase_id}.json.gz"
    
    def _collect_phase_data(self) -> Dict[str, Any]:
        """Collect all data from current collectors and temporary storage."""
        phase_data = {
            'phase_id': self.current_phase,
            'save_type': self.save_type,
            'episodes': []
        }
        
        # Collect any temporarily stored data from flushes
        if self._temp_data:
            phase_data['episodes'].extend(self._temp_data)
            print(f"   Collected {len(self._temp_data)} episodes from temporary storage")

        # Collect remaining data from all collectors
        for collector_name, collector in self.current_collectors.items():
            if hasattr(collector, 'episodes'):
                phase_data['episodes'].extend(collector.episodes)
                print(f"   Collected {len(collector.episodes)} remaining episodes from {collector_name}")

        # Clear temporary data
        self._temp_data = []
        
        # Clear all collectors to free memory (like curriculum phases do)
        self._clear_collectors()
        
        return phase_data
    
    def _clear_collectors(self):
        """Clear all episode data from collectors to free memory."""
        total_episodes_cleared = 0
        
        for collector_name, collector in self.current_collectors.items():
            if hasattr(collector, 'episodes'):
                episodes_count = len(collector.episodes)
                collector.episodes = []  # Clear the episodes list
                total_episodes_cleared += episodes_count
                
                # Also reset episode count if it exists
                if hasattr(collector, 'episode_count'):
                    collector.episode_count = 0
                
                # Reset any other collection state
                if hasattr(collector, 'reset'):
                    collector.reset()
        
        if total_episodes_cleared > 0:
            print(f"   Cleared {total_episodes_cleared} episodes from memory after saving")
    
    def _save_data(self, filepath: Path, data: Dict[str, Any]):
        """Save data to compressed JSON file."""
        with gzip.open(filepath, 'wt', encoding='utf-8') as f:
            json.dump(data, f, default=self._json_serializer)
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy and torch objects."""
        import numpy as np
        import torch
        
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.void):
            # Handle numpy structured data types (e.g., metadata from HDF5)
            if obj.dtype.names:
                # Convert structured array to dictionary
                return {name: self._json_serializer(obj[name]) for name in obj.dtype.names}
            else:
                # Convert to Python object
                return obj.item()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.str_, str)):
            return str(obj)
        elif isinstance(obj, bytes):
            return obj.decode('utf-8', errors='replace')
        
        # Let the default serializer handle other types
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def get_saved_files(self) -> list:
        """Get list of all files saved by this saver."""
        return self.saved_files.copy()
    


class TrainingProgressiveSaver(ProgressiveSaver):
    """Specialized progressive saver for training phases."""
    
    def __init__(self, base_filename: str, output_dir: str = "results"):
        super().__init__(base_filename, "training", output_dir)
    
    def start_curriculum_phase(self, phase_number: int, collectors: Dict[str, Any]) -> Callable:
        """Start a curriculum training phase."""
        phase_id = f"phase{phase_number}"
        return self.start_phase(phase_id, collectors)


class EvaluationProgressiveSaver(ProgressiveSaver):
    """Specialized progressive saver for evaluation agents."""

    def __init__(self, base_filename: str, output_dir: str = "results"):
        super().__init__(base_filename, "evaluation", output_dir)

    def start_agent_evaluation(self, agent_name: str, collectors: Dict[str, Any]) -> Callable:
        """Start evaluation for a specific agent."""
        phase_id = f"agent_{agent_name}"
        return self.start_phase(phase_id, collectors)

    def _generate_filename(self, phase_id: str) -> str:
        """Generate agent-specific filename."""
        return f"{self.base_filename}_{phase_id}.json.gz"


class TrainingFlushSaver:
    """
    Immediate disk write saver for training flush events.

    Unlike TrainingProgressiveSaver which buffers in memory until phase end,
    this class writes flushed data immediately to disk. This prevents data loss
    when a single training phase has more episodes than can fit in memory.

    Files are created with pattern: {experiment}_{timestamp}_training_phase{phase}_flush{N}.json.gz
    """

    def __init__(self, config: Dict[str, Any], metrics_manager):
        """
        Initialize flush saver.

        Args:
            config: Experiment configuration (saved with each flush file)
            metrics_manager: MetricsManager instance for saving files
        """
        import copy
        self.config = copy.deepcopy(config)
        self.metrics_manager = metrics_manager
        self.current_phase = 1
        self.flush_counts = {}  # {phase: flush_count}
        self.saved_files = []

    def set_phase(self, phase: int):
        """
        Set current phase (called on phase transition).

        Args:
            phase: New curriculum phase number
        """
        self.current_phase = phase
        if phase not in self.flush_counts:
            self.flush_counts[phase] = 0

    def create_flush_callback(self) -> Callable:
        """
        Create callback for EpisodeDataCollector flush events.

        The callback will be invoked when the collector hits its memory limit,
        and will immediately write the data to disk.

        Returns:
            Callable that accepts list of episode dicts
        """
        def flush_callback(episodes_data: list):
            self._save_flush_immediately(episodes_data)
        return flush_callback

    def _save_flush_immediately(self, episodes_data: list):
        """
        Save flush data to disk immediately.

        Args:
            episodes_data: List of episode dictionaries to save
        """
        phase = self.current_phase
        self.flush_counts[phase] = self.flush_counts.get(phase, 0) + 1
        flush_num = self.flush_counts[phase]

        saved_path = self.metrics_manager.save_training_flush(
            config=self.config,
            training_data=episodes_data,
            phase=phase,
            flush_num=flush_num
        )
        self.saved_files.append(str(saved_path))
        print(f"   Flush {flush_num} saved for Phase {phase} ({len(episodes_data)} episodes) -> {saved_path.name}")

    def get_saved_files(self) -> list:
        """Return list of all saved flush files."""
        return self.saved_files.copy()

    def get_flush_count(self, phase: int = None) -> int:
        """
        Get flush count for a phase.

        Args:
            phase: Phase number (defaults to current phase)

        Returns:
            Number of flushes for that phase
        """
        if phase is None:
            phase = self.current_phase
        return self.flush_counts.get(phase, 0)