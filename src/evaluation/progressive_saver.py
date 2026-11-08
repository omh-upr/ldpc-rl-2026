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

    Episodes are written to disk as they are flushed rather than accumulated, so
    the collector's episode limit bounds what is held in memory instead of merely
    moving it from the collector to the saver. The output is one gzipped JSON
    document per phase, unchanged in shape from when it was built in memory.

    Two failure rules follow from that:

    - A phase writes to a temporary file and is renamed into place only once it is
      complete, so an interrupted or failed phase never leaves a file that looks
      finished, and collectors are cleared only after that rename.
    - Per-phase state is created in start_phase and dropped in end_phase or
      abort_phase, so one agent's episodes can never appear in the next agent's
      file.
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
        self.saved_files = []

        self._reset_phase_state()

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _reset_phase_state(self):
        """Drop everything belonging to a single phase."""
        self.current_phase = None
        self.current_collectors = None
        self._handle = None
        self._temp_path = None
        self._final_path = None
        self._episodes_written = 0

    def start_phase(self, phase_id: str, collectors: Dict[str, Any]) -> Callable:
        """
        Begin new phase/agent with collectors configured for progressive saving.

        Args:
            phase_id: Identifier for the phase (e.g., "phase1", "agent_rl")
            collectors: Dictionary of collectors to configure with flush callbacks

        Returns:
            Flush callback function for the collectors
        """
        # A phase still open here failed without saying so. Discard it rather than
        # letting its episodes and its open file leak into this one.
        if self.current_phase is not None:
            print(f"   Discarding unfinished phase {self.current_phase}")
            self.abort_phase()

        self.current_phase = phase_id
        self.current_collectors = collectors

        self._final_path = self.output_dir / self._generate_filename(phase_id)
        self._temp_path = self._final_path.with_name(self._final_path.name + '.partial')
        self._handle = gzip.open(self._temp_path, 'wt', encoding='utf-8')
        self._handle.write(
            '{"phase_id": %s, "save_type": %s, "episodes": ['
            % (json.dumps(phase_id), json.dumps(self.save_type))
        )
        self._episodes_written = 0

        # Create flush callback for this phase
        flush_callback = self._create_flush_callback(phase_id)

        # Configure collectors with the flush callback
        for collector in collectors.values():
            if hasattr(collector, 'flush_callback'):
                collector.flush_callback = flush_callback

        return flush_callback

    def end_phase(self) -> str:
        """
        End current phase, writing whatever the collectors still hold.

        Returns:
            Path to the saved file
        """
        if not self.current_phase or self.current_collectors is None:
            raise ValueError("No active phase to end. Call start_phase() first.")

        for collector_name, collector in self.current_collectors.items():
            remaining = getattr(collector, 'episodes', None)
            if remaining:
                self._write_episodes(remaining)
                print(f"   Wrote {len(remaining)} remaining episodes from {collector_name}")

        self._handle.write(']}')
        self._handle.close()
        self._handle = None

        # Rename only after a complete document is on disk, so a reader never sees
        # a partial file under the final name.
        os.replace(self._temp_path, self._final_path)
        print(f"   Saved {self._episodes_written} episodes to {self._final_path.name}")

        # Safe to drop the in-memory copies now, and only now.
        self._clear_collectors()

        saved_path = str(self._final_path)
        self.saved_files.append(saved_path)
        self._reset_phase_state()

        return saved_path

    def abort_phase(self):
        """
        Abandon the current phase and remove its partial file.

        Called when a phase fails. Collectors are left alone: the caller decides
        whether their contents are still wanted.
        """
        if self._handle is not None:
            try:
                self._handle.close()
            except Exception:
                pass
        if self._temp_path is not None and self._temp_path.exists():
            try:
                self._temp_path.unlink()
            except OSError as e:
                print(f"   Could not remove partial file {self._temp_path}: {e}")
        self._reset_phase_state()

    def _write_episodes(self, episodes):
        """Append episodes to the open phase file."""
        if self._handle is None:
            raise RuntimeError(
                "No phase file is open. start_phase() must be called before episodes "
                "can be written, and a flush callback must not outlive its phase.")
        for episode in episodes:
            if self._episodes_written:
                self._handle.write(',')
            self._handle.write(json.dumps(episode, default=self._json_serializer))
            self._episodes_written += 1

    def _create_flush_callback(self, phase_id: str) -> Callable:
        """
        Create phase-specific flush callback for collectors.

        The collector calls this when it hits its memory limit. Writing straight
        through to disk is what makes that limit mean something.
        """
        def flush_callback(collector_data: Any):
            if isinstance(collector_data, list):
                episodes = collector_data
            elif isinstance(collector_data, dict) and 'episodes' in collector_data:
                episodes = collector_data['episodes']
            else:
                episodes = []

            self._write_episodes(episodes)
            print(f"   Wrote {len(episodes)} flushed episodes for {phase_id}")

        return flush_callback

    def _generate_filename(self, phase_id: str) -> str:
        """Generate consistent filename for a phase."""
        return f"{self.base_filename}_{phase_id}.json.gz"

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

    Files are created with pattern:
    {experiment}_{timestamp}_training_epoch{E}_phase{N}_flush{F}.json.gz
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
        self.current_epoch = 1
        self.current_phase = 1
        self.flush_counts = {}  # {(epoch, phase): flush_count}
        self.saved_files = []

    def set_segment(self, epoch: int, phase: int):
        """
        Set the current (epoch, phase) segment, called on phase transition.

        Flush numbering restarts per segment, so a second epoch does not
        continue the numbering of the same phase in the first.

        Args:
            epoch: Current training epoch, 1-based
            phase: New curriculum phase number
        """
        self.current_epoch = epoch
        self.current_phase = phase

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
        epoch, phase = self.current_epoch, self.current_phase
        key = (epoch, phase)
        self.flush_counts[key] = self.flush_counts.get(key, 0) + 1
        flush_num = self.flush_counts[key]

        saved_path = self.metrics_manager.save_training_flush(
            config=self.config,
            training_data=episodes_data,
            epoch=epoch,
            phase=phase,
            flush_num=flush_num
        )
        self.saved_files.append(str(saved_path))
        print(f"   Flush {flush_num} saved for Epoch {epoch} Phase {phase} "
              f"({len(episodes_data)} episodes) -> {saved_path.name}")

    def get_saved_files(self) -> list:
        """Return list of all saved flush files."""
        return self.saved_files.copy()

    def get_flush_count(self, epoch: int = None, phase: int = None) -> int:
        """
        Get flush count for an (epoch, phase) segment.

        Args:
            epoch: Training epoch (defaults to current epoch)
            phase: Phase number (defaults to current phase)

        Returns:
            Number of flushes written for that segment
        """
        epoch = self.current_epoch if epoch is None else epoch
        phase = self.current_phase if phase is None else phase
        return self.flush_counts.get((epoch, phase), 0)