"""
File Pattern Utilities

Consistent file naming, discovery, and pattern matching for experiment results.
Only the current formats are read: (epoch, phase) training segments
and agent-separated evaluation files.
"""

import os
import re
from glob import glob
from pathlib import Path
from typing import List, Optional, Dict, NamedTuple, Any
from dataclasses import dataclass


@dataclass
class EvaluationFiles:
    """Container for evaluation file information."""
    format_type: str  # "agent_separated" or "none"
    files: List[str]
    agent_names: Optional[List[str]] = None


class ExperimentFiles:
    """
    Utility class for consistent experiment file naming and discovery.
    
    Handles the current formats only: training segments named by (epoch, phase)
    and one evaluation file per agent (evaluation_agent_rl.json.gz, etc.).
    """
    
    # File naming patterns
    TRAINING_SINGLE_PATTERN = "{experiment_id}_training.json.gz"
    TRAINING_PHASE_PATTERN = "{experiment_id}_training_epoch{epoch}_phase{phase}.json.gz"
    TRAINING_FLUSH_PATTERN = "{experiment_id}_training_epoch{epoch}_phase{phase}_flush{flush_num}.json.gz"
    EVALUATION_AGENT_PATTERN = "{experiment_id}_evaluation_agent_{agent}.json.gz"

    # Regex patterns for parsing
    AGENT_FILE_REGEX = re.compile(r'(.+)_evaluation_agent_([^.]+)\.json\.gz$')
    TRAINING_SINGLE_REGEX = re.compile(r'(.+)_training\.json\.gz$')
    TRAINING_PHASE_REGEX = re.compile(r'(.+)_training_epoch(\d+)_phase(\d+)\.json\.gz$')
    TRAINING_FLUSH_REGEX = re.compile(r'(.+)_training_epoch(\d+)_phase(\d+)_flush(\d+)\.json\.gz$')
    
    @staticmethod
    def get_training_phase_file(experiment_id: str, epoch: int, phase: int) -> str:
        """Get the training file path for one (epoch, phase) segment."""
        return ExperimentFiles.TRAINING_PHASE_PATTERN.format(
            experiment_id=experiment_id,
            epoch=epoch,
            phase=phase
        )

    @staticmethod
    def get_training_flush_file(experiment_id: str, epoch: int, phase: int,
                                flush_num: int) -> str:
        """Get the training flush file path within one segment."""
        return ExperimentFiles.TRAINING_FLUSH_PATTERN.format(
            experiment_id=experiment_id,
            epoch=epoch,
            phase=phase,
            flush_num=flush_num
        )
    
    @staticmethod
    def get_evaluation_agent_file(experiment_id: str, agent_name: str) -> str:
        """Get agent-specific evaluation file path."""
        return ExperimentFiles.EVALUATION_AGENT_PATTERN.format(
            experiment_id=experiment_id,
            agent=agent_name
        )
    
    @staticmethod
    def find_evaluation_files(experiment_id: str, results_dir: str = "results") -> EvaluationFiles:
        """
        Find all evaluation files for an experiment.
        
        Args:
            experiment_id: Experiment identifier
            results_dir: Directory to search in
            
        Returns:
            EvaluationFiles object with discovered files
        """
        results_path = Path(results_dir)
        
        # Try agent-separated format first (use wildcard prefix to match experiment names)
        agent_pattern = str(results_path / f"*{experiment_id}_evaluation_agent_*.json.gz")
        agent_files = glob(agent_pattern)
        
        if agent_files:
            # Extract agent names from filenames
            agent_names = []
            for file_path in agent_files:
                agent_name = ExperimentFiles.extract_agent_name(file_path)
                if agent_name:
                    agent_names.append(agent_name)
            
            return EvaluationFiles(
                format_type="agent_separated",
                files=sorted(agent_files),
                agent_names=sorted(agent_names)
            )
        
        # No files found. Agent-separated files are the only evaluation output the
        # writer produces, and this module reads no legacy form, so there is
        # nothing else to look for.
        return EvaluationFiles(
            format_type="none",
            files=[]
        )
    
    @staticmethod
    def find_training_files(experiment_id: str, results_dir: str = "results") -> List[str]:
        """
        Find all training files for an experiment.

        Covers both persistence policies: the (epoch, phase) segments and their
        flush files written when progressive saving is on, and the single file
        written at the end of the run when it is off. A run uses one policy or the
        other, so the two cannot both be present and nothing is double counted.

        Order is not significant: every episode carries a run-global episode_id,
        and the analysis layer sorts on that. Returned sorted by name only so
        that logs and file listings are reproducible.

        Args:
            experiment_id: Experiment identifier
            results_dir: Directory to search in

        Returns:
            List of training file paths for the experiment
        """
        results_path = Path(results_dir)
        files = glob(str(results_path / f"*{experiment_id}_training_epoch*_phase*.json.gz"))
        files += glob(str(results_path / f"*{experiment_id}_training.json.gz"))
        return sorted(files)
    
    @staticmethod
    def extract_agent_name(filename: str) -> Optional[str]:
        """
        Extract agent name from evaluation file path.
        
        Args:
            filename: File path or name
            
        Returns:
            Agent name or None if not an agent file
        """
        match = ExperimentFiles.AGENT_FILE_REGEX.search(os.path.basename(filename))
        if match:
            return match.group(2)
        return None
    
    @staticmethod
    def extract_phase_number(filename: str) -> Optional[int]:
        """
        Extract phase number from training phase file.

        Args:
            filename: File path or name

        Returns:
            Phase number or None if not a phase file
        """
        basename = os.path.basename(filename)
        # Try flush file pattern first (more specific)
        match = ExperimentFiles.TRAINING_FLUSH_REGEX.search(basename)
        if match:
            return int(match.group(3))
        # Then try regular phase file pattern
        match = ExperimentFiles.TRAINING_PHASE_REGEX.search(basename)
        if match:
            return int(match.group(3))
        return None

    @staticmethod
    def extract_epoch_number(filename: str) -> Optional[int]:
        """
        Extract the epoch number from a training file name.

        Args:
            filename: Training phase or flush file name

        Returns:
            Epoch number, or None if the name matches neither pattern
        """
        basename = os.path.basename(filename)
        for regex in (ExperimentFiles.TRAINING_FLUSH_REGEX,
                      ExperimentFiles.TRAINING_PHASE_REGEX):
            match = regex.search(basename)
            if match:
                return int(match.group(2))
        return None

    @staticmethod
    def extract_flush_number(filename: str) -> Optional[int]:
        """
        Extract flush number from training flush file.

        Args:
            filename: File path or name

        Returns:
            Flush number or None if not a flush file
        """
        match = ExperimentFiles.TRAINING_FLUSH_REGEX.search(os.path.basename(filename))
        if match:
            return int(match.group(4))
        return None

    @staticmethod
    def is_flush_file(filename: str) -> bool:
        """
        Check if a filename is a training flush file.

        Args:
            filename: File path or name

        Returns:
            True if this is a flush file
        """
        return ExperimentFiles.TRAINING_FLUSH_REGEX.search(os.path.basename(filename)) is not None
    
    @staticmethod
    def validate_experiment_files(experiment_id: str, results_dir: str = "results") -> Dict[str, bool]:
        """
        Validate that expected files exist for an experiment.
        
        Args:
            experiment_id: Experiment identifier
            results_dir: Directory to check
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'has_training_files': False,
            'has_evaluation_files': False,
            'training_format': 'none',  # 'single', 'phases', or 'none'
            'evaluation_format': 'none'  # 'agent_separated' or 'none'
        }
        
        # Check training files
        training_files = ExperimentFiles.find_training_files(experiment_id, results_dir)
        if training_files:
            validation['has_training_files'] = True
            validation['training_format'] = 'phases' if len(training_files) > 1 else 'single'
        
        # Check evaluation files
        eval_files = ExperimentFiles.find_evaluation_files(experiment_id, results_dir)
        if eval_files.files:
            validation['has_evaluation_files'] = True
            validation['evaluation_format'] = eval_files.format_type
        
        return validation
    
    # An experiment id is the run timestamp, which is what every discovery helper
    # here takes and what the analysis CLI is given.
    TIMESTAMP_REGEX = re.compile(r'(\d{8}_\d{6})')

    @staticmethod
    def get_all_experiment_ids(results_dir: str = "results") -> List[str]:
        """
        Get all experiment IDs found in results directory.

        Searches the formats this module reads: (epoch, phase) training segments,
        their flush files, and per-agent evaluation files. It previously looked for
        an older phase pattern and never for the epoch pattern, so a training-only
        run produced no identifiers at all.

        Args:
            results_dir: Directory to search

        Returns:
            List of unique experiment IDs
        """
        results_path = Path(results_dir)
        if not results_path.exists():
            return []

        experiment_ids = set()

        patterns = [
            "*_training_epoch*_phase*.json.gz",
            "*_training.json.gz",
            "*_evaluation_agent_*.json.gz"
        ]

        for pattern in patterns:
            for file_path in glob(str(results_path / pattern)):
                filename = os.path.basename(file_path)
                for regex in (ExperimentFiles.TRAINING_FLUSH_REGEX,
                              ExperimentFiles.TRAINING_PHASE_REGEX,
                              ExperimentFiles.TRAINING_SINGLE_REGEX,
                              ExperimentFiles.AGENT_FILE_REGEX):
                    match = regex.search(filename)
                    if not match:
                        continue
                    # group(1) is "{experiment name}_{timestamp}"; the timestamp
                    # alone is the identifier the rest of the module expects.
                    stamps = ExperimentFiles.TIMESTAMP_REGEX.findall(match.group(1))
                    experiment_ids.add(stamps[-1] if stamps else match.group(1))
                    break

        return sorted(experiment_ids)


class ResultsDirectoryManager:
    """
    Manager for results directory operations and organization.
    
    Provides utilities for managing experiment results across different formats.
    """
    
    def __init__(self, results_dir: str = "results"):
        """Initialize with results directory."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def list_experiments(self) -> Dict[str, Dict[str, Any]]:
        """
        List all experiments with their file information.
        
        Returns:
            Dictionary mapping experiment IDs to their file info
        """
        experiments = {}
        
        for experiment_id in ExperimentFiles.get_all_experiment_ids(str(self.results_dir)):
            experiments[experiment_id] = ExperimentFiles.validate_experiment_files(
                experiment_id, str(self.results_dir)
            )
        
        return experiments
    
    def cleanup_empty_files(self) -> List[str]:
        """
        Find and optionally remove empty result files.
        
        Returns:
            List of empty files found
        """
        empty_files = []
        
        for file_path in self.results_dir.glob("*.json.gz"):
            if file_path.stat().st_size == 0:
                empty_files.append(str(file_path))
        
        return empty_files
    
    def get_experiment_summary(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get detailed summary of an experiment's files.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Dictionary with experiment file summary
        """
        summary = {
            'experiment_id': experiment_id,
            'training_files': ExperimentFiles.find_training_files(experiment_id, str(self.results_dir)),
            'evaluation_files': ExperimentFiles.find_evaluation_files(experiment_id, str(self.results_dir)),
            'validation': ExperimentFiles.validate_experiment_files(experiment_id, str(self.results_dir))
        }
        
        return summary