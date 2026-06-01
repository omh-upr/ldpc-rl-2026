"""
File Pattern Utilities

Consistent file naming, discovery, and pattern matching for experiment results.
Supports both legacy single-file format and new agent-separated format.
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
    format_type: str  # "legacy" or "agent_separated"
    files: List[str]
    agent_names: Optional[List[str]] = None


class ExperimentFiles:
    """
    Utility class for consistent experiment file naming and discovery.
    
    Handles both legacy format (single evaluation.json.gz) and new format
    (separate agent files: evaluation_agent_rl.json.gz, etc.)
    """
    
    # File naming patterns
    TRAINING_PATTERN = "{experiment_id}_training.json.gz"
    TRAINING_PHASE_PATTERN = "{experiment_id}_training_phase{phase}.json.gz"
    TRAINING_FLUSH_PATTERN = "{experiment_id}_training_phase{phase}_flush{flush_num}.json.gz"
    EVALUATION_LEGACY_PATTERN = "{experiment_id}_evaluation.json.gz"
    EVALUATION_AGENT_PATTERN = "{experiment_id}_evaluation_agent_{agent}.json.gz"

    # Regex patterns for parsing
    AGENT_FILE_REGEX = re.compile(r'(.+)_evaluation_agent_([^.]+)\.json\.gz$')
    TRAINING_PHASE_REGEX = re.compile(r'(.+)_training_phase(\d+)\.json\.gz$')
    TRAINING_FLUSH_REGEX = re.compile(r'(.+)_training_phase(\d+)_flush(\d+)\.json\.gz$')
    
    @staticmethod
    def get_training_file(experiment_id: str) -> str:
        """Get training file path for experiment."""
        return ExperimentFiles.TRAINING_PATTERN.format(experiment_id=experiment_id)
    
    @staticmethod
    def get_training_phase_file(experiment_id: str, phase: int) -> str:
        """Get training phase file path."""
        return ExperimentFiles.TRAINING_PHASE_PATTERN.format(
            experiment_id=experiment_id,
            phase=phase
        )

    @staticmethod
    def get_training_flush_file(experiment_id: str, phase: int, flush_num: int) -> str:
        """Get training flush file path."""
        return ExperimentFiles.TRAINING_FLUSH_PATTERN.format(
            experiment_id=experiment_id,
            phase=phase,
            flush_num=flush_num
        )
    
    @staticmethod
    def get_evaluation_legacy_file(experiment_id: str) -> str:
        """Get legacy evaluation file path."""
        return ExperimentFiles.EVALUATION_LEGACY_PATTERN.format(experiment_id=experiment_id)
    
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
        
        # Fallback to legacy format
        legacy_file = str(results_path / ExperimentFiles.get_evaluation_legacy_file(experiment_id))
        if os.path.exists(legacy_file):
            return EvaluationFiles(
                format_type="legacy",
                files=[legacy_file]
            )
        
        # No files found
        return EvaluationFiles(
            format_type="none",
            files=[]
        )
    
    @staticmethod
    def find_training_files(experiment_id: str, results_dir: str = "results") -> List[str]:
        """
        Find all training files for an experiment, including flush files.

        Files are sorted by (phase, flush_num) to ensure correct episode ordering:
        - phase1_flush1, phase1_flush2, ..., phase1 (end), phase2_flush1, etc.

        Args:
            experiment_id: Experiment identifier
            results_dir: Directory to search in

        Returns:
            List of training file paths in correct order
        """
        results_path = Path(results_dir)

        # Try phase-separated format first (use wildcard prefix to match experiment names)
        # This pattern matches both phase files and flush files
        phase_pattern = str(results_path / f"*{experiment_id}_training_phase*.json.gz")
        phase_files = glob(phase_pattern)

        if phase_files:
            # Sort by (phase_number, flush_number)
            # Non-flush files (phase-end files) get flush_num=infinity to come last within a phase
            def sort_key(filepath):
                phase = ExperimentFiles.extract_phase_number(filepath) or 0
                flush = ExperimentFiles.extract_flush_number(filepath)
                # Flush files come before phase-end file; use large number for phase-end
                flush_order = flush if flush is not None else float('inf')
                return (phase, flush_order)

            return sorted(phase_files, key=sort_key)

        # Fallback to single training file
        single_file = str(results_path / ExperimentFiles.get_training_file(experiment_id))
        if os.path.exists(single_file):
            return [single_file]

        return []
    
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
            return int(match.group(2))
        # Then try regular phase file pattern
        match = ExperimentFiles.TRAINING_PHASE_REGEX.search(basename)
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
            return int(match.group(3))
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
            'evaluation_format': 'none'  # 'legacy', 'agent_separated', or 'none'
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
    
    @staticmethod
    def get_all_experiment_ids(results_dir: str = "results") -> List[str]:
        """
        Get all experiment IDs found in results directory.
        
        Args:
            results_dir: Directory to search
            
        Returns:
            List of unique experiment IDs
        """
        results_path = Path(results_dir)
        if not results_path.exists():
            return []
        
        experiment_ids = set()
        
        # Pattern to match experiment files
        patterns = [
            "*_training.json.gz",
            "*_training_phase*.json.gz",
            "*_evaluation.json.gz",
            "*_evaluation_agent_*.json.gz"
        ]
        
        for pattern in patterns:
            files = glob(str(results_path / pattern))
            for file_path in files:
                filename = os.path.basename(file_path)
                
                # Extract experiment ID from different patterns
                if "_training_phase" in filename:
                    match = ExperimentFiles.TRAINING_PHASE_REGEX.search(filename)
                    if match:
                        experiment_ids.add(match.group(1))
                elif "_evaluation_agent_" in filename:
                    match = ExperimentFiles.AGENT_FILE_REGEX.search(filename)
                    if match:
                        experiment_ids.add(match.group(1))
                elif "_training.json.gz" in filename:
                    experiment_id = filename.replace("_training.json.gz", "")
                    experiment_ids.add(experiment_id)
                elif "_evaluation.json.gz" in filename:
                    experiment_id = filename.replace("_evaluation.json.gz", "")
                    experiment_ids.add(experiment_id)
        
        return sorted(list(experiment_ids))


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