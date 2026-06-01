#!/usr/bin/env python3
"""
Multi-Matrix LDPC RL Experiment Orchestrator

Automates the complete 4-stage LDPC RL experiment workflow across multiple
LDPC codes with comprehensive logging, error handling, and configuration
discovery.

Supports two discovery modes:
  - Matrix mode: Only folders named with code_id format (e.g., '64x128')
  - Folder mode: All folders with valid configs (including ablation variants)

4-Stage Pipeline:
  Stage 1: Main curriculum dataset creation (200K samples)
  Stage 2: Expert curriculum dataset creation (30K samples)
  Stage 3: Expert experience generation
  Stage 4: Complete experiment (training + evaluation)

Usage Examples:
  # Quick smoke test (~5 min on a single GPU; validates the full 4-stage workflow
  # end-to-end before committing time to the flagship run)
  PYTHONPATH=. python scripts/run_multi_matrix_experiments.py \
      --config-folder config/experiments_config/ --folders 64x128_QuickStart

  # Run all matrix experiments (code_id format folders only)
  PYTHONPATH=. python scripts/run_multi_matrix_experiments.py \
      --config-folder config/experiments_config/ --all-matrices

  # Run ALL experiments including ablation variants
  PYTHONPATH=. python scripts/run_multi_matrix_experiments.py \
      --config-folder config/experiments_config/ --all-folders

  # Run specific matrix experiments
  PYTHONPATH=. python scripts/run_multi_matrix_experiments.py \
      --config-folder config/experiments_config/ --matrices 24x48 64x128

  # Run specific ablation variant folders
  PYTHONPATH=. python scripts/run_multi_matrix_experiments.py \
      --config-folder config/experiments_config/ --folders 64x128_NoC_NoE 64x128_Full

  # Dry run (validation only)
  PYTHONPATH=. python scripts/run_multi_matrix_experiments.py \
      --config-folder config/experiments_config/ --all-folders --dry-run

  # Override expert agent and output directory
  PYTHONPATH=. python scripts/run_multi_matrix_experiments.py \
      --config-folder config/experiments_config/ --all-matrices \
      --expert omniscient --output-dir exp_v2
"""

import argparse
import subprocess
import sys
import time
import gc
import json
import yaml
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class ExperimentDiscovery:
    """Discovers and validates experiment configurations."""

    # Required config files per stage
    REQUIRED_CONFIGS = {
        'stage1': 'main_dataset_config.json',
        'stage2': 'expert_dataset_config.json',
        'stage3': 'expert_experience_config.yml',
        'stage4': 'experiment_config.yml'
    }

    def __init__(self, config_folder: Path):
        """Initialize discovery with root config folder."""
        self.config_folder = config_folder
        self.discovered_experiments = {}

    def discover_all_matrices(self) -> Dict[str, Dict]:
        """
        Discover all valid experiments in subfolders with code_id format names.

        Only discovers subfolders named with code_id format (e.g., '64x128').

        Returns:
            Dict mapping folder_name to experiment config paths
        """
        if not self.config_folder.exists():
            raise FileNotFoundError(f"Config folder not found: {self.config_folder}")

        # Scan for code_id subfolders
        for subfolder in sorted(self.config_folder.iterdir()):
            if not subfolder.is_dir():
                continue

            folder_name = subfolder.name

            # Only include folders with valid code_id format
            if not self._is_valid_code_id(folder_name):
                print(f"Skipping non-matrix folder: {folder_name}")
                continue

            # Try to find all required configs
            config_paths = self._find_configs(subfolder)

            if config_paths:
                self.discovered_experiments[folder_name] = {
                    'folder_name': folder_name,
                    'code_id': folder_name,  # For matrix folders, code_id = folder_name
                    'config_dir': subfolder,
                    'configs': config_paths
                }
                print(f"Discovered matrix experiment: {folder_name}")
            else:
                print(f"Incomplete configs for {folder_name}, skipping")

        return self.discovered_experiments

    def discover_all_folders(self) -> Dict[str, Dict]:
        """
        Discover all valid experiments in ALL subfolders (including non-code_id names).

        This method discovers ablation variants like '64x128_NoC_NoE' by reading
        the code_id from experiment_config.yml.

        Returns:
            Dict mapping folder_name to experiment config paths
        """
        if not self.config_folder.exists():
            raise FileNotFoundError(f"Config folder not found: {self.config_folder}")

        # Scan ALL subfolders
        for subfolder in sorted(self.config_folder.iterdir()):
            if not subfolder.is_dir():
                continue

            folder_name = subfolder.name

            # Try to find all required configs
            config_paths = self._find_configs(subfolder)

            if config_paths:
                # Extract code_id from experiment_config.yml
                code_id = self._extract_code_id_from_config(config_paths['stage4'])

                if code_id is None:
                    print(f"No code_id in experiment_config.yml for {folder_name}, skipping")
                    continue

                self.discovered_experiments[folder_name] = {
                    'folder_name': folder_name,
                    'code_id': code_id,
                    'config_dir': subfolder,
                    'configs': config_paths
                }
                is_variant = folder_name != code_id
                variant_note = f" (variant of {code_id})" if is_variant else ""
                print(f"Discovered experiment: {folder_name}{variant_note}")
            else:
                print(f"Incomplete configs for {folder_name}, skipping")

        return self.discovered_experiments

    def discover_specific_matrices(self, code_ids: List[str]) -> Dict[str, Dict]:
        """
        Discover specific experiments by code_id (folder name = code_id).

        Args:
            code_ids: List of code_ids to discover

        Returns:
            Dict mapping code_id to experiment config paths
        """
        for code_id in code_ids:
            subfolder = self.config_folder / code_id

            if not subfolder.exists():
                raise FileNotFoundError(f"Config folder not found for {code_id}: {subfolder}")

            config_paths = self._find_configs(subfolder)

            if not config_paths:
                raise ValueError(f"Incomplete configuration for {code_id}")

            self.discovered_experiments[code_id] = {
                'folder_name': code_id,
                'code_id': code_id,
                'config_dir': subfolder,
                'configs': config_paths
            }
            print(f"Discovered matrix experiment: {code_id}")

        return self.discovered_experiments

    def discover_specific_folders(self, folder_names: List[str]) -> Dict[str, Dict]:
        """
        Discover specific experiments by folder name (may or may not be code_id format).

        For ablation studies where folder names like '64x128_NoC_NoE' contain
        experiment configs for code_id '64x128'.

        Args:
            folder_names: List of folder names to discover

        Returns:
            Dict mapping folder_name to experiment config paths
        """
        for folder_name in folder_names:
            subfolder = self.config_folder / folder_name

            if not subfolder.exists():
                raise FileNotFoundError(f"Config folder not found: {subfolder}")

            config_paths = self._find_configs(subfolder)

            if not config_paths:
                raise ValueError(f"Incomplete configuration for {folder_name}")

            # Extract code_id from experiment_config.yml
            code_id = self._extract_code_id_from_config(config_paths['stage4'])

            if code_id is None:
                raise ValueError(f"No code_id found in experiment_config.yml for {folder_name}")

            self.discovered_experiments[folder_name] = {
                'folder_name': folder_name,
                'code_id': code_id,
                'config_dir': subfolder,
                'configs': config_paths
            }
            is_variant = folder_name != code_id
            variant_note = f" (variant of {code_id})" if is_variant else ""
            print(f"Discovered experiment: {folder_name}{variant_note}")

        return self.discovered_experiments

    def _extract_code_id_from_config(self, config_path: Path) -> Optional[str]:
        """
        Extract code_id from experiment_config.yml.

        Args:
            config_path: Path to experiment_config.yml

        Returns:
            code_id string or None if not found
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config.get('code_id')
        except Exception as e:
            print(f"   Error reading code_id from {config_path}: {e}")
            return None

    def _find_configs(self, subfolder: Path) -> Optional[Dict[str, Path]]:
        """
        Find all required config files in subfolder.

        Returns:
            Dict mapping stage to config path, or None if incomplete
        """
        config_paths = {}

        for stage, filename in self.REQUIRED_CONFIGS.items():
            config_file = subfolder / filename

            if not config_file.exists():
                print(f"   Missing {filename} for {subfolder.name}")
                return None

            config_paths[stage] = config_file
            print(f"   Found {filename}")

        return config_paths

    def _is_valid_code_id(self, code_id: str) -> bool:
        """
        Validate strict code_id format (e.g., '24x48', '128x256').

        Only matches pure 'NxM' matrix dimension folders.
        Folders with suffixes (e.g., '64x128_Full') are excluded;
        use --all-folders to include those.

        Returns:
            True if valid format
        """
        pattern = r'^\d+x\d+$'
        return bool(re.match(pattern, code_id))

    def validate_configs(self, code_id: str, config_paths: Dict[str, Path]) -> bool:
        """
        Validate that config files are properly formatted and contain required fields.

        Args:
            code_id: The LDPC code identifier
            config_paths: Dict mapping stage to config file path

        Returns:
            True if all configs are valid
        """
        try:
            # Validate Stage 1 & 2 (JSON curriculum configs)
            for stage in ['stage1', 'stage2']:
                config_path = config_paths[stage]
                with open(config_path, 'r') as f:
                    config = json.load(f)

                # Check required curriculum fields
                required = ['curriculum', 'seed', 'dataset_name']
                for field in required:
                    if field not in config:
                        print(f"   {stage}: Missing required field '{field}'")
                        return False

                print(f"   {stage}: Valid curriculum config")

            # Validate Stage 3 (YAML expert experience config)
            stage3_path = config_paths['stage3']
            with open(stage3_path, 'r') as f:
                config = yaml.safe_load(f)

            required = ['code_id', 'agent_type', 'output_file', 'dataset']
            for field in required:
                if field not in config:
                    print(f"   stage3: Missing required field '{field}'")
                    return False

            # Verify code_id matches
            if config['code_id'] != code_id:
                print(f"   stage3: code_id mismatch ({config['code_id']} != {code_id})")
                return False

            print(f"   stage3: Valid expert experience config")

            # Validate Stage 4 (complete experiment YAML config)
            stage4_path = config_paths['stage4']
            with open(stage4_path, 'r') as f:
                config = yaml.safe_load(f)

            # Check code_id if present
            if 'code_id' in config and config['code_id'] != code_id:
                print(f"   stage4: code_id mismatch ({config['code_id']} != {code_id})")
                return False

            # Verify config has necessary sections for both training and evaluation
            required_sections = ['training', 'evaluation', 'environment', 'dataset_creation']
            for section in required_sections:
                if section not in config:
                    print(f"   stage4: Missing required section '{section}'")
                    return False

            # Verify that both training and evaluation are enabled
            if not config.get('run_training', False):
                print(f"   stage4: run_training is not enabled - this may not be intentional")
            if not config.get('run_evaluation', False):
                print(f"   stage4: run_evaluation is not enabled - this may not be intentional")

            print(f"   stage4: Valid experiment config")

            return True

        except json.JSONDecodeError as e:
            print(f"   Invalid JSON: {e}")
            return False
        except yaml.YAMLError as e:
            print(f"   Invalid YAML: {e}")
            return False
        except Exception as e:
            print(f"   Validation error: {e}")
            return False


class StageExecutor:
    """Executes individual experiment stages as subprocesses."""

    def __init__(self, log_file_handle, dry_run: bool = False):
        """
        Initialize executor.

        Args:
            log_file_handle: Open file handle for logging
            dry_run: If True, validate but don't execute
        """
        self.log_file = log_file_handle
        self.dry_run = dry_run
        self.pythonpath = "."  # Required for all script execution

    def execute_stage1_main_dataset(
        self,
        code_id: str,
        config_path: Path,
        output_path: str,
        test_size: float = 0.3
    ) -> bool:
        """
        Execute Stage 1: Main curriculum dataset generation.

        Uses: src.data.create_dataset.create_ldpc_dataset()

        Args:
            code_id: LDPC code identifier
            config_path: Path to curriculum JSON config
            output_path: Path for output HDF5 file
            test_size: Train/test split ratio

        Returns:
            True if successful
        """
        stage_name = "Stage 1: Main Dataset Creation"
        self._log_stage_start(stage_name, code_id)

        if self.dry_run:
            self._log(f"[DRY RUN] Would create main dataset: {output_path}")
            return True

        # Import and call directly (more efficient than subprocess for this)
        try:
            from src.data.create_dataset import create_ldpc_dataset
            from src.utils.path_utils import resolve_project_path

            resolved_output = resolve_project_path(output_path)

            # Check if dataset already exists
            if resolved_output.exists():
                self._log(f"Dataset already exists: {resolved_output}")
                return True

            self._log(f"Creating main dataset...")
            self._log(f"   Config: {config_path}")
            self._log(f"   Output: {output_path}")
            self._log(f"   Test size: {test_size}")

            success = create_ldpc_dataset(
                config_path=str(config_path),
                output_path=output_path,
                test_size=test_size,
                code_id=code_id
            )

            if success:
                self._log_stage_success(stage_name)
                return True
            else:
                self._log_stage_failure(stage_name, "Dataset creation returned False")
                return False

        except Exception as e:
            self._log_stage_failure(stage_name, str(e))
            return False

    def execute_stage2_expert_dataset(
        self,
        code_id: str,
        config_path: Path,
        output_path: str,
        test_size: float = 0.3
    ) -> bool:
        """
        Execute Stage 2: Expert-specific dataset generation.

        Identical to Stage 1 but with smaller dataset for expert experiences.

        Args:
            code_id: LDPC code identifier
            config_path: Path to expert curriculum JSON config
            output_path: Path for output HDF5 file
            test_size: Train/test split ratio

        Returns:
            True if successful
        """
        stage_name = "Stage 2: Expert Dataset Creation"
        self._log_stage_start(stage_name, code_id)

        if self.dry_run:
            self._log(f"[DRY RUN] Would create expert dataset: {output_path}")
            return True

        # Reuse same function as Stage 1
        try:
            from src.data.create_dataset import create_ldpc_dataset
            from src.utils.path_utils import resolve_project_path

            resolved_output = resolve_project_path(output_path)

            if resolved_output.exists():
                self._log(f"Expert dataset already exists: {resolved_output}")
                return True

            self._log(f"Creating expert dataset...")
            self._log(f"   Config: {config_path}")
            self._log(f"   Output: {output_path}")

            success = create_ldpc_dataset(
                config_path=str(config_path),
                output_path=output_path,
                test_size=test_size,
                code_id=code_id
            )

            if success:
                self._log_stage_success(stage_name)
                return True
            else:
                self._log_stage_failure(stage_name, "Dataset creation returned False")
                return False

        except Exception as e:
            self._log_stage_failure(stage_name, str(e))
            return False

    def execute_stage3_expert_experience(
        self,
        code_id: str,
        config_path: Path,
        expert_override: Optional[str] = None
    ) -> bool:
        """
        Execute Stage 3: Expert experience generation.

        Uses: scripts/run_expert_experience.py

        Args:
            code_id: LDPC code identifier
            config_path: Path to expert experience YAML config
            expert_override: Optional expert type to override config

        Returns:
            True if successful
        """
        stage_name = "Stage 3: Expert Experience Generation"
        self._log_stage_start(stage_name, code_id)

        if self.dry_run:
            cmd_str = f"scripts/run_expert_experience.py --config {config_path}"
            if expert_override:
                cmd_str += f" --agent-type {expert_override}"
            self._log(f"[DRY RUN] Would run: {cmd_str}")
            return True

        # Build command
        cmd = [
            sys.executable,
            "-u",  # Unbuffered output for real-time logging
            "scripts/run_expert_experience.py",
            "--config", str(config_path)
        ]

        # Handle expert override by passing --agent-type parameter
        if expert_override:
            self._log(f"Expert override requested: {expert_override}")
            cmd.extend(["--agent-type", expert_override])

        self._log(f"Running command: {' '.join(cmd)}")

        return self._run_subprocess(cmd, stage_name)

    def execute_stage4_experiment(
        self,
        code_id: str,
        config_path: Path,
        output_dir_override: Optional[str] = None
    ) -> bool:
        """
        Execute Stage 4: Complete experiment (training + evaluation).

        Uses: scripts/run_experiment.py with the config as-is (no modifications).
        The config should have both run_training=true and run_evaluation=true.

        Args:
            code_id: LDPC code identifier
            config_path: Path to experiment YAML config
            output_dir_override: Optional output parent directory override

        Returns:
            True if successful
        """
        stage_name = "Stage 4: Complete Experiment (Training + Evaluation)"
        self._log_stage_start(stage_name, code_id)

        if self.dry_run:
            cmd_str = f"scripts/run_experiment.py --config {config_path}"
            if output_dir_override:
                cmd_str += f" --output-parent-dir {output_dir_override}"
            self._log(f"[DRY RUN] Would run: {cmd_str}")
            return True

        # Use the config file directly - no modifications needed!
        cmd = [
            sys.executable,
            "-u",  # Unbuffered output for real-time logging
            "scripts/run_experiment.py",
            "--config", str(config_path)
        ]

        # Handle output directory override by passing --output-parent-dir parameter
        if output_dir_override:
            self._log(f"Output directory override requested: {output_dir_override}")
            cmd.extend(["--output-parent-dir", output_dir_override])

        self._log(f"Running command: {' '.join(cmd)}")
        self._log(f"   (config used as-is with run_training=true, run_evaluation=true)")

        return self._run_subprocess(cmd, stage_name)

    def _run_subprocess(self, cmd: List[str], stage_name: str) -> bool:
        """
        Run subprocess with proper environment and logging.

        Args:
            cmd: Command to execute
            stage_name: Name of stage for error reporting

        Returns:
            True if subprocess succeeded
        """
        import os

        # Set PYTHONPATH environment
        env = os.environ.copy()
        env['PYTHONPATH'] = self.pythonpath

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                bufsize=1
            )

            # Stream output in real-time
            for line in process.stdout:
                line = line.rstrip()
                self._log(f"   {line}")

            process.wait()

            if process.returncode == 0:
                self._log_stage_success(stage_name)
                return True
            else:
                self._log_stage_failure(stage_name, f"Process exited with code {process.returncode}")
                return False

        except Exception as e:
            self._log_stage_failure(stage_name, str(e))
            return False

    def _log(self, message: str):
        """Write message to both console and log file."""
        print(message)
        self.log_file.write(message + '\n')
        self.log_file.flush()

    def _log_stage_start(self, stage_name: str, code_id: str):
        """Log stage start banner."""
        self._log("")
        self._log("=" * 80)
        self._log(f"{stage_name} - {code_id}")
        self._log("=" * 80)

    def _log_stage_success(self, stage_name: str):
        """Log stage success."""
        self._log(f"{stage_name} completed successfully")

    def _log_stage_failure(self, stage_name: str, error: str):
        """Log stage failure."""
        self._log(f"{stage_name} FAILED: {error}")


class MultiMatrixOrchestrator:
    """Main orchestrator for multi-matrix experiments."""

    def __init__(
        self,
        config_folder: str,
        log_dir: str = 'experiments',
        dry_run: bool = False,
        skip_stages: List[int] = None,
        expert_override: Optional[str] = None,
        output_dir_override: Optional[str] = None
    ):
        """
        Initialize orchestrator.

        Args:
            config_folder: Root config folder path
            log_dir: Directory for experiment logs
            dry_run: If True, validate without executing
            skip_stages: List of stage numbers to skip
            expert_override: Optional expert agent type override for Stage 3
            output_dir_override: Optional output parent directory override for Stage 4
        """
        self.config_folder = Path(config_folder)
        self.log_dir = Path(log_dir)
        self.dry_run = dry_run
        self.skip_stages = skip_stages or []
        self.expert_override = expert_override
        self.output_dir_override = output_dir_override

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f"experiment_log_{timestamp}.txt"
        self.log_path = self.log_dir / log_filename

        self.log_file = None
        self.executor = None
        self.discovery = ExperimentDiscovery(self.config_folder)

    def run_all_matrices(self) -> bool:
        """
        Discover and run experiments from all code_id format subfolders.

        Only runs experiments from subfolders named with code_id format (e.g., '64x128').

        Returns:
            True if all experiments succeeded
        """
        experiments = self.discovery.discover_all_matrices()

        if not experiments:
            print("No valid matrix experiments found")
            return False

        return self._run_experiments(experiments)

    def run_all_folders(self) -> bool:
        """
        Discover and run experiments from ALL subfolders (including ablation variants).

        Runs experiments from any subfolder with valid configs, regardless of naming.
        The code_id is extracted from experiment_config.yml.

        Returns:
            True if all experiments succeeded
        """
        experiments = self.discovery.discover_all_folders()

        if not experiments:
            print("No valid experiments found")
            return False

        return self._run_experiments(experiments)

    def run_specific_matrices(self, code_ids: List[str]) -> bool:
        """
        Run specific experiments by code_id (folder name = code_id).

        Args:
            code_ids: List of code_ids to run

        Returns:
            True if all experiments succeeded
        """
        try:
            experiments = self.discovery.discover_specific_matrices(code_ids)
            return self._run_experiments(experiments)
        except (FileNotFoundError, ValueError) as e:
            print(f"{e}")
            return False

    def run_specific_folders(self, folder_names: List[str]) -> bool:
        """
        Run specific experiments by folder name (may or may not be code_id format).

        For ablation studies where folder names like '64x128_NoC_NoE' contain
        experiment configs for code_id '64x128'.

        Args:
            folder_names: List of folder names to run

        Returns:
            True if all experiments succeeded
        """
        try:
            experiments = self.discovery.discover_specific_folders(folder_names)
            return self._run_experiments(experiments)
        except (FileNotFoundError, ValueError) as e:
            print(f"{e}")
            return False

    def _run_experiments(self, experiments: Dict[str, Dict]) -> bool:
        """
        Execute the 4-stage pipeline for all experiments.

        Args:
            experiments: Dict mapping folder_name to experiment configurations

        Returns:
            True if all experiments succeeded
        """
        # Open log file
        self.log_file = open(self.log_path, 'w')
        self.executor = StageExecutor(self.log_file, dry_run=self.dry_run)

        try:
            self._log_header()

            # Run each experiment
            for folder_name, exp_config in experiments.items():
                success = self._run_single_experiment(folder_name, exp_config)

                if not success:
                    self._log(f"\nExperiment failed for {folder_name}")
                    self._log(f"Halting pipeline to prevent corrupted results")
                    return False

            self._log_footer()
            return True

        finally:
            if self.log_file:
                self.log_file.close()

    def _run_single_experiment(self, folder_name: str, exp_config: Dict) -> bool:
        """
        Run complete 4-stage pipeline for a single experiment.

        Args:
            folder_name: Folder name (may be code_id or ablation variant name)
            exp_config: Experiment configuration dict containing:
                - folder_name: Same as parameter
                - code_id: Actual LDPC code identifier (e.g., '64x128')
                - config_dir: Path to configuration directory
                - configs: Dict of config file paths

        Returns:
            True if all stages succeeded
        """
        code_id = exp_config['code_id']
        is_variant = folder_name != code_id

        self._log("")
        self._log("#" * 80)
        if is_variant:
            self._log(f"# STARTING EXPERIMENT: {folder_name} (code: {code_id})")
        else:
            self._log(f"# STARTING EXPERIMENT: {folder_name}")
        self._log("#" * 80)

        config_paths = exp_config['configs']

        # Validate configurations
        self._log(f"\nValidating configurations for {folder_name}...")
        if not self.discovery.validate_configs(code_id, config_paths):
            self._log(f"Configuration validation failed for {folder_name}")
            return False
        self._log(f"All configurations valid for {folder_name}")

        # Define output paths (following MASTER_EXPERIMENT_DESIGN_REVISED.md)
        main_dataset = f"data/ldpc_{code_id}_main_dataset.h5"
        expert_dataset = f"data/ldpc_{code_id}_expert_dataset.h5"

        # Stage 1: Main dataset
        if 1 not in self.skip_stages:
            success = self.executor.execute_stage1_main_dataset(
                code_id=code_id,
                config_path=config_paths['stage1'],
                output_path=main_dataset,
                test_size=0.3
            )
            if not success:
                return False
        else:
            self._log(f"Skipping Stage 1 (--skip-stages)")

        # Stage 2: Expert dataset
        if 2 not in self.skip_stages:
            success = self.executor.execute_stage2_expert_dataset(
                code_id=code_id,
                config_path=config_paths['stage2'],
                output_path=expert_dataset,
                test_size=0.3
            )
            if not success:
                return False
            # Force garbage collection to close HDF5 file handles
            self._log("\nForcing garbage collection to release file handles...")
            gc.collect()
            # Allow filesystem to flush HDF5 file locks before Stage 3
            self._log("Waiting 2 seconds for filesystem to release file locks...")
            time.sleep(2)
        else:
            self._log(f"Skipping Stage 2 (--skip-stages)")

        # Stage 3: Expert experience
        if 3 not in self.skip_stages:
            success = self.executor.execute_stage3_expert_experience(
                code_id=code_id,
                config_path=config_paths['stage3'],
                expert_override=self.expert_override
            )
            if not success:
                return False
        else:
            self._log(f"Skipping Stage 3 (--skip-stages)")

        # Stage 4: Complete Experiment (Training + Evaluation)
        if 4 not in self.skip_stages:
            success = self.executor.execute_stage4_experiment(
                code_id=code_id,
                config_path=config_paths['stage4'],
                output_dir_override=self.output_dir_override
            )
            if not success:
                return False
        else:
            self._log(f"Skipping Stage 4 (--skip-stages)")

        self._log(f"\nAll stages completed successfully for {folder_name}")
        return True

    def _log(self, message: str):
        """Write to both console and log file."""
        print(message)
        if self.log_file:
            self.log_file.write(message + '\n')
            self.log_file.flush()

    def _log_header(self):
        """Write experiment session header."""
        self._log("=" * 80)
        self._log("Multi-Matrix LDPC RL Experiment Orchestrator")
        self._log("=" * 80)
        self._log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log(f"Config folder: {self.config_folder}")
        self._log(f"Log file: {self.log_path}")
        self._log(f"Dry run: {self.dry_run}")
        self._log(f"Skip stages: {self.skip_stages if self.skip_stages else 'None'}")
        self._log("=" * 80)

    def _log_footer(self):
        """Write experiment session footer."""
        self._log("")
        self._log("=" * 80)
        self._log("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
        self._log("=" * 80)
        self._log(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log(f"Log saved to: {self.log_path}")
        self._log("=" * 80)


def setup_argument_parser():
    """Configure command-line argument parser."""
    parser = argparse.ArgumentParser(
        description='Run multi-matrix LDPC RL experiments with 4-stage pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all matrix experiments (folders named with code_id format like 64x128)
  PYTHONPATH=. python scripts/run_multi_matrix_experiments.py \\
      --config-folder config/experiments_config/ --all-matrices

  # Run all experiments including ablation variants (any folder with valid configs)
  PYTHONPATH=. python scripts/run_multi_matrix_experiments.py \\
      --config-folder config/experiments_config/ --all-folders

  # Run specific matrix experiments
  PYTHONPATH=. python scripts/run_multi_matrix_experiments.py \\
      --config-folder config/experiments_config/ --matrices 24x48 64x128

  # Run specific ablation variant folders
  PYTHONPATH=. python scripts/run_multi_matrix_experiments.py \\
      --config-folder config/experiments_config/ --folders 64x128_NoC_NoE 64x128_Full

  # Dry run (validation only)
  PYTHONPATH=. python scripts/run_multi_matrix_experiments.py \\
      --config-folder config/experiments_config/ --all-folders --dry-run

  # Override expert agent
  PYTHONPATH=. python scripts/run_multi_matrix_experiments.py \\
      --config-folder config/experiments_config/ --all-matrices --expert omniscient

  # Custom output directory
  PYTHONPATH=. python scripts/run_multi_matrix_experiments.py \\
      --config-folder config/experiments_config/ --all-folders --output-dir experiments_v2
        """
    )

    parser.add_argument(
        '--config-folder',
        type=str,
        required=True,
        help='Root folder containing experiment subfolders with config files'
    )

    # Experiment selection (mutually exclusive)
    selection_group = parser.add_mutually_exclusive_group(required=True)
    selection_group.add_argument(
        '--all-matrices',
        action='store_true',
        help='Run experiments from all code_id format subfolders (e.g., 64x128, 128x256)'
    )
    selection_group.add_argument(
        '--all-folders',
        action='store_true',
        help='Run experiments from ALL subfolders with valid configs (including ablation variants)'
    )
    selection_group.add_argument(
        '--matrices',
        nargs='+',
        type=str,
        metavar='CODE_ID',
        help='Specific code_id folders to process (e.g., 24x48 64x128)'
    )
    selection_group.add_argument(
        '--folders',
        nargs='+',
        type=str,
        metavar='FOLDER',
        help='Specific folder names to process (e.g., 64x128_NoC_NoE 64x128_Full)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configurations without executing experiments'
    )

    parser.add_argument(
        '--expert',
        type=str,
        default=None,
        help='Override expert agent type for Stage 3 (e.g., omniscient, min_sum_expert)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Custom output directory for results and models (overrides config)'
    )

    parser.add_argument(
        '--skip-stages',
        nargs='+',
        type=int,
        choices=[1, 2, 3, 4],
        default=[],
        help='Skip specific stages (e.g., --skip-stages 1 2 to skip dataset creation)'
    )

    parser.add_argument(
        '--log-dir',
        type=str,
        default='experiments',
        help='Directory for experiment logs (default: experiments/)'
    )

    return parser


def main():
    """Main entry point for orchestrator script."""
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Initialize orchestrator
    orchestrator = MultiMatrixOrchestrator(
        config_folder=args.config_folder,
        log_dir=args.log_dir,
        dry_run=args.dry_run,
        skip_stages=args.skip_stages,
        expert_override=args.expert,
        output_dir_override=args.output_dir
    )

    # Run experiments based on selection mode
    try:
        if args.all_matrices:
            success = orchestrator.run_all_matrices()
        elif args.all_folders:
            success = orchestrator.run_all_folders()
        elif args.matrices:
            success = orchestrator.run_specific_matrices(args.matrices)
        elif args.folders:
            success = orchestrator.run_specific_folders(args.folders)
        else:
            # Should never reach here due to mutually exclusive group
            print("No selection mode specified")
            sys.exit(1)

        if success:
            print("\nAll experiments completed successfully")
            sys.exit(0)
        else:
            print("\nExperiment pipeline failed")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
