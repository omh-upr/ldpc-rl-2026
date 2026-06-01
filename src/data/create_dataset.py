"""
Dataset Creation Script for LDPC RL Training

This script creates a robust dataset using RobustDataGenerator for training
the LDPC RL decoder. This replaces the dynamic data generation approach
with a static, reproducible dataset.
"""

import json
import os
import time
from typing import Optional
from src.core.ldpc_code import LDPC_Code
from src.data.robust_data_generator import RobustDataGenerator
from src.utils.path_utils import resolve_config_path, resolve_data_path

def create_ldpc_dataset(config_path: str, output_path: str, test_size: float = 0.3, code_id: str = '56x80'):
    """
    Create LDPC dataset using RobustDataGenerator.

    Args:
        config_path: Path to curriculum configuration JSON
        output_path: Path to save the generated dataset
        test_size: Fraction of data to use for testing
        code_id: LDPC code identifier (e.g., '24x48', '56x80', etc.)
    """
    start_time = time.time()
    print(f"Starting LDPC dataset creation at {time.strftime('%H:%M:%S')}")
    print(f"Config: {config_path}")
    print(f"Output: {output_path}")

    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded curriculum config from {config_path}")
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        return False
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in config file: {e}")
        return False

    # Create LDPC code and generator
    ldpc_code = LDPC_Code.from_default(code_id=code_id)
    generator = RobustDataGenerator(config, code=ldpc_code)

    # Generate and save dataset
    try:
        generator.generate_and_save_dataset(output_path)
        generation_time = time.time() - start_time
        print(f"Dataset generation completed in {generation_time:.1f}s")

        # Create train/test split. Pass the same LDPC_Code instance so
        # RobustDataGenerator does not fall back to the default 56x80 code.
        print(f"Creating train/test split (test_size={test_size})...")
        split_start = time.time()
        generator = RobustDataGenerator.from_h5(output_path, code=ldpc_code)
        generator.create_split(test_size=test_size, stratify=True, shuffle=True, seed=42)
        split_time = time.time() - split_start
        print(f"Split created in {split_time:.1f}s")

        # Print dataset summary
        print_dataset_summary(output_path, code=ldpc_code)

        total_time = time.time() - start_time
        print(f"\nDataset creation complete in {total_time:.1f}s")
        print(f"Dataset saved to: {output_path}")

        return True

    except Exception as e:
        print(f"Dataset creation failed: {e}")
        return False

def print_dataset_summary(dataset_path: str, code: Optional[LDPC_Code] = None):
    """Print summary of created dataset.

    Pass `code` to avoid RobustDataGenerator falling back to the default 56x80
    LDPC_Code (the summary itself reads only HDF5 attributes that do not
    depend on code dimensions, but skipping the fallback avoids misleading
    log output about matrix shapes).
    """
    try:
        generator = RobustDataGenerator.from_h5(dataset_path, code=code)

        print(f"\nDataset Summary:")
        print(f"   Total samples: {generator.h5_file.attrs['total_samples']:,}")
        print(f"   Dataset name: {generator.h5_file.attrs['dataset_name']}")
        print(f"   Seed: {generator.h5_file.attrs['seed']}")

        # Print phase breakdown
        print(f"\nPhase Breakdown:")
        for i in range(len(generator.config['curriculum'])):
            phase_group = generator.h5_file[f'phase_{i}']
            phase_name = phase_group.attrs['phase_name']
            num_samples = phase_group.attrs['num_samples']
            print(f"   {phase_name}: {num_samples:,} samples")

        # Print split information if available
        if 'splits' in generator.h5_file:
            train_size = len(generator.h5_file['splits/train_indices'])
            test_size = len(generator.h5_file['splits/test_indices'])
            print(f"\nData Split:")
            print(f"   Training: {train_size:,} samples ({train_size/(train_size+test_size):.1%})")
            print(f"   Testing: {test_size:,} samples ({test_size/(train_size+test_size):.1%})")

    except Exception as e:
        print(f"Could not print dataset summary: {e}")

def main():
    """Main execution function."""
    # Use the robust curriculum config from the config/ folder
    try:
        config_path = resolve_config_path("robust_curriculum_config.json")
        output_path = resolve_data_path("ldpc_robust_dataset.h5")

        # Alternative: use short curriculum config
        # config_path = resolve_config_path("short_curriculum_config.json")
        # output_path = resolve_data_path("ldpc_short_dataset.h5")

        print(f"Using config: {config_path}")
        print(f"Output path: {output_path}")

    except FileNotFoundError as e:
        print(f"Configuration file not found: {e}")
        print("Please ensure the robust curriculum config exists in the config/ folder.")
        return

    # Create dataset
    success = create_ldpc_dataset(str(config_path), str(output_path), test_size=0.3)

    if success:
        print(f"\nDataset creation successful!")
        print(f"Ready for training with RobustDataGenerator")
        print(f"Next steps:")
        print(f"   1. Verify your experiment config points to this dataset:")
        print(f"      dataset_creation.output_path: '{output_path}'")
        print(f"   2. Run the experiment pipeline:")
        print(f"      PYTHONPATH=. python scripts/run_experiment.py --config config/experiments_config/<folder>/experiment_config.yml")
    else:
        print(f"\nDataset creation failed!")
        print(f"   Check error messages above for details")

if __name__ == "__main__":
    main()
