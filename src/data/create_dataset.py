"""
Dataset Creation Script for LDPC RL Training

This script creates a robust dataset using RobustDataGenerator for training
the LDPC RL decoder. This replaces the dynamic data generation approach
with a static, reproducible dataset.
"""

import copy
import json
from pathlib import Path
import os
import time
from typing import Optional
from src.core.ldpc_code import LDPC_Code
from src.data.robust_data_generator import RobustDataGenerator, rescale_phase_samples
from src.utils.path_utils import resolve_config_path, resolve_data_path

PHASE_ARRAYS = ('original_codewords', 'noisy_codewords', 'error_patterns', 'metadata')

# The subset of those that is one row per sample by n bits. `metadata` is a
# structured one-dimensional array and has no width to check.
BIT_ARRAYS = ('original_codewords', 'noisy_codewords', 'error_patterns')


def expected_test_samples(phase_sizes, test_size: float) -> int:
    """Test-split size a stratified split produces, mirroring create_split()."""
    return sum(size - int(size * (1 - test_size)) for size in phase_sizes)




def _partition_fault(train, test, combined, total) -> str:
    """Say which way a split failed to be a partition, most serious first."""
    import numpy as np

    def plural(count, one, many):
        return f"{count:,} {one if count == 1 else many}"

    overlap = np.intersect1d(train, test)
    if len(overlap):
        return (f"{plural(len(overlap), 'sample is', 'samples are')} in both "
                f"halves, starting at index {overlap[0]}")

    outside = combined[(combined < 0) | (combined >= total)]
    if len(outside):
        return (f"{plural(len(outside), 'index falls', 'indices fall')} outside "
                f"[0, {total:,}), such as {outside[0]}")

    values, counts = np.unique(combined, return_counts=True)
    repeated = values[counts > 1]
    if len(repeated):
        return (f"{plural(len(repeated), 'sample appears', 'samples appear')} "
                f"more than once, starting at index {repeated[0]}")

    return (f"{len(train):,} train + {len(test):,} test indices cover "
            f"{len(np.unique(combined)):,} of {total:,} samples")


def _validate_split(splits, phase_sizes, test_size: Optional[float]):
    """
    Judge a train/test split by its index arrays rather than by its metadata.

    Training and evaluation consume `train_indices` and `test_indices`. The count
    attributes beside them are written from those arrays but are not the arrays,
    so a split whose indices are missing, overlapping or truncated can carry
    attributes that still look right. Every check here reads the arrays.

    Args:
        splits: The `splits` group of an open dataset.
        phase_sizes: Samples per phase, in order.
        test_size: Requested split, or None to check only structure.

    Returns:
        (ok, reason) where reason describes what failed.
    """
    import numpy as np

    for name in ('train_indices', 'test_indices'):
        if name not in splits:
            return False, (f"split has no {name}; evaluation reads that array, "
                           f"not the count beside it")
        if not np.issubdtype(splits[name].dtype, np.integer):
            return False, f"split/{name} holds {splits[name].dtype}, not indices"

    train = splits['train_indices'][:]
    test = splits['test_indices'][:]
    total = sum(phase_sizes)

    # One comparison covers bounds, duplicates, overlap between the halves, and
    # samples belonging to neither: a correct split is a permutation of every
    # sample in the dataset.
    combined = np.concatenate([train, test])
    if len(combined) != total or not np.array_equal(np.sort(combined), np.arange(total)):
        return False, (f"split is not a partition of the dataset "
                       f"({_partition_fault(train, test, combined, total)}); "
                       f"training and evaluation would not be independent")

    if test_size is None:
        return True, "split is a partition"

    expected = expected_test_samples(phase_sizes, test_size)
    if len(test) != expected:
        return False, (f"split holds {len(test):,} test samples, "
                       f"test_size={test_size} asks for {expected:,}")

    # Stratified: each phase contributes its own share, so a split with the right
    # total can still be drawn from the wrong phases.
    boundaries = np.concatenate([[0], np.cumsum(phase_sizes)])
    per_phase = np.diff(np.searchsorted(np.sort(test), boundaries))
    for index, (size, actual) in enumerate(zip(phase_sizes, per_phase)):
        want = size - int(size * (1 - test_size))
        if actual != want:
            return False, (f"phase_{index} contributes {actual:,} test samples, "
                           f"a stratified test_size={test_size} split asks for "
                           f"{want:,}")

    return True, "split is a stratified partition"


def validate_existing_dataset(dataset_path, config_path, test_size=None, code=None):
    """
    Decide whether an existing file is the dataset a request would have produced.

    Existence is not evidence. Experiment variants share dataset paths while asking
    for different contents, a run can be interrupted after the file is created but
    before it is filled, and a split written for one test_size persists into a run
    that asked for another. Any of those silently trains on data the configuration
    did not request, which is why both entry points call this before reusing a file.

    Checks, in order of what they catch:

    - the file opens as HDF5 and carries the config it was generated from
    - that config equals the requested one after the same rescaling the generator
      applies, so a changed seed, error model or sample count is caught
    - every phase group holds arrays of its declared length, so a partially written
      dataset is not mistaken for a complete one
    - every phase holds them at one width, matching the code in use
    - the split's own index arrays partition the dataset, and do so in the
      per-phase proportions a stratified split at this test_size produces

    The split checks read `train_indices` and `test_indices` rather than the count
    attributes beside them, because those arrays are what training and evaluation
    consume. A split can carry a correct count and indices that overlap, run short
    or be absent.

    Args:
        dataset_path: Existing HDF5 file to judge
        config_path: Curriculum config the caller is asking for
        test_size: Requested split, or None to skip the split check
        code: LDPC_Code in use, or None to skip the width check

    Returns:
        (matches, reason) where reason describes what matched or what did not
    """
    import h5py

    try:
        with open(config_path, 'r') as f:
            requested = json.load(f)
    except Exception as e:
        return False, f"could not read the requested config {config_path}: {e}"

    if 'curriculum' not in requested:
        return False, f"{config_path} has no curriculum"

    requested = copy.deepcopy(requested)
    if requested.get('total_samples') is not None:
        rescale_phase_samples(requested)
    phase_sizes = [phase.get('num_samples', 0) for phase in requested['curriculum']]

    try:
        with h5py.File(dataset_path, 'r') as f:
            stored_raw = f.attrs.get('source_config')
            if stored_raw is None:
                return False, "no source_config attribute; not a dataset this project wrote"
            stored = json.loads(stored_raw)

            if stored != requested:
                differing = sorted(
                    key for key in set(stored) | set(requested)
                    if stored.get(key) != requested.get(key))
                return False, (f"was generated from a different configuration "
                               f"(differs in: {', '.join(differing) or 'phase details'})")

            for index, size in enumerate(phase_sizes):
                name = f'phase_{index}'
                if name not in f:
                    return False, f"missing {name}"
                group = f[name]
                for array in PHASE_ARRAYS:
                    if array not in group:
                        return False, f"{name} has no {array}"
                    if len(group[array]) != size:
                        return False, (f"{name}/{array} holds {len(group[array])} rows, "
                                       f"expected {size}; the dataset is incomplete")

            # Widths, every phase rather than only the first. A phase written at
            # the wrong width still holds the right number of rows, so the length
            # check above passes and the iterator hands the agent observations of
            # a shape the network cannot take.
            for array in BIT_ARRAYS:
                widths = {index: f[f'phase_{index}/{array}'].shape[1]
                          for index in range(len(phase_sizes))}
                distinct = set(widths.values())
                if len(distinct) > 1:
                    return False, (f"{array} is not one width across phases: "
                                   f"{', '.join(f'phase_{i} {w}' for i, w in widths.items())}")
                width = distinct.pop()
                if code is not None and width != code.n:
                    return False, (f"{array} are {width} bits, code expects {code.n}")

            if 'splits' in f:
                ok, reason = _validate_split(f['splits'], phase_sizes, test_size)
                if not ok:
                    return False, reason
            elif test_size is not None:
                return False, "has no train/test split"

    except OSError as e:
        return False, f"not a readable HDF5 dataset ({e})"
    except Exception as e:
        return False, f"could not be validated ({e})"

    return True, (f"{sum(phase_sizes):,} samples across {len(phase_sizes)} phases, "
                  f"matching {config_path}")


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
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
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
