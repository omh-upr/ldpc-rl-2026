#!/usr/bin/env python3
"""
Validate train/test split stratification for LDPC datasets.

This script checks:
1. Phase stratification - each phase contributes expected proportion to test set
2. Error distribution - error counts are balanced between train/test
3. Pattern balance - identifies significant imbalances

Usage:
    PYTHONPATH=. python scripts/run_validate_split.py data/ldpc_64x128_dataset.h5
    PYTHONPATH=. python scripts/run_validate_split.py data/ldpc_64x128_dataset.h5 --test-size 0.2
    PYTHONPATH=. python scripts/run_validate_split.py data/ldpc_64x128_dataset.h5 --verbose
"""

import sys
import argparse
import h5py
import numpy as np
import pandas as pd
from pathlib import Path


def analyze_split(h5_file_path, test_size=0.2, verbose=True):
    """
    Comprehensive split validation.

    Args:
        h5_file_path: Path to HDF5 dataset file
        test_size: Expected test proportion (e.g., 0.2 for 20%)
        verbose: Print detailed information

    Returns:
        dict with validation results:
        - phase_stratification: bool
        - error_distribution: bool
        - warnings: list of warning messages
        - details: dict with detailed data
    """
    results = {
        'phase_stratification': True,
        'error_distribution': True,
        'warnings': [],
        'details': {}
    }

    with h5py.File(h5_file_path, 'r') as f:
        # Check if splits exist
        if 'splits' not in f:
            raise ValueError(
                "Dataset does not contain splits. "
                "Run RobustDataGenerator.create_split() first."
            )

        train_indices = f['splits/train_indices'][:]
        test_indices = f['splits/test_indices'][:]

        total_train = len(train_indices)
        total_test = len(test_indices)
        total_samples = total_train + total_test
        actual_test_pct = total_test / total_samples

        if verbose:
            print(f"Dataset: {Path(h5_file_path).name}")
            print(f"   Total samples: {total_samples:,}")
            print(f"   Train: {total_train:,} ({100*(1-actual_test_pct):.1f}%)")
            print(f"   Test: {total_test:,} ({100*actual_test_pct:.1f}%)")
            print(f"   Expected test: {100*test_size:.1f}%")
            print()

        # Check overall split ratio
        overall_deviation = abs(actual_test_pct - test_size)
        if overall_deviation > 0.01:  # >1% deviation
            results['warnings'].append(
                f"Overall test split {actual_test_pct:.1%} deviates from expected {test_size:.1%}"
            )

        # Analyze each phase
        phase_data = []
        error_data = []
        global_offset = 0

        num_phases = len([k for k in f.keys() if k.startswith('phase_')])

        for phase_idx in range(num_phases):
            phase_group = f[f'phase_{phase_idx}']
            num_samples = phase_group.attrs['num_samples']
            metadata = phase_group['metadata'][:]
            phase_name = phase_group.attrs.get('phase_name', f'Phase_{phase_idx+1}')

            # Determine which samples are in train/test for this phase
            phase_indices = np.arange(global_offset, global_offset + num_samples)
            train_mask = np.isin(phase_indices, train_indices)
            test_mask = np.isin(phase_indices, test_indices)

            phase_train = np.sum(train_mask)
            phase_test = np.sum(test_mask)
            phase_test_pct = phase_test / num_samples

            phase_data.append({
                'phase': phase_idx + 1,
                'name': phase_name,
                'total': num_samples,
                'train': phase_train,
                'test': phase_test,
                'test_pct': phase_test_pct,
                'deviation': phase_test_pct - test_size
            })

            # Check phase stratification
            phase_deviation = abs(phase_test_pct - test_size)
            if phase_deviation > 0.02:  # >2% deviation
                results['phase_stratification'] = False
                results['warnings'].append(
                    f"{phase_name}: test split {phase_test_pct:.1%} deviates "
                    f"from expected {test_size:.1%} by {phase_deviation:.1%}"
                )

            # Analyze error distribution within phase
            num_errors = metadata['num_errors']

            # Create local masks for this phase's train/test samples
            local_train_mask = train_mask[global_offset:global_offset+num_samples] - global_offset
            local_test_mask = test_mask[global_offset:global_offset+num_samples] - global_offset

            # Correct approach: use masks relative to phase data
            phase_train_errors = num_errors[np.arange(num_samples)[np.isin(phase_indices, train_indices)] - global_offset]
            phase_test_errors = num_errors[np.arange(num_samples)[np.isin(phase_indices, test_indices)] - global_offset]

            for error_count in np.unique(num_errors):
                error_mask = (num_errors == error_count)
                error_total = np.sum(error_mask)
                error_train = np.sum(phase_train_errors == error_count)
                error_test = np.sum(phase_test_errors == error_count)

                if error_total > 0:
                    error_test_pct = error_test / error_total
                    error_deviation = error_test_pct - test_size

                    error_data.append({
                        'phase': phase_idx + 1,
                        'phase_name': phase_name,
                        'num_errors': error_count,
                        'total': error_total,
                        'train': error_train,
                        'test': error_test,
                        'test_pct': error_test_pct,
                        'deviation': error_deviation
                    })

                    # Flag significant deviations for patterns with reasonable sample size
                    if error_total >= 50 and abs(error_deviation) > 0.15:  # >15% deviation
                        results['error_distribution'] = False
                        results['warnings'].append(
                            f"{phase_name}, {error_count}-error: "
                            f"test split {error_test_pct:.1%} (expected {test_size:.1%}), "
                            f"deviation {error_deviation:+.1%}, n={error_total}"
                        )

            global_offset += num_samples

        results['details']['phases'] = pd.DataFrame(phase_data)
        results['details']['errors'] = pd.DataFrame(error_data)

        if verbose:
            print("=" * 70)
            print("PHASE STRATIFICATION")
            print("=" * 70)
            phase_df = results['details']['phases']
            print(phase_df[['phase', 'name', 'total', 'train', 'test', 'test_pct', 'deviation']].to_string(index=False))
            print()

            # Overall error distribution
            error_summary = results['details']['errors'].groupby('num_errors').agg({
                'total': 'sum',
                'train': 'sum',
                'test': 'sum'
            })
            error_summary['test_pct'] = error_summary['test'] / error_summary['total']
            error_summary['deviation'] = error_summary['test_pct'] - test_size

            print("=" * 70)
            print("ERROR COUNT DISTRIBUTION (Overall)")
            print("=" * 70)
            print(error_summary.to_string())
            print()

            # Identify worst imbalances
            worst_imbalances = results['details']['errors'].copy()
            worst_imbalances['abs_deviation'] = worst_imbalances['deviation'].abs()
            worst_imbalances = worst_imbalances[worst_imbalances['total'] >= 50]  # Filter small samples
            worst_imbalances = worst_imbalances.nlargest(10, 'abs_deviation')

            if len(worst_imbalances) > 0:
                print("=" * 70)
                print("LARGEST DEVIATIONS (min 50 samples)")
                print("=" * 70)
                print(worst_imbalances[['phase_name', 'num_errors', 'total', 'test', 'test_pct', 'deviation']].to_string(index=False))
                print()

            print("=" * 70)
            print("VALIDATION SUMMARY")
            print("=" * 70)

            if results['phase_stratification']:
                print("Phase stratification: PASSED")
            else:
                print("Phase stratification: FAILED")

            if results['error_distribution']:
                print("Error distribution balance: PASSED")
            else:
                print("Error distribution balance: WARNINGS")

            if not results['warnings']:
                print("\nAll checks passed! Split is well-balanced.")
            else:
                print(f"\n{len(results['warnings'])} warning(s) detected:")
                for i, warning in enumerate(results['warnings'][:10], 1):  # Show first 10
                    print(f"   {i}. {warning}")
                if len(results['warnings']) > 10:
                    print(f"   ... and {len(results['warnings']) - 10} more")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate train/test split stratification for LDPC datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic validation:
    PYTHONPATH=. python scripts/run_validate_split.py data/ldpc_64x128_dataset.h5

  With custom test size:
    PYTHONPATH=. python scripts/run_validate_split.py data/ldpc_64x128_dataset.h5 --test-size 0.25

  Quiet mode (only show warnings):
    PYTHONPATH=. python scripts/run_validate_split.py data/ldpc_64x128_dataset.h5 --quiet

Exit codes:
  0 - All checks passed
  1 - Warnings detected or validation failed
  2 - Error (file not found, no splits, etc.)
        """
    )

    parser.add_argument(
        'dataset',
        type=str,
        help="Path to HDF5 dataset file (e.g., data/ldpc_64x128_dataset.h5)"
    )

    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help="Expected test proportion (default: 0.2 = 20%%)"
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help="Quiet mode - only show warnings and summary"
    )

    args = parser.parse_args()

    # Check file exists
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset file not found: {args.dataset}")
        sys.exit(2)

    try:
        results = analyze_split(
            str(dataset_path),
            test_size=args.test_size,
            verbose=not args.quiet
        )

        # Exit code based on validation results
        if results['phase_stratification'] and results['error_distribution']:
            sys.exit(0)  # All good
        else:
            sys.exit(1)  # Warnings detected

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(2)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == '__main__':
    main()
