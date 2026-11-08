#!/usr/bin/env python
"""
Q-Value Error Recognition Analysis

Evaluates how well a trained DQN agent recognizes error positions from the
initial observation by analyzing Q-value rankings. This measures whether the
network can identify errors in the noisy codeword without running full episodes.

The --top-k argument controls the analysis threshold:
  - If not specified: uses (num_errors + 1) as threshold for each test case
  - If specified: uses a fixed threshold for all test cases

Usage:
    # Adaptive threshold (top-(k+1) for k errors)
    PYTHONPATH=. python scripts/run_qvalue_analysis.py \
        --config config/experiments_config/64x128/experiment_config.yml \
        --model models/64x128_model.pth \
        --samples-per-k 1000 \
        --max-errors 6

    # Fixed threshold (e.g., always check top-10)
    PYTHONPATH=. python scripts/run_qvalue_analysis.py \
        --config config/experiments_config/64x128/experiment_config.yml \
        --model models/64x128_model.pth \
        --top-k 10

    # With JSON output
    PYTHONPATH=. python scripts/run_qvalue_analysis.py \
        --config config/experiments_config/64x128/experiment_config.yml \
        --model models/64x128_model.pth \
        --output results/qvalue_analysis.json
"""

import argparse
import json
import numpy as np
import torch
import yaml
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from src.core.ldpc_code import LDPC_Code
from src.core.environment import LDPCEnvironment
from src.core.learning_agent import DQNAgent


@dataclass
class QValueMetrics:
    """Metrics for a single test case."""
    num_errors: int              # Number of actual errors in test case
    top_k: int                   # Top-k threshold used for analysis
    recall: float                # Fraction of errors found in top-k
    precision: float             # Fraction of top-k that are errors
    exact_match: bool            # All errors found in top-k
    mrr: float                   # Mean reciprocal rank of error positions
    mean_rank: float             # Mean rank of error positions
    max_rank: int                # Worst (highest) rank among errors
    stop_rank: int               # Rank of STOP action (1 = highest Q-value)
    errors_in_top_k: int         # Count of errors in top-k
    stop_in_top_k: bool          # Whether STOP is in top-k


@dataclass
class AggregatedMetrics:
    """Aggregated metrics for a specific error count."""
    num_errors: int
    top_k: int
    samples: int
    mean_recall: float
    std_recall: float
    mean_precision: float
    std_precision: float
    exact_match_rate: float
    mean_mrr: float
    std_mrr: float
    mean_error_rank: float
    std_error_rank: float
    mean_max_rank: float
    mean_stop_rank: float
    std_stop_rank: float
    mean_errors_in_top_k: float
    stop_in_top_k_rate: float


def compute_metrics(q_values: np.ndarray,
                    error_positions: np.ndarray,
                    n: int,
                    top_k: Optional[int] = None) -> QValueMetrics:
    """
    Compute Q-value analysis metrics for a single test case.

    Args:
        q_values: Q-values for all actions, shape (action_dim,)
        error_positions: Indices of actual bit errors
        n: Codeword length (STOP action is at index n)
        top_k: Fixed top-k threshold. If None, uses (num_errors + 1)

    Returns:
        QValueMetrics with all computed metrics
    """
    num_errors = len(error_positions)

    # Determine threshold: fixed top_k or adaptive (num_errors + 1)
    threshold = top_k if top_k is not None else (num_errors + 1)

    # Get rankings (highest Q-value = rank 1)
    sorted_indices = np.argsort(q_values)[::-1]  # Descending order
    rank_map = {int(idx): rank + 1 for rank, idx in enumerate(sorted_indices)}

    # Top-k indices based on threshold
    top_indices = set(sorted_indices[:threshold].tolist())
    error_set = set(error_positions.tolist())

    # Compute overlap
    overlap = error_set & top_indices
    errors_in_top_k = len(overlap)

    # Metrics
    recall = errors_in_top_k / num_errors if num_errors > 0 else 0.0
    precision = errors_in_top_k / threshold if threshold > 0 else 0.0
    exact_match = error_set <= top_indices  # All errors found in top-k

    # Rank statistics for error positions
    if num_errors > 0:
        error_ranks = [rank_map[int(e)] for e in error_positions]
        mrr = float(np.mean([1.0 / r for r in error_ranks]))
        mean_rank = float(np.mean(error_ranks))
        max_rank = int(np.max(error_ranks))
    else:
        mrr = 0.0
        mean_rank = 0.0
        max_rank = 0

    # STOP action analysis (action index = n)
    stop_rank = rank_map[n]
    stop_in_top_k = n in top_indices

    return QValueMetrics(
        num_errors=num_errors,
        top_k=threshold,
        recall=recall,
        precision=precision,
        exact_match=exact_match,
        mrr=mrr,
        mean_rank=mean_rank,
        max_rank=max_rank,
        stop_rank=stop_rank,
        errors_in_top_k=errors_in_top_k,
        stop_in_top_k=stop_in_top_k
    )


def generate_test_case(code: LDPC_Code,
                       num_errors: int,
                       rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a single test case with known error positions.

    Args:
        code: LDPC code instance
        num_errors: Number of bit errors to introduce
        rng: Random number generator

    Returns:
        Tuple of (original_codeword, noisy_codeword, error_positions)
    """
    # Generate random message and encode
    message = rng.integers(0, 2, size=code.k)
    original_codeword = code.encode(message)

    # Create error vector with exactly num_errors bit flips
    error_positions = rng.choice(code.n, size=num_errors, replace=False)
    error_positions = np.sort(error_positions)  # Sort for consistency

    error_vector = np.zeros(code.n, dtype=int)
    error_vector[error_positions] = 1

    # Apply errors (XOR)
    noisy_codeword = (original_codeword + error_vector) % 2

    return original_codeword, noisy_codeword, error_positions


def load_agent(checkpoint_path: str,
               state_dim: int,
               action_dim: int,
               network_config: dict,
               device: str = 'cpu') -> DQNAgent:
    """
    Load trained DQN agent from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint file
        state_dim: State dimension for the agent
        action_dim: Action dimension for the agent
        network_config: Network architecture configuration
        device: Device to load the model on

    Returns:
        Loaded DQNAgent in evaluation mode
    """
    # Create agent with matching architecture
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        network_config=network_config,
        device=device
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if 'network_state_dict' in checkpoint:
        state_dict = checkpoint['network_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        # Assume checkpoint is the state dict itself
        state_dict = checkpoint

    agent.network.load_state_dict(state_dict)
    agent.network.eval()

    # Verify dimensions match
    if 'state_dim' in checkpoint and checkpoint['state_dim'] != state_dim:
        print(f"Warning: Checkpoint state_dim ({checkpoint['state_dim']}) differs from code ({state_dim})")
    if 'action_dim' in checkpoint and checkpoint['action_dim'] != action_dim:
        print(f"Warning: Checkpoint action_dim ({checkpoint['action_dim']}) differs from code ({action_dim})")

    return agent


def aggregate_results(results: Dict[int, List[QValueMetrics]]) -> Dict[int, AggregatedMetrics]:
    """
    Aggregate metrics across all test cases for each error count.

    Args:
        results: Dictionary mapping error count to list of QValueMetrics

    Returns:
        Dictionary mapping error count to AggregatedMetrics
    """
    summary = {}

    for num_errors, metrics_list in results.items():
        if not metrics_list:
            continue

        recalls = [m.recall for m in metrics_list]
        precisions = [m.precision for m in metrics_list]
        mrrs = [m.mrr for m in metrics_list]
        mean_ranks = [m.mean_rank for m in metrics_list]
        max_ranks = [m.max_rank for m in metrics_list]
        stop_ranks = [m.stop_rank for m in metrics_list]

        # Get top_k from first metrics (same for all in this group)
        top_k = metrics_list[0].top_k

        summary[num_errors] = AggregatedMetrics(
            num_errors=num_errors,
            top_k=top_k,
            samples=len(metrics_list),
            mean_recall=float(np.mean(recalls)),
            std_recall=float(np.std(recalls)),
            mean_precision=float(np.mean(precisions)),
            std_precision=float(np.std(precisions)),
            exact_match_rate=float(np.mean([m.exact_match for m in metrics_list])),
            mean_mrr=float(np.mean(mrrs)),
            std_mrr=float(np.std(mrrs)),
            mean_error_rank=float(np.mean(mean_ranks)),
            std_error_rank=float(np.std(mean_ranks)),
            mean_max_rank=float(np.mean(max_ranks)),
            mean_stop_rank=float(np.mean(stop_ranks)),
            std_stop_rank=float(np.std(stop_ranks)),
            mean_errors_in_top_k=float(np.mean([m.errors_in_top_k for m in metrics_list])),
            stop_in_top_k_rate=float(np.mean([m.stop_in_top_k for m in metrics_list]))
        )

    return summary


def run_analysis(config_path: str,
                 model_path: str,
                 samples_per_k: int = 1000,
                 max_errors: int = 6,
                 min_errors: int = 1,
                 top_k: Optional[int] = None,
                 seed: int = 42,
                 verbose: bool = True) -> Dict[int, AggregatedMetrics]:
    """
    Run the Q-value error recognition analysis.

    Args:
        config_path: Path to experiment config YAML
        model_path: Path to trained model checkpoint
        samples_per_k: Number of test cases per error count
        max_errors: Maximum number of errors to test
        min_errors: Minimum number of errors to test
        top_k: Fixed top-k threshold. If None, uses (num_errors + 1) adaptively
        seed: Random seed for reproducibility
        verbose: Whether to print progress

    Returns:
        Dictionary with aggregated results per error count
    """
    if samples_per_k < 1:
        raise ValueError(f"samples_per_k must be at least 1, got {samples_per_k}")
    if min_errors < 1 or max_errors < min_errors:
        raise ValueError(
            f"error range must satisfy 1 <= min_errors <= max_errors, got "
            f"min_errors={min_errors}, max_errors={max_errors}")
    if top_k is not None and top_k < 1:
        raise ValueError(f"top_k must be at least 1 when given, got {top_k}")

    # Report progress ten times per error count, but never divide by zero on a run
    # of fewer than ten samples.
    progress_interval = max(1, samples_per_k // 10)

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    code_id = config.get('code_id', '64x128')
    network_config = config.get('training', {}).get('network', {'type': 'SimpleMLP'})

    if verbose:
        print(f"Loading LDPC code: {code_id}")

    # Initialize components
    code = LDPC_Code.from_default(code_id)
    env = LDPCEnvironment(
        code=code,
        max_steps=config.get('environment', {}).get('max_steps', 10),
        max_flips_vector=max_errors
    )

    if verbose:
        print(f"Code dimensions: n={code.n}, m={code.m}, k={code.k}")
        print(f"State dim: {code.state_dimensions['total']}, Action dim: {code.action_dim}")
        print(f"Network config: {network_config}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose:
        print(f"Using device: {device}")
        print(f"Loading model from: {model_path}")
        if top_k is not None:
            print(f"Top-k threshold: {top_k} (fixed)")
        else:
            print(f"Top-k threshold: adaptive (num_errors + 1)")

    agent = load_agent(
        model_path,
        state_dim=code.state_dimensions['total'],
        action_dim=code.action_dim,
        network_config=network_config,
        device=device
    )

    rng = np.random.default_rng(seed)

    # Collect results
    results: Dict[int, List[QValueMetrics]] = {ne: [] for ne in range(min_errors, max_errors + 1)}

    for num_err in range(min_errors, max_errors + 1):
        if verbose:
            threshold_str = f"top-{top_k}" if top_k else f"top-{num_err + 1}"
            print(f"\nTesting {num_err} errors, {threshold_str} ({samples_per_k} samples)...", end='', flush=True)

        for i in range(samples_per_k):
            # Generate test case
            original, noisy, error_pos = generate_test_case(code, num_err, rng)

            # Get initial state (first step only)
            state = env.reset(noisy_codeword=noisy, original_codeword=original)

            # Get Q-values from network
            q_values = agent.get_q_values(state)

            # Compute metrics
            metrics = compute_metrics(q_values, error_pos, code.n, top_k=top_k)
            results[num_err].append(metrics)

            # Progress indicator
            if verbose and (i + 1) % progress_interval == 0:
                print('.', end='', flush=True)

        if verbose:
            print(' done')

    return aggregate_results(results)


def print_results(summary: Dict[int, AggregatedMetrics], code_n: int, top_k: Optional[int] = None) -> None:
    """Print formatted results table."""
    print("\n" + "=" * 80)
    print("Q-VALUE ERROR RECOGNITION ANALYSIS RESULTS")
    print("=" * 80)

    # Determine if using fixed or adaptive threshold
    if top_k is not None:
        threshold_mode = f"Fixed top-{top_k}"
    else:
        threshold_mode = "Adaptive top-(num_errors+1)"

    print(f"\nThreshold Mode: {threshold_mode}")
    print(f"Action Space: {code_n + 1} actions ({code_n} bit flips + STOP)")

    print("\nMetric Definitions:")
    print("  - Recall: Fraction of actual errors found in top-k Q-values")
    print("  - Precision: Fraction of top-k Q-values that are actual errors")
    print("  - Exact Match: Rate where ALL errors appear in top-k predictions")
    print("  - Mean Rank: Average rank of error positions (1 = highest Q-value)")
    print("  - Max Rank: Worst (highest) rank among error positions")
    print("  - MRR: Mean Reciprocal Rank (higher = errors ranked higher)")
    print("  - STOP Rank: Where STOP action ranks (1 = highest Q-value)")

    for num_err in sorted(summary.keys()):
        r = summary[num_err]
        print(f"\n{'─' * 80}")
        print(f"{num_err} errors | top-{r.top_k} threshold | {r.samples} samples")
        print(f"{'─' * 80}")
        print(f"  Recall:              {r.mean_recall:.4f} +/- {r.std_recall:.4f}")
        print(f"  Precision:           {r.mean_precision:.4f} +/- {r.std_precision:.4f}")
        print(f"  Exact Match Rate:    {r.exact_match_rate:.4f}")
        print(f"  Errors in Top-{r.top_k}:    {r.mean_errors_in_top_k:.2f} / {num_err}")
        print(f"  Mean Error Rank:     {r.mean_error_rank:.1f} +/- {r.std_error_rank:.1f}")
        print(f"  Mean Max Rank:       {r.mean_max_rank:.1f}")
        print(f"  MRR:                 {r.mean_mrr:.4f} +/- {r.std_mrr:.4f}")
        print(f"  Mean STOP Rank:      {r.mean_stop_rank:.1f} +/- {r.std_stop_rank:.1f}")
        print(f"  STOP in Top-{r.top_k}:      {r.stop_in_top_k_rate:.4f}")

    # Overall summary
    print(f"\n{'=' * 80}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 80}")

    all_recalls = [summary[k].mean_recall for k in sorted(summary.keys())]
    all_exact = [summary[k].exact_match_rate for k in sorted(summary.keys())]
    all_mean_ranks = [summary[k].mean_error_rank for k in sorted(summary.keys())]

    print(f"  Average Recall:           {np.mean(all_recalls):.4f}")
    print(f"  Average Exact Match:      {np.mean(all_exact):.4f}")
    print(f"  Average Mean Error Rank:  {np.mean(all_mean_ranks):.1f} / {code_n + 1}")
    print(f"  Random Baseline Rank:     {(code_n + 1) / 2:.1f}")

    # Interpretation guidance
    print(f"\n{'─' * 80}")
    print("INTERPRETATION GUIDE")
    print(f"{'─' * 80}")
    avg_recall = np.mean(all_recalls)
    avg_rank = np.mean(all_mean_ranks)
    random_rank = (code_n + 1) / 2

    if avg_recall > 0.8:
        print("  [EXCELLENT] Agent strongly recognizes errors from initial observation")
    elif avg_recall > 0.6:
        print("  [GOOD] Agent has reasonable error recognition from initial state")
    elif avg_recall > 0.4:
        print("  [MODERATE] Agent shows some error recognition capability")
    else:
        print("  [WEAK] Agent struggles to identify errors from initial observation")

    rank_ratio = avg_rank / random_rank
    print(f"  Error ranks are {rank_ratio:.1%} of random baseline ({avg_rank:.1f} vs {random_rank:.1f})")


def main():
    parser = argparse.ArgumentParser(
        description='Q-Value Error Recognition Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Adaptive threshold (top-(num_errors+1) for each test case):
    PYTHONPATH=. python scripts/run_qvalue_analysis.py \\
        --config config/experiments_config/64x128/experiment_config.yml \\
        --model models/64x128_model.pth

  Fixed top-k threshold (e.g., always check top-10):
    PYTHONPATH=. python scripts/run_qvalue_analysis.py \\
        --config config/experiments_config/64x128/experiment_config.yml \\
        --model models/64x128_model.pth \\
        --top-k 10

  With custom parameters and JSON output:
    PYTHONPATH=. python scripts/run_qvalue_analysis.py \\
        --config config/experiments_config/64x128/experiment_config.yml \\
        --model models/64x128_model.pth \\
        --samples-per-k 5000 \\
        --top-k 20 \\
        --output results/qvalue_analysis.json
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to experiment configuration YAML file'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model checkpoint (.pth file)'
    )
    parser.add_argument(
        '--samples-per-k',
        type=int,
        default=1000,
        help='Number of test cases per error count (default: 1000)'
    )
    parser.add_argument(
        '--min-errors',
        type=int,
        default=1,
        help='Minimum number of errors to test (default: 1)'
    )
    parser.add_argument(
        '--max-errors',
        type=int,
        default=6,
        help='Maximum number of errors to test (default: 6)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=None,
        help='Fixed top-k threshold. If not specified, uses (num_errors + 1) adaptively'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file path (optional)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    # Validate paths
    config_path = Path(args.config)
    model_path = Path(args.model)

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return 1

    # Run analysis
    summary = run_analysis(
        config_path=str(config_path),
        model_path=str(model_path),
        samples_per_k=args.samples_per_k,
        max_errors=args.max_errors,
        min_errors=args.min_errors,
        top_k=args.top_k,
        seed=args.seed,
        verbose=not args.quiet
    )

    # Get code dimensions for output
    with open(args.config) as f:
        config = yaml.safe_load(f)
    code = LDPC_Code.from_default(config.get('code_id', '64x128'))

    # Print results
    print_results(summary, code.n, top_k=args.top_k)

    # Save to JSON if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert dataclasses to dicts for JSON serialization
        output_data = {
            'config': {
                'config_path': str(config_path),
                'model_path': str(model_path),
                'code_id': config.get('code_id', '64x128'),
                'samples_per_k': args.samples_per_k,
                'min_errors': args.min_errors,
                'max_errors': args.max_errors,
                'top_k': args.top_k,
                'top_k_mode': 'fixed' if args.top_k else 'adaptive',
                'seed': args.seed
            },
            'results': {str(k): asdict(v) for k, v in summary.items()}
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == '__main__':
    exit(main())
