#!/usr/bin/env python3
"""
Extract Experiences Script

This script extracts a specified number of experiences from a replay buffer file
and saves them to a new file. Useful for creating smaller subsets of large
experience collections.

Usage:
    PYTHONPATH=. python scripts/extract_experiences.py \
        --input data/expert_experiences_64x128_minsum_10k_ep.pkl \
        --output data/expert_experiences_64x128_minsum.pkl \
        --count 10000
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.replay_buffer import ReplayBuffer
from src.core.ldpc_code import LDPC_Code


def extract_experiences(input_file: str, output_file: str, count: int, code_id: str = "64x128"):
    """
    Extract a specified number of experiences from an input file and save to output file.
    
    Args:
        input_file: Path to input replay buffer file
        output_file: Path to output replay buffer file
        count: Number of experiences to extract
        code_id: LDPC code ID for validation
    """
    print(f"Extracting {count:,} experiences")
    print(f"   Input:  {input_file}")
    print(f"   Output: {output_file}")
    
    # Setup LDPC code for validation
    ldpc_code = LDPC_Code.from_default(code_id)
    print(f"   Code:   {code_id} ({ldpc_code.n}x{ldpc_code.k})")
    
    # Load input replay buffer
    print(f"\nLoading input file...")
    input_buffer = ReplayBuffer(
        capacity=1000000,  # Large capacity to hold all experiences
        replay_file=input_file,
        code=ldpc_code
    )
    
    total_experiences = len(input_buffer)
    print(f"   Found {total_experiences:,} experiences")
    
    if total_experiences < count:
        print(f"Warning: Requested {count:,} experiences but only {total_experiences:,} available")
        count = total_experiences
    
    # Create output buffer and extract experiences
    print(f"\nExtracting first {count:,} experiences...")
    output_buffer = ReplayBuffer(
        capacity=count,
        code=ldpc_code
    )
    
    # Copy experiences one by one
    for i in range(count):
        experience = input_buffer.buffer[i]
        output_buffer.push(*experience)
        
        if (i + 1) % 1000 == 0:
            print(f"   Copied {i + 1:,}/{count:,} experiences")
    
    # Save to output file
    print(f"\nSaving to output file...")
    success = output_buffer.save_to_file(output_file)
    
    if success:
        output_path = Path(output_file)
        file_size_mb = output_path.stat().st_size / 1024 / 1024
        
        print(f"Extraction complete!")
        print(f"   Extracted: {len(output_buffer):,} experiences")
        print(f"   Output: {output_file}")
        print(f"   File size: {file_size_mb:.1f} MB")
    else:
        print(f"Failed to save to {output_file}")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract a specified number of experiences from a replay buffer file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Extract 10K experiences:
    PYTHONPATH=. python scripts/extract_experiences.py \\
        --input data/expert_experiences_64x128_minsum_10k_ep.pkl \\
        --output data/expert_experiences_64x128_minsum.pkl \\
        --count 10000

  Extract first 5K experiences:
    PYTHONPATH=. python scripts/extract_experiences.py \\
        --input data/large_buffer.pkl \\
        --output data/small_buffer.pkl \\
        --count 5000
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help="Input replay buffer file path"
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help="Output replay buffer file path"
    )
    
    parser.add_argument(
        '--count',
        type=int,
        required=True,
        help="Number of experiences to extract"
    )
    
    parser.add_argument(
        '--code-id',
        type=str,
        default="64x128",
        help="LDPC code ID for validation (default: 64x128)"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract experiences
    try:
        extract_experiences(
            input_file=args.input,
            output_file=args.output,
            count=args.count,
            code_id=args.code_id
        )
    except Exception as e:
        print(f"Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()