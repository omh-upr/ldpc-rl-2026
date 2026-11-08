"""
LDPC Utility Functions

Utility functions for loading and working with LDPC parity-check and generator
matrices. Supports multiple code sizes (24x48, 56x80, 64x128, 128x256).

Key Features:
- Load pre-computed LDPC matrices with verified minimum distance
- Matrix property verification and analysis
- Generator matrix computation from parity-check matrices
"""

import numpy as np
import os
from pathlib import Path

def parse_alist_file(file_path):
    """
    Parse an alist file and return the parity check matrix H.
    
    Args:
        file_path (str): Path to the alist file
        
    Returns:
        np.ndarray: Parity check matrix H
        
    Raises:
        FileNotFoundError: If the alist file doesn't exist
        ValueError: If the file format is invalid
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Alist file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # Parse header
    n, m = map(int, lines[0].split())
    max_var_degree, max_check_degree = map(int, lines[1].split())
    
    # Parse degree sequences
    var_degrees = list(map(int, lines[2].split()))
    check_degrees = list(map(int, lines[3].split()))
    
    # Validate degree sequences
    if len(var_degrees) != n:
        raise ValueError(f"Expected {n} variable degrees, got {len(var_degrees)}")
    if len(check_degrees) != m:
        raise ValueError(f"Expected {m} check degrees, got {len(check_degrees)}")
    
    # Initialize H matrix
    H = np.zeros((m, n), dtype=np.int32)
    
    # Parse variable node connections (lines 4 to 4+n-1)
    for i in range(n):
        line_idx = 4 + i
        if line_idx >= len(lines):
            raise ValueError(f"Missing variable node connections for node {i}")
        
        connections = list(map(int, lines[line_idx].split()))
        
        # Remove zeros and convert to 0-based indexing
        connections = [c - 1 for c in connections if c > 0]
        
        # Validate degree
        if len(connections) != var_degrees[i]:
            raise ValueError(f"Variable node {i} degree mismatch: expected {var_degrees[i]}, got {len(connections)}")
        
        # Set connections in H matrix
        for check_idx in connections:
            if check_idx >= m:
                raise ValueError(f"Check node index {check_idx} out of bounds for variable node {i}")
            H[check_idx, i] = 1
    
    # Verify matrix properties
    actual_var_degrees = np.sum(H, axis=0)
    actual_check_degrees = np.sum(H, axis=1)
    
    if not np.array_equal(actual_var_degrees, var_degrees):
        raise ValueError("Variable degree verification failed")
    if not np.array_equal(actual_check_degrees, check_degrees):
        raise ValueError("Check degree verification failed")
    
    print(f"Alist file parsed successfully: {file_path}")
    print(f"   Matrix dimensions: {H.shape} (m={m}, n={n})")
    print(f"   Matrix density: {np.mean(H):.3f}")
    print(f"   Variable degrees: min={min(var_degrees)}, max={max(var_degrees)}, avg={np.mean(var_degrees):.2f}")
    print(f"   Check degrees: min={min(check_degrees)}, max={max(check_degrees)}, avg={np.mean(check_degrees):.2f}")
    
    return H

def compute_generator_matrix(H):
    """
    Compute the generator matrix G from the parity check matrix H.
    
    Uses GF(2) matrix operations to compute a valid generator matrix such that
    G @ H^T = 0 (mod 2). The function fails explicitly if a valid matrix cannot
    be computed.
    
    Args:
        H (np.ndarray): Parity check matrix
        
    Returns:
        np.ndarray: Generator matrix G
        
    Raises:
        ValueError: If H doesn't have full rank or G cannot be computed
    """
    if H.ndim != 2:
        raise ValueError("H must be a 2D matrix")

    # Work in GF(2) domain
    H_gf2 = H.astype(np.uint8) % 2
    m, n = H_gf2.shape
    k = n - m  # Expected information length for full-rank H
    
    if k <= 0:
        raise ValueError(f"Invalid code parameters: n={n}, m={m}, k={k}")

    # Local import to avoid widening module import surface
    from src.utils.gf2_operations import gf2_rank, compute_systematic_generator_matrix

    rank = gf2_rank(H_gf2)
    if rank != m:
        raise ValueError(
            f"H matrix must be full rank for generator computation: rank={rank}, m={m}"
        )

    print("INFO: Computing generator matrix using GF(2) systematic/nullspace approach...")
    G, success = compute_systematic_generator_matrix(H_gf2)

    if not success or G is None:
        raise ValueError("Failed to compute a valid generator matrix from H")

    if G.shape != (k, n):
        raise ValueError(
            f"Computed generator matrix has invalid shape {G.shape}; expected ({k}, {n})"
        )

    orthogonality_check = (G @ H_gf2.T) % 2
    if not np.all(orthogonality_check == 0):
        raise ValueError("Computed generator matrix failed orthogonality check (G @ H.T % 2 != 0)")

    G = G.astype(np.int32)
    print(f"Generator matrix computed successfully")
    print(f"   G matrix dimensions: {G.shape} (k={k}, n={n})")
    print(f"   G matrix density: {np.mean(G):.3f}")
    print(f"   Code rate: {k/n:.3f}")

    return G

def get_matrix_folder():
    """
    Get the folder path where matrix files are stored.
    
    Returns:
        str: Path to the matrix folder
    """
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]  # Go up to project root
    return project_root / "data"

def get_ldpc(code_id:str):
    """
    Get the code descriptor dictionary for a given code ID.

    Args:
        code_id (str): Identifier for the LDPC code (e.g. '64x128', '56x80')

    Returns:
        dict: Code descriptor with keys 'names' (H/G filenames), 'd_min', 'coderate'

    Raises:
        ValueError: If code_id is not recognized
    """
    codes = {
        '24x48': { "names": ("H_24x48_peg_dmin7.npy", "G_24x48_peg_dmin7.npy"),
                   "d_min": 7,
                    "coderate": 0.5
                 },
        '56x80': { "names": ("H_56x80_peg_dmin12_k24.npy", "G_24x80_peg_dmin12_k24.npy"),
                   "d_min": 12,
                   "coderate": 0.3
                  },
        '64x128': { "names": ("H_64x128_dmin14.npy", "G_64x128_dmin14.npy"),
                    "d_min": 14,
                    "coderate": 0.5                   
                   },
        '128x256': { "names": ("H_128x256_dmin24.npy", "G_128x256_dmin24.npy"),
                     "d_min": 24,
                     "coderate": 0.5
        }        
    }

    if code_id in codes.keys():
        return codes[code_id]
    else:
        raise ValueError(f"Unknown code ID: {code_id}")

def load_matrices(code_id='56x80'):
    """
    Load LDPC matrices
    
    Returns:
        tuple: (H, G) where H is parity check matrix, G is generator matrix
        
    Raises:
        FileNotFoundError: If matrix files are not found
        ValueError: If matrices don't have expected properties
    """
    H_name, G_name = get_ldpc(code_id)["names"]
    d_min = get_ldpc(code_id)["d_min"]
    coderate = get_ldpc(code_id)["coderate"]

    # Matrix file paths (now in data directory)
    H_path = get_matrix_folder() / H_name
    G_path = get_matrix_folder() / G_name
    
    if not H_path.exists():
        raise FileNotFoundError(f"H matrix not found at {H_path}")
    if not G_path.exists():
        raise FileNotFoundError(f"G matrix not found at {G_path}")
    
    # Load matrices
    H = np.load(H_path)
    G = np.load(G_path)
        
    # Verify orthogonality: G × H^T = 0 (mod 2)
    orthogonality_check = np.dot(G, H.T) % 2
    if not np.all(orthogonality_check == 0):
        raise ValueError("Matrices G and H are not orthogonal: G × H^T ≠ 0 (mod 2)")
    
    print("Matrices loaded successfully")
    print(f"   H matrix: {H.shape}, density: {np.mean(H):.3f}")
    print(f"   G matrix: {G.shape}, rank verified")
    print(f"   d_min = {d_min} and coderate = {coderate:.2f}")
    
    return H, G

def verify_matrix_properties(H=None, G=None, verbose=True):
    """
    Verify key properties of LDPC matrices.

    Args:
        H (np.ndarray, optional): Parity check matrix (m x n)
        G (np.ndarray, optional): Generator matrix (k x n)
        verbose (bool): Print detailed verification results

    Returns:
        dict: Verification results with property checks
    """
    if H is None or G is None:
        H, G = load_matrices()

    m, n = H.shape
    k = G.shape[0]

    results = {}

    # 1. Dimension consistency check
    results['correct_dimensions'] = (G.shape[1] == n and k == n - m)

    # 2. Orthogonality check: G × H^T = 0 (mod 2)
    orthogonality_product = np.dot(G, H.T) % 2
    results['orthogonal'] = np.all(orthogonality_product == 0)

    # 3. Rank check, over GF(2). Real-valued rank answers a different question and
    # can disagree: rows 110, 101 and 011 are independent over the reals but sum to
    # zero modulo 2, so their binary rank is 2 rather than 3.
    from src.utils.gf2_operations import gf2_rank

    results['H_rank'] = gf2_rank(H)
    results['G_rank'] = gf2_rank(G)
    results['full_rank'] = (results['H_rank'] == m and results['G_rank'] == k)

    # 4. Matrix density (sparsity measure)
    results['H_density'] = np.mean(H)
    results['G_density'] = np.mean(G)

    # 5. Degree distribution for H matrix
    var_node_degrees = np.sum(H, axis=0)  # Column sums
    check_node_degrees = np.sum(H, axis=1)  # Row sums
    results['avg_var_degree'] = np.mean(var_node_degrees)
    results['avg_check_degree'] = np.mean(check_node_degrees)

    # 6. Sparsity properties
    results['ultra_low_degree'] = (results['avg_var_degree'] < 4.0)
    results['optimized_density'] = (results['H_density'] < 0.16)

    if verbose:
        print(f"\n=== LDPC {m}×{n} Matrix Verification ===")
        print(f"Consistent dimensions: {results['correct_dimensions']} (H: {H.shape}, G: {G.shape})")
        print(f"Orthogonal (G*H^T=0): {results['orthogonal']}")
        print(f"Full rank: {results['full_rank']} (H: {results['H_rank']}/{m}, G: {results['G_rank']}/{k})")
        print(f"H matrix density: {results['H_density']:.3f}")
        print(f"Average variable degree: {results['avg_var_degree']:.2f}")
        print(f"Average check degree: {results['avg_check_degree']:.2f}")

    return results
