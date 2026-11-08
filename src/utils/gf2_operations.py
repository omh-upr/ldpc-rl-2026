"""
GF(2) Matrix Operations for LDPC Code Generation

This module provides essential operations for working with binary matrices over
the Galois Field GF(2), specifically for computing systematic generator matrices
from parity-check matrices of LDPC codes.

Key Functions:
- gaussian_elimination_gf2: Transform matrix to row echelon form
- systematic_form_gf2: Convert H matrix to systematic [P | I] form
- compute_systematic_generator_matrix: Generate orthogonal G matrix from H

Mathematical Foundation:
For LDPC codes, we need G × H^T = 0 (mod 2), where:
- H (m×n): Parity-check matrix
- G (k×n): Generator matrix, k = n - m
- Systematic form: H = [P | I_m] implies G = [I_k | P^T]
"""

import numpy as np
from typing import Tuple, Optional


def gf2_rank(matrix: np.ndarray) -> int:
    """
    Compute the rank of a binary matrix over GF(2).

    Every column is a candidate pivot, so all n of them are scanned. Stopping at
    column m - 1 would undercount any wide matrix whose independent pivots do not
    all fall in the leading square block: [[0, 1]] has rank 1, but a scan that sees
    only its zero first column reports 0.

    Args:
        matrix: Binary matrix (values 0 or 1)

    Returns:
        Rank of the matrix over GF(2)
    """
    work_matrix = (matrix.astype(np.uint8) % 2).copy()
    m, n = work_matrix.shape
    rank = 0

    for col in range(n):
        # Rank cannot exceed the number of rows, so once every row carries a pivot
        # the remaining columns can add nothing.
        if rank == m:
            break

        # Find pivot row at or below the current rank
        candidates = np.nonzero(work_matrix[rank:, col])[0]
        if candidates.size == 0:
            continue  # No pivot in this column

        pivot_row = rank + int(candidates[0])

        # Swap rows if needed
        if pivot_row != rank:
            work_matrix[[rank, pivot_row]] = work_matrix[[pivot_row, rank]]

        # Eliminate this column from every other row. Entries are 0 or 1, so XOR is
        # addition modulo 2, applied to all affected rows at once.
        affected = np.nonzero(work_matrix[:, col])[0]
        affected = affected[affected != rank]
        if affected.size:
            work_matrix[affected] ^= work_matrix[rank]

        rank += 1

    return rank


def gaussian_elimination_gf2(matrix: np.ndarray, augmented: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Gaussian elimination on a binary matrix over GF(2).

    Args:
        matrix: Input binary matrix (m×n)
        augmented: Optional augmented matrix to transform simultaneously

    Returns:
        tuple: (row_echelon_matrix, row_permutation, column_permutation)
            - row_echelon_matrix: Matrix in row echelon form
            - row_permutation: Permutation matrix for rows
            - column_permutation: Permutation matrix for columns
    """
    matrix = matrix.astype(np.uint8) % 2
    m, n = matrix.shape

    # Create working copy
    work_matrix = matrix.copy()

    # Initialize permutation tracking
    row_perm = np.arange(m)
    col_perm = np.arange(n)

    # Handle augmented matrix if provided
    if augmented is not None:
        augmented = augmented.astype(np.uint8) % 2
        work_augmented = augmented.copy()
    else:
        work_augmented = None

    current_row = 0

    for col in range(n):
        if current_row >= m:
            break

        # Find pivot row (first non-zero element in current column)
        pivot_row = None
        for row in range(current_row, m):
            if work_matrix[row, col] == 1:
                pivot_row = row
                break

        if pivot_row is None:
            continue  # No pivot in this column, move to next column

        # Swap rows if needed
        if pivot_row != current_row:
            work_matrix[[current_row, pivot_row]] = work_matrix[[pivot_row, current_row]]
            row_perm[[current_row, pivot_row]] = row_perm[[pivot_row, current_row]]
            if work_augmented is not None:
                work_augmented[[current_row, pivot_row]] = work_augmented[[pivot_row, current_row]]

        # Eliminate other rows (both above and below pivot)
        for row in range(m):
            if row != current_row and work_matrix[row, col] == 1:
                work_matrix[row] = (work_matrix[row] + work_matrix[current_row]) % 2
                if work_augmented is not None:
                    work_augmented[row] = (work_augmented[row] + work_augmented[current_row]) % 2

        current_row += 1

    # Create permutation matrices
    P_row = np.zeros((m, m), dtype=np.uint8)
    P_col = np.eye(n, dtype=np.uint8)  # No column permutation for now
    for i, j in enumerate(row_perm):
        P_row[i, j] = 1

    return work_matrix, P_row, P_col


def systematic_form_gf2(H: np.ndarray, tolerance: float = 1e-10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Transform parity-check matrix H to systematic form [P | I_m].

    Args:
        H: Parity-check matrix (m×n)
        tolerance: Numerical tolerance (unused for GF(2), kept for interface)

    Returns:
        tuple: (H_systematic, P_matrix, column_permutation, success)
            - H_systematic: Matrix in systematic form [P | I_m]
            - P_matrix: The P submatrix extracted from systematic form
            - column_permutation: Permutation applied to columns
            - success: Whether systematic form was achieved
    """
    H = H.astype(np.uint8) % 2
    m, n = H.shape

    if m > n:
        raise ValueError(f"Invalid matrix dimensions: m={m} > n={n}")

    # Create working copy
    work_H = H.copy()
    col_perm = np.arange(n)

    # Try to achieve systematic form by column operations
    for target_row in range(m):
        # Find a column with 1 in target_row position (starting from rightmost)
        pivot_col = None
        for col in range(n-1, -1, -1):  # Start from right (where identity should be)
            if work_H[target_row, col] == 1:
                pivot_col = col
                break

        if pivot_col is None:
            print(f"Cannot find pivot for row {target_row}")
            return work_H, None, col_perm, False

        # Move this column to position (n-1-target_row) to form identity on the right
        target_col = n - 1 - target_row
        if pivot_col != target_col:
            # Swap columns
            work_H[:, [pivot_col, target_col]] = work_H[:, [target_col, pivot_col]]
            col_perm[[pivot_col, target_col]] = col_perm[[target_col, pivot_col]]

        # Eliminate other 1's in this column
        for row in range(m):
            if row != target_row and work_H[row, target_col] == 1:
                work_H[row] = (work_H[row] + work_H[target_row]) % 2

    # Check if we achieved systematic form
    identity_part = work_H[:, n-m:]
    expected_identity = np.eye(m, dtype=np.uint8)

    if not np.array_equal(identity_part, expected_identity):
        # Try alternative approach with Gaussian elimination
        work_H, row_perm, _ = gaussian_elimination_gf2(H.T)
        work_H = work_H.T

        # Check if this worked
        identity_part = work_H[:, n-m:]
        if not np.array_equal(identity_part, expected_identity):
            print("Could not achieve systematic form")
            return work_H, None, col_perm, False

    # Extract P matrix (left part)
    P_matrix = work_H[:, :n-m]

    print(f"Systematic form achieved: H = [P | I_{m}]")
    print(f"   P matrix dimensions: {P_matrix.shape}")
    print(f"   Identity part verified: {np.array_equal(identity_part, expected_identity)}")

    return work_H, P_matrix, col_perm, True


def compute_generator_matrix_nullspace_v2(H: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Compute generator matrix using improved nullspace approach.

    This implementation uses a more robust method to find the nullspace of H
    over GF(2) by solving the equation H·g = 0 for each generator vector g.

    Args:
        H: Parity-check matrix (m×n), binary values

    Returns:
        tuple: (G, success)
    """
    H = H.astype(np.uint8) % 2
    m, n = H.shape

    # Determine actual nullspace dimension
    actual_rank = gf2_rank(H)
    k = n - actual_rank  # Use actual rank, not assumed rank

    if k <= 0:
        print(f"No nullspace exists: rank({actual_rank}) >= n({n})")
        return None, False

    print(f"Computing generator matrix using improved nullspace approach:")
    print(f"  H dimensions: {H.shape}")
    print(f"  H rank: {actual_rank}/{m}")
    print(f"  Nullspace dimension: {k}")
    print(f"  Target G dimensions: ({k}, {n})")

    # Method: Solve H·x = 0 systematically
    # First, put H in reduced row echelon form
    H_work = H.copy()

    # Track column permutations
    col_order = list(range(n))
    pivot_cols = []

    # Forward elimination
    current_row = 0
    for col in range(n):
        if current_row >= m:
            break

        # Find pivot in current column
        pivot_row = None
        for row in range(current_row, m):
            if H_work[row, col] == 1:
                pivot_row = row
                break

        if pivot_row is None:
            continue  # No pivot in this column

        # Swap rows to bring pivot to diagonal
        if pivot_row != current_row:
            H_work[[current_row, pivot_row]] = H_work[[pivot_row, current_row]]

        pivot_cols.append(col)

        # Eliminate other rows
        for row in range(m):
            if row != current_row and H_work[row, col] == 1:
                H_work[row] = (H_work[row] + H_work[current_row]) % 2

        current_row += 1

    print(f"  Found {len(pivot_cols)} pivot columns: {pivot_cols[:10]}{'...' if len(pivot_cols) > 10 else ''}")

    # Free variables are columns not in pivot_cols
    free_cols = [col for col in range(n) if col not in pivot_cols]
    print(f"  Found {len(free_cols)} free variables: {free_cols[:10]}{'...' if len(free_cols) > 10 else ''}")

    if len(free_cols) != k:
        print(f"Dimension mismatch: expected {k} free variables, got {len(free_cols)}")
        return None, False

    # Generate basis vectors for nullspace
    G = np.zeros((k, n), dtype=np.uint8)

    for i, free_col in enumerate(free_cols):
        # Create a vector with 1 in the free column position
        x = np.zeros(n, dtype=np.uint8)
        x[free_col] = 1

        # Solve for the pivot variables using back-substitution
        for j in range(len(pivot_cols) - 1, -1, -1):
            pivot_col = pivot_cols[j]
            pivot_row = j  # After elimination, pivot is at diagonal

            if pivot_row < m:  # Safety check
                # Calculate what the pivot variable should be
                sum_val = 0
                for col in range(pivot_col + 1, n):
                    sum_val += H_work[pivot_row, col] * x[col]

                x[pivot_col] = sum_val % 2

        G[i] = x

    # Verify orthogonality
    product = (G @ H.T) % 2
    is_orthogonal = np.all(product == 0)
    G_rank = gf2_rank(G)

    print(f"Improved nullspace generator computed:")
    print(f"   G dimensions: {G.shape}")
    print(f"   G rank: {G_rank}/{k} {'PASS' if G_rank == k else 'FAIL'}")
    print(f"   G density: {np.mean(G):.3f}")
    print(f"   Orthogonality check: {'PASSED' if is_orthogonal else 'FAILED'}")

    if not is_orthogonal:
        print(f"   Non-zero elements in G×H^T: {np.sum(product != 0)}/{product.size}")

    success = is_orthogonal and (G_rank == k)
    return G, success


def compute_generator_matrix_nullspace(H: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Compute generator matrix G using nullspace approach over GF(2).

    This method finds the nullspace of H over GF(2), giving us G such that
    G × H^T = 0 (mod 2). Falls back to alternative approaches if the primary
    nullspace computation fails (e.g., for rank-deficient matrices).

    Args:
        H: Parity-check matrix (m×n), binary values

    Returns:
        tuple: (G, success)
            - G: Generator matrix (k×n) where k = n - m
            - success: Whether computation succeeded
    """
    # Try the improved version first
    G, success = compute_generator_matrix_nullspace_v2(H)
    if success:
        return G, True

    print(f"Improved nullspace failed, trying original approach...")

    H = H.astype(np.uint8) % 2
    m, n = H.shape
    k = n - m

    if k <= 0:
        raise ValueError(f"Invalid code parameters: n={n}, m={m}, k={k} <= 0")

    print(f"Computing generator matrix using original nullspace approach:")
    print(f"  H dimensions: {H.shape} (m={m}, n={n})")
    print(f"  Target G dimensions: ({k}, {n})")
    print(f"  Code rate: {k/n:.3f}")

    # Step 1: Compute row-reduced form of H^T to find nullspace
    HT = H.T  # n×m matrix

    # Augment with identity to track transformations
    augmented = np.hstack([HT, np.eye(n, dtype=np.uint8)])  # n×(m+n)

    # Gaussian elimination on augmented matrix
    work_matrix, _, _ = gaussian_elimination_gf2(augmented)

    # Step 2: Find nullspace vectors
    # After elimination, rows corresponding to zero rows in H^T part give nullspace
    HT_part = work_matrix[:, :m]  # First m columns (H^T part)
    transform_part = work_matrix[:, m:]  # Last n columns (transformation part)

    # Find rows where H^T part is all zeros
    zero_rows = []
    for i in range(n):
        if np.all(HT_part[i] == 0):
            zero_rows.append(i)

    if len(zero_rows) < k:
        print(f"Found only {len(zero_rows)} nullspace vectors, need {k}")
        # Try alternative approach: use SVD-like method over GF(2)
        return compute_generator_matrix_alternative(H)

    # Extract nullspace vectors (these are rows of transform_part)
    G = np.zeros((k, n), dtype=np.uint8)
    for i, row_idx in enumerate(zero_rows[:k]):
        G[i] = transform_part[row_idx]

    # Step 3: Verify orthogonality
    product = (G @ H.T) % 2
    is_orthogonal = np.all(product == 0)

    print(f"Generator matrix computed using original nullspace:")
    print(f"   G dimensions: {G.shape}")
    print(f"   G density: {np.mean(G):.3f}")
    print(f"   Orthogonality check: {'PASSED' if is_orthogonal else 'FAILED'}")

    if not is_orthogonal:
        print(f"   Non-zero elements in G×H^T: {np.sum(product != 0)}/{product.size}")
        return G, False

    return G, True


def compute_generator_matrix_alternative(H: np.ndarray, seed: int = 42) -> Tuple[np.ndarray, bool]:
    """
    Alternative generator matrix computation using basis construction.

    This method constructs a generator matrix by finding vectors that are
    orthogonal to all rows of H over GF(2). Uses iterative approach with
    enhanced linear independence checking.

    Args:
        H: Parity-check matrix (m×n)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        tuple: (G, success)
    """
    rng = np.random.default_rng(seed)
    H = H.astype(np.uint8) % 2
    m, n = H.shape
    k = n - m

    # For rank-deficient H, adjust k to actual nullspace dimension
    actual_rank = gf2_rank(H)
    if actual_rank < m:
        print(f"H matrix is rank deficient: {actual_rank}/{m}")
        k_adjusted = n - actual_rank
        print(f"   Adjusting k: {k} → {k_adjusted}")
        k = k_adjusted

    print(f"Using enhanced alternative generator matrix computation...")
    print(f"  Target k: {k} (nullspace dimension)")

    # Find k linearly independent vectors orthogonal to H
    G_candidates = []
    max_attempts = 10000
    attempts = 0

    # Strategy 1: Try unit vectors and sparse combinations
    for sparsity in [1, 2, 3, 4, 5]:  # Increasing sparsity
        if len(G_candidates) >= k:
            break

        # Try all combinations of given sparsity
        from itertools import combinations
        if sparsity == 1:
            # Unit vectors
            for i in range(n):
                if len(G_candidates) >= k or attempts >= max_attempts:
                    break
                attempts += 1

                candidate = np.zeros(n, dtype=np.uint8)
                candidate[i] = 1

                # Check orthogonality
                if np.all((H @ candidate) % 2 == 0):
                    # Check linear independence
                    if len(G_candidates) == 0:
                        G_candidates.append(candidate.copy())
                        print(f"  Found generator {len(G_candidates)}: unit vector at position {i}")
                    else:
                        test_matrix = np.vstack(G_candidates + [candidate])
                        if gf2_rank(test_matrix) > len(G_candidates):
                            G_candidates.append(candidate.copy())
                            print(f"  Found generator {len(G_candidates)}: unit vector at position {i}")
        else:
            # Sparse combinations
            position_combinations = list(combinations(range(n), sparsity))
            rng.shuffle(position_combinations)  # Randomize order

            for positions in position_combinations[:min(100, len(position_combinations))]:
                if len(G_candidates) >= k or attempts >= max_attempts:
                    break
                attempts += 1

                candidate = np.zeros(n, dtype=np.uint8)
                candidate[list(positions)] = 1

                # Check orthogonality
                if np.all((H @ candidate) % 2 == 0):
                    # Check linear independence
                    if len(G_candidates) == 0:
                        G_candidates.append(candidate.copy())
                        print(f"  Found generator {len(G_candidates)}: sparse vector (sparsity {sparsity})")
                    else:
                        test_matrix = np.vstack(G_candidates + [candidate])
                        if gf2_rank(test_matrix) > len(G_candidates):
                            G_candidates.append(candidate.copy())
                            print(f"  Found generator {len(G_candidates)}: sparse vector (sparsity {sparsity})")

    # Strategy 2: Random search if still not enough
    print(f"  Switching to random search (found {len(G_candidates)}/{k} so far)...")
    while len(G_candidates) < k and attempts < max_attempts:
        attempts += 1

        # Generate random binary vector with controlled density
        density = rng.uniform(0.1, 0.5)  # Random density between 10% and 50%
        candidate = (rng.random(n) < density).astype(np.uint8)

        # Check orthogonality
        if np.all((H @ candidate) % 2 == 0):
            # Check linear independence
            if len(G_candidates) == 0:
                G_candidates.append(candidate.copy())
                print(f"  Found generator {len(G_candidates)}: random vector")
            else:
                test_matrix = np.vstack(G_candidates + [candidate])
                if gf2_rank(test_matrix) > len(G_candidates):
                    G_candidates.append(candidate.copy())
                    print(f"  Found generator {len(G_candidates)}: random vector")

        # Progress update
        if attempts % 1000 == 0:
            print(f"    Search progress: {len(G_candidates)}/{k} found after {attempts} attempts")

    print(f"  Search completed: {len(G_candidates)}/{k} generators found in {attempts} attempts")

    if len(G_candidates) < k:
        print(f"Could not find {k} independent orthogonal vectors")
        print(f"   This might indicate the H matrix has fundamental issues")
        return None, False

    # Form generator matrix with exactly k rows
    G = np.vstack(G_candidates[:k])

    # Final verification
    product = (G @ H.T) % 2
    is_orthogonal = np.all(product == 0)
    final_rank = gf2_rank(G)

    print(f"Enhanced alternative generator matrix computed:")
    print(f"   G dimensions: {G.shape}")
    print(f"   G rank: {final_rank}/{k} {'PASS' if final_rank == k else 'FAIL'}")
    print(f"   G density: {np.mean(G):.3f}")
    print(f"   Orthogonality check: {'PASSED' if is_orthogonal else 'FAILED'}")

    return G, (is_orthogonal and final_rank == k)


def compute_systematic_generator_matrix(H: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Compute systematic generator matrix G from parity-check matrix H.

    This function tries multiple approaches:
    1. Traditional systematic form [P | I]
    2. Nullspace computation over GF(2)
    3. Alternative basis construction

    Args:
        H: Parity-check matrix (m×n), binary values

    Returns:
        tuple: (G, success)
            - G: Generator matrix (k×n) where k = n - m
            - success: Whether computation succeeded
    """
    H = H.astype(np.uint8) % 2
    m, n = H.shape
    k = n - m

    if k <= 0:
        raise ValueError(f"Invalid code parameters: n={n}, m={m}, k={k} <= 0")

    print(f"Computing generator matrix for LDPC code:")
    print(f"  H dimensions: {H.shape} (m={m}, n={n})")
    print(f"  Target G dimensions: ({k}, {n})")
    print(f"  Code rate: {k/n:.3f}")

    # Approach 1: Try systematic form
    print(f"\nApproach 1: Systematic form [P | I]")
    H_sys, P_matrix, col_perm, sys_success = systematic_form_gf2(H)

    if sys_success:
        # Construct G = [I_k | P^T]
        G = np.zeros((k, n), dtype=np.uint8)
        G[:, :k] = np.eye(k, dtype=np.uint8)
        G[:, k:] = P_matrix.T

        # Apply inverse column permutation
        inv_col_perm = np.zeros_like(col_perm)
        inv_col_perm[col_perm] = np.arange(len(col_perm))
        G_permuted = G[:, inv_col_perm]

        # Verify orthogonality
        product = (G_permuted @ H.T) % 2
        if np.all(product == 0):
            print(f"Systematic form approach succeeded!")
            return G_permuted, True
        else:
            print(f"Systematic form created but not orthogonal")

    # Approach 2: Nullspace computation
    print(f"\nApproach 2: Nullspace computation")
    try:
        G, success = compute_generator_matrix_nullspace(H)
        if success:
            print(f"Nullspace approach succeeded!")
            return G, True
        else:
            print(f"Nullspace approach failed orthogonality check")
    except Exception as e:
        print(f"Nullspace approach failed: {e}")

    # Approach 3: Alternative basis construction
    print(f"\nApproach 3: Alternative basis construction")
    try:
        G, success = compute_generator_matrix_alternative(H)
        if success:
            print(f"Alternative approach succeeded!")
            return G, True
        else:
            print(f"Alternative approach failed")
    except Exception as e:
        print(f"Alternative approach failed: {e}")

    print(f"All approaches failed to generate valid generator matrix")
    return None, False


def verify_ldpc_matrices(H: np.ndarray, G: np.ndarray) -> dict:
    """
    Comprehensive verification of LDPC matrix pair (H, G).

    Args:
        H: Parity-check matrix (m×n)
        G: Generator matrix (k×n)

    Returns:
        dict: Verification results
    """
    H = H.astype(np.uint8) % 2
    G = G.astype(np.uint8) % 2

    m, n = H.shape
    k, n_g = G.shape

    results = {
        'dimensions_valid': (n == n_g) and (k == n - m),
        'H_rank': gf2_rank(H),
        'G_rank': gf2_rank(G),
        'H_expected_rank': m,
        'G_expected_rank': k,
        'H_density': np.mean(H),
        'G_density': np.mean(G),
        'code_rate': k / n
    }

    # Orthogonality check
    if results['dimensions_valid']:
        product = (G @ H.T) % 2
        results['orthogonal'] = np.all(product == 0)
        results['orthogonality_violations'] = np.sum(product != 0)
        results['orthogonality_error_rate'] = results['orthogonality_violations'] / product.size
    else:
        results['orthogonal'] = False
        results['orthogonality_violations'] = float('inf')
        results['orthogonality_error_rate'] = 1.0

    # Rank checks
    results['H_full_rank'] = (results['H_rank'] == results['H_expected_rank'])
    results['G_full_rank'] = (results['G_rank'] == results['G_expected_rank'])

    # Overall validity
    results['valid_ldpc_pair'] = (
        results['dimensions_valid'] and
        results['orthogonal'] and
        results['H_full_rank'] and
        results['G_full_rank']
    )

    return results


def print_verification_report(H: np.ndarray, G: np.ndarray, code_name: str = "Unknown"):
    """
    Print comprehensive verification report for LDPC matrix pair.

    Args:
        H: Parity-check matrix
        G: Generator matrix
        code_name: Name/identifier for the code
    """
    results = verify_ldpc_matrices(H, G)

    print(f"\n{'='*60}")
    print(f"LDPC Matrix Verification Report: {code_name}")
    print(f"{'='*60}")

    print(f"Matrix Dimensions:")
    print(f"  H matrix: {H.shape}")
    print(f"  G matrix: {G.shape}")
    print(f"  Code parameters: n={H.shape[1]}, k={G.shape[0]}, m={H.shape[0]}")
    print(f"  Code rate: {results['code_rate']:.3f}")

    print(f"\nMatrix Properties:")
    print(f"  H density: {results['H_density']:.3f}")
    print(f"  G density: {results['G_density']:.3f}")
    print(f"  H rank: {results['H_rank']}/{results['H_expected_rank']} {'PASS' if results['H_full_rank'] else 'FAIL'}")
    print(f"  G rank: {results['G_rank']}/{results['G_expected_rank']} {'PASS' if results['G_full_rank'] else 'FAIL'}")

    print(f"\nOrthogonality Check:")
    print(f"  G × H^T = 0 (mod 2): {'PASSED' if results['orthogonal'] else 'FAILED'}")
    if not results['orthogonal']:
        print(f"  Violations: {results['orthogonality_violations']}/{H.shape[0] * G.shape[0]}")
        print(f"  Error rate: {results['orthogonality_error_rate']:.6f}")

    print(f"\nOverall Assessment:")
    print(f"  Valid LDPC pair: {'YES' if results['valid_ldpc_pair'] else 'NO'}")

    print(f"{'='*60}\n")

    return results
