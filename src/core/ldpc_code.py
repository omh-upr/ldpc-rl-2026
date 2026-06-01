"""
LDPC Code Class for Centralized Matrix Operations

This module provides a centralized LDPC_Code class that encapsulates all properties
and operations related to a specific LDPC code, eliminating hardcoded dimensions
throughout the codebase.
"""

import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
from scipy import sparse
from src.core.ldpc_utils import load_matrices, parse_alist_file, compute_generator_matrix


class LDPC_Code:
    """
    Represents a specific LDPC code, defined by its H and G matrices.
    
    This class centralizes all properties and fundamental operations related to an LDPC code,
    acting as a single source of truth for the rest of the system.
    """
    
    def __init__(self, H: np.ndarray, G: np.ndarray, code_id: Optional[str] = None):
        """
        Initializes the LDPC code instance.
        
        Args:
            H (np.ndarray): The parity-check matrix
            G (np.ndarray): The generator matrix
            code_id (str): Code identifier for metadata (e.g., '64x128')
            
        Raises:
            ValueError: If matrix dimensions or properties are invalid
        """
        if H.ndim != 2 or G.ndim != 2:
            raise ValueError("H and G must be 2D matrices.")
        
        self._H = H.astype(np.int32)
        self._G = G.astype(np.int32)
        self._code_id = code_id
        
        # Key dimensions
        self._m, self._n = H.shape
        self._k = G.shape[0]
        
        # Pre-compute for efficiency
        self._H_T = self._H.T

        # NEW: Sparse matrix representations for efficient operations
        self._H_csr = sparse.csr_matrix(self._H)  # Efficient for H @ codeword (row access)
        self._H_csc = self._H_csr.tocsc()          # Efficient for H.T @ syndrome (column access)
                    
        # Validate properties on initialization
        self._validate_properties()

    def _validate_properties(self):
        """Ensures the provided matrices are valid and compatible."""
        if self._n != self._G.shape[1]:
            raise ValueError(f"Dimension mismatch: H has {self._n} columns, G has {self._G.shape[1]} columns.")
        
        # Verify orthogonality: G * H^T = 0 (mod 2)
        orthogonality_check = (self._G @ self._H_T) % 2
        if not np.all(orthogonality_check == 0):
            raise ValueError("Matrices G and H are not orthogonal (G @ H.T % 2 != 0).")
        
        print(f"LDPC Code initialized and validated successfully.")
        print(f"   Matrix dimensions: H={self._H.shape}, G={self._G.shape}")
        print(f"   Code parameters: n={self._n}, k={self._k}, m={self._m}")
        print(f"   Sparse density: {self._H_csr.nnz / (self._m * self._n):.4f} ({self._H_csr.nnz} non-zeros)") # NEW

    # --- Public Properties ---

    @property
    def n(self) -> int:
        """Returns the codeword length (number of variable nodes)."""
        return self._n

    @property
    def k(self) -> int:
        """Returns the message length."""
        return self._k

    @property
    def m(self) -> int:
        """Returns the number of check nodes."""
        return self._m

    @property
    def H(self) -> np.ndarray:
        """Returns the parity-check matrix."""
        return self._H

    @property
    def G(self) -> np.ndarray:
        """Returns the generator matrix."""
        return self._G

    @property
    def state_dimensions(self) -> Dict[str, int]:
        """Provides dimensions for constructing the RL state vector."""
        return {
            'codeword': self.n,
            'syndrome': self.m,
            'flip_vector': self.n,
            'penalty_vector': self.n,
            'metadata': 3,
            'total': self.n + self.m + self.n + self.n + 3
        }

    @property
    def action_dim(self) -> int:
        """Returns the dimension of the action space (n bit flips + 1 STOP)."""
        return self.n + 1

    @property
    def code_id(self) -> Optional[str]:
        """Returns the code identifier."""
        return self._code_id

    @property
    def correction_capability(self) -> int:
        """Returns the theoretical error correction capability (t = floor((d_min - 1) / 2))."""
        from src.core.ldpc_utils import get_ldpc
        if not getattr(self, "_code_id", None):
            raise ValueError(
                "correction_capability is unavailable because this LDPC_Code instance has no code_id. "
                "Use LDPC_Code.from_default(code_id=...), or load/save with metadata containing code_id."
            )
        try:
            d_min = get_ldpc(self._code_id)["d_min"]
        except Exception as e:
            raise ValueError(
                f"Unknown code_id '{self._code_id}' for correction_capability lookup."
            ) from e
        return (d_min - 1) // 2

    @property
    def state_component_ranges(self) -> Dict[str, tuple]:
        """Returns start and end indices for each component in the state vector."""
        dims = self.state_dimensions
        ranges = {}
        offset = 0
        
        for component in ['codeword', 'syndrome', 'flip_vector', 'penalty_vector']:
            start = offset
            end = offset + dims[component]
            ranges[component] = (start, end)
            offset = end
        
        # Metadata components
        ranges['metadata'] = (offset, offset + dims['metadata'])
        ranges['syndrome_weight'] = (offset, offset + 1)
        ranges['penalty_sum'] = (offset + 1, offset + 2)
        ranges['step_count'] = (offset + 2, offset + 3)
        
        return ranges

    # --- Core Operations ---

    def encode(self, message: np.ndarray) -> np.ndarray:
        """Encodes a message of length k into a codeword of length n."""
        if message.shape[-1] != self.k:
            raise ValueError(f"Message length must be {self.k}, but got {message.shape[-1]}.")
        return (message @ self._G) % 2

    def syndrome(self, codeword: np.ndarray) -> np.ndarray:
        # """Calculates the syndrome for a given codeword."""
        # if codeword.shape[-1] != self.n:
        #     raise ValueError(f"Codeword length must be {self.n}, but got {codeword.shape[-1]}.")
        # return (self._H @ codeword.T).T % 2

        """Calculates the syndrome for a given codeword using sparse matrix."""
        if codeword.shape[-1] != self.n:
            raise ValueError(f"Codeword length must be {self.n}, but got {codeword.shape[-1]}.")
        # Use sparse matrix for ~20-40× speedup
        return (self._H_csr @ codeword.T).T % 2        

    def penalty_vector(self, syndrome: np.ndarray) -> np.ndarray:
        # """Calculates the penalty vector from a syndrome."""
        # if syndrome.shape[-1] != self.m:
        #     raise ValueError(f"Syndrome length must be {self.m}, but got {syndrome.shape[-1]}.")
        # return (self._H_T @ syndrome.T).T  # Remove % 2 to count unsatisfied constraints

        """Calculates the penalty vector from a syndrome using sparse matrix."""
        if syndrome.shape[-1] != self.m:
            raise ValueError(f"Syndrome length must be {self.m}, but got {syndrome.shape[-1]}.")
        # Use sparse CSC format for efficient column access (H.T @ syndrome)
        return (self._H_csc.T @ syndrome.T).T  # Note: CSC.T is efficient

    def is_valid(self, codeword: np.ndarray) -> bool:
        """Checks if a codeword is valid (syndrome is all zeros)."""
        s = self.syndrome(codeword)
        return np.all(s == 0)

    @classmethod
    def from_default(cls, code_id='56x80'):
        """Factory method to load the default matrices for the specified code."""
        H, G = load_matrices(code_id=code_id)
        return cls(H, G, code_id=code_id)
    
    @classmethod
    def from_alist(cls, alist_path, code_id: Optional[str] = None):
        """
        Factory method to load LDPC code from an alist file.
        
        Args:
            alist_path (str): Path to the alist file
            code_id (Optional[str]): Optional code identifier for metadata-dependent properties
            
        Returns:
            LDPC_Code: New instance loaded from the alist file
            
        Raises:
            FileNotFoundError: If the alist file doesn't exist
            ValueError: If the file format is invalid or matrices are invalid
        """
        # Parse the alist file to get H matrix
        H = parse_alist_file(alist_path)
        
        # Compute the generator matrix G from H
        G = compute_generator_matrix(H)
        
        # Create LDPC_Code instance with custom validation
        instance = cls.__new__(cls)
        
        # Set matrices directly
        instance._H = H.astype(np.int32)
        instance._G = G.astype(np.int32)
        instance._code_id = code_id
        
        # Key dimensions
        instance._m, instance._n = H.shape
        instance._k = G.shape[0]
        
        # Pre-compute for efficiency
        instance._H_T = instance._H.T

        # NEW: Sparse matrix representations
        instance._H_csr = sparse.csr_matrix(instance._H)
        instance._H_csc = instance._H_csr.tocsc()

        # Skip orthogonality check for alist-loaded matrices
        print(f"LDPC Code initialized from alist file")
        print(f"   Matrix dimensions: H={instance._H.shape}, G={instance._G.shape}")
        print(f"   Code parameters: n={instance._n}, k={instance._k}, m={instance._m}")
        print(f"   Note: Orthogonality check skipped for alist-loaded matrices")
        
        return instance

    def save(self, file_path, format='npy'):
        """
        Save the H and G matrices to files.
        
        Args:
            file_path (str): Base path for saving files (without extension)
            format (str): Format to save in ('npy', 'txt', 'csv')
            
        Raises:
            ValueError: If format is not supported
        """
        if format not in ['npy', 'txt', 'csv']:
            raise ValueError(f"Unsupported format: {format}. Use 'npy', 'txt', or 'csv'")
        
        base_path = Path(file_path)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'npy':
            # Save as numpy binary files
            H_path = base_path.with_suffix('.H.npy')
            G_path = base_path.with_suffix('.G.npy')
            
            np.save(H_path, self._H)
            np.save(G_path, self._G)
            
            print(f"LDPC matrices saved successfully")
            print(f"   H matrix: {H_path}")
            print(f"   G matrix: {G_path}")
            
        elif format == 'txt':
            # Save as plain text files
            H_path = base_path.with_suffix('.H.txt')
            G_path = base_path.with_suffix('.G.txt')
            
            np.savetxt(H_path, self._H, fmt='%d')
            np.savetxt(G_path, self._G, fmt='%d')
            
            print(f"LDPC matrices saved as text files")
            print(f"   H matrix: {H_path}")
            print(f"   G matrix: {G_path}")
            
        elif format == 'csv':
            # Save as CSV files
            H_path = base_path.with_suffix('.H.csv')
            G_path = base_path.with_suffix('.G.csv')
            
            np.savetxt(H_path, self._H, fmt='%d', delimiter=',')
            np.savetxt(G_path, self._G, fmt='%d', delimiter=',')
            
            print(f"LDPC matrices saved as CSV files")
            print(f"   H matrix: {H_path}")
            print(f"   G matrix: {G_path}")
        
        # Save metadata
        metadata = {
            'code_id': self._code_id,
            'n': self.n,
            'k': self.k,
            'm': self.m,
            'rate': self.k / self.n,
            'H_shape': self._H.shape,
            'G_shape': self._G.shape,
            'H_density': float(np.mean(self._H)),
            'G_density': float(np.mean(self._G))
        }
        
        metadata_path = base_path.with_suffix('.metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   Metadata: {metadata_path}")
        print(f"   Code parameters: n={self.n}, k={self.k}, m={self.m}, rate={self.k/self.n:.3f}")

    @classmethod
    def load(cls, file_path, format='npy'):
        """
        Load LDPC code from saved files.
        
        Args:
            file_path (str): Base path for loading files (without extension)
            format (str): Format to load from ('npy', 'txt', 'csv')
            
        Returns:
            LDPC_Code: New instance loaded from saved files
            
        Raises:
            FileNotFoundError: If the files don't exist
            ValueError: If format is not supported
        """
        if format not in ['npy', 'txt', 'csv']:
            raise ValueError(f"Unsupported format: {format}. Use 'npy', 'txt', or 'csv'")
        
        base_path = Path(file_path)
        
        if format == 'npy':
            H_path = base_path.with_suffix('.H.npy')
            G_path = base_path.with_suffix('.G.npy')
            
            if not H_path.exists() or not G_path.exists():
                raise FileNotFoundError(f"Matrix files not found: {H_path} or {G_path}")
            
            H = np.load(H_path)
            G = np.load(G_path)
            
        elif format == 'txt':
            H_path = base_path.with_suffix('.H.txt')
            G_path = base_path.with_suffix('.G.txt')
            
            if not H_path.exists() or not G_path.exists():
                raise FileNotFoundError(f"Matrix files not found: {H_path} or {G_path}")
            
            H = np.loadtxt(H_path, dtype=np.int32)
            G = np.loadtxt(G_path, dtype=np.int32)
            
        elif format == 'csv':
            H_path = base_path.with_suffix('.H.csv')
            G_path = base_path.with_suffix('.G.csv')
            
            if not H_path.exists() or not G_path.exists():
                raise FileNotFoundError(f"Matrix files not found: {H_path} or {G_path}")
            
            H = np.loadtxt(H_path, dtype=np.int32, delimiter=',')
            G = np.loadtxt(G_path, dtype=np.int32, delimiter=',')
        
        metadata = None
        # Load metadata if available
        metadata_path = base_path.with_suffix('.metadata.json')
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"LDPC code loaded from saved files")
            print(f"   Loaded metadata: n={metadata['n']}, k={metadata['k']}, m={metadata['m']}")
        else:
            print(f"LDPC code loaded from saved files (no metadata found)")
        
        # Create instance using the same approach as from_alist to skip orthogonality check
        instance = cls.__new__(cls)
        
        # Set matrices directly
        instance._H = H.astype(np.int32)
        instance._G = G.astype(np.int32)
        instance._code_id = metadata.get('code_id') if metadata else None
        
        # Key dimensions
        instance._m, instance._n = H.shape
        instance._k = G.shape[0]
        
        # Pre-compute for efficiency
        instance._H_T = instance._H.T

        # NEW: Sparse matrix representations
        instance._H_csr = sparse.csr_matrix(instance._H)
        instance._H_csc = instance._H_csr.tocsc()        
        
        print(f"   Matrix dimensions: H={instance._H.shape}, G={instance._G.shape}")
        print(f"   Code parameters: n={instance._n}, k={instance._k}, m={instance._m}")
        
        return instance

    def __repr__(self):
        return f"LDPC_Code(n={self.n}, k={self.k}, m={self.m}, rate={self.k/self.n:.3f})"
