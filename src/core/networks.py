"""
Neural Network Architectures for LDPC Decoding
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class SimpleMLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) to approximate the Q-function.
    This network takes the entire state vector and processes it through
    a series of fully connected layers.
    """
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim1: int = 256, hidden_dim2: int = 128):
        super(SimpleMLP, self).__init__()
        self.layer1 = nn.Linear(state_dim, hidden_dim1)
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.output_layer = nn.Linear(hidden_dim2, action_dim)

    def forward(self, state):
        """
        Performs a forward pass through the network.

        Args:
            state (torch.Tensor): The input state tensor.

        Returns:
            torch.Tensor: The Q-values for each action.
        """
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        q_values = self.output_layer(x)
        return q_values

class LDPCOptimizedMLP(nn.Module):
    """
    5-layer MLP optimized specifically for LDPC decoding characteristics.
    
    This network is designed based on deep analysis of LDPC problem structure:
    - Deterministic mathematical relationships (not noisy data)
    - Sparse parity check matrix (4.8% density, each action affects ~2.7 syndromes)
    - Mixed-scale state components (binary + continuous)
    - Sequential decision making with precise state transitions
    - Curriculum learning across error complexities
    
    Architecture Philosophy:
    - Progressive abstraction: Input → 640 → 480 → 360 → 270 → (bit-flips + STOP)
    
    Key Design Decisions:
    - Layer normalization (better than batch norm for structured mathematical data)
    - Minimal dropout (0.05) only in middle layers (avoid disrupting deterministic relationships)
    - Leaky ReLU to prevent dead neurons in mathematical problems

    """
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim1: int = 640, hidden_dim2: int = 480, 
                 hidden_dim3: int = 360, hidden_dim4: int = 270,
                 dropout_rate: float = 0.05, leaky_slope: float = 0.01):
        super(LDPCOptimizedMLP, self).__init__()
        
        # Layer 1: Input processing - learn basic bit-syndrome mappings
        self.layer1 = nn.Linear(state_dim, hidden_dim1)
        self.norm1 = nn.LayerNorm(hidden_dim1)
        
        # Layer 2: Local patterns - immediate flip effects (with minimal dropout)
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.norm2 = nn.LayerNorm(hidden_dim2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Layer 3: Multi-hop reasoning - bit → syndrome → other bits (with minimal dropout)
        self.layer3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.norm3 = nn.LayerNorm(hidden_dim3)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Layer 4: Global integration - constraint satisfaction
        self.layer4 = nn.Linear(hidden_dim3, hidden_dim4)
        self.norm4 = nn.LayerNorm(hidden_dim4)
        
        # Output layer: Q-value ranking
        self.output_layer = nn.Linear(hidden_dim4, action_dim)
        
        # Leaky ReLU activation to prevent dead neurons in deterministic problems
        self.activation = nn.LeakyReLU(negative_slope=leaky_slope)
        
    def forward(self, state):
        """
        Forward pass through LDPC-optimized 5-layer architecture.
        
        Args:
            state (torch.Tensor): Input state tensor [batch_size, state_dim]
                State = codeword(n) + syndrome(m) + flip_vector(n) +
                penalty_vector(n) + metadata(3). Dimensions adapt to the LDPC code:
                  24x48: state_dim=171, action_dim=49
                  56x80: state_dim=299, action_dim=81
                  64x128: state_dim=451, action_dim=129
                  128x256: state_dim=899, action_dim=257

        Returns:
            torch.Tensor: Q-values [batch_size, action_dim]
                One output per bit-flip action plus STOP action.
        """
        # Layer 1: Input processing with normalization
        x = self.activation(self.norm1(self.layer1(state)))
        
        # Layer 2: Local patterns with minimal dropout
        x = self.activation(self.norm2(self.layer2(x)))
        x = self.dropout2(x)
        
        # Layer 3: Multi-hop reasoning with minimal dropout
        x = self.activation(self.norm3(self.layer3(x)))
        x = self.dropout3(x)
        
        # Layer 4: Global integration
        x = self.activation(self.norm4(self.layer4(x)))
        
        # Output: Linear layer for Q-values (no activation)
        q_values = self.output_layer(x)
        
        return q_values