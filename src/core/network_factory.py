"""
Network Factory for LDPC Decoding Agents

This module provides a factory function for creating neural network architectures
with automatic parameter mapping and constructor signature introspection.
"""
import inspect
from typing import Dict, Any, Optional
from src.core.networks import SimpleMLP, LDPCOptimizedMLP


def create_network(network_type: str,
                   network_config: Dict[str, Any],
                   state_dim: int,
                   action_dim: int):
    """
    Factory function that creates neural networks with automatic parameter mapping.

    This function uses introspection to automatically pass parameters from the
    network configuration to the appropriate constructor, handling different
    signature patterns transparently.

    Args:
        network_type: Name of the network class ('SimpleMLP', 'LDPCOptimizedMLP')
        network_config: Dictionary of network-specific parameters from YAML
        state_dim: Total dimensionality of the state space
        action_dim: Dimensionality of the action space

    Returns:
        Instantiated neural network

    Raises:
        ValueError: If network_type is unknown or required parameters are missing
    """
    # Registry of available network classes
    network_classes = {
        'SimpleMLP': SimpleMLP,
        'LDPCOptimizedMLP': LDPCOptimizedMLP
    }

    if network_type not in network_classes:
        raise ValueError(f"Unknown network type: {network_type}. "
                        f"Available types: {list(network_classes.keys())}")

    network_class = network_classes[network_type]

    # Handle standard MLP networks with flexible parameters
    params = {
        'state_dim': state_dim,
        'action_dim': action_dim
    }

    # Add network-specific parameters from configuration
    # These will be filtered by signature introspection
    params.update(network_config)

    # Use signature introspection to filter valid parameters
    sig = inspect.signature(network_class.__init__)
    valid_params = {k: v for k, v in params.items() if k in sig.parameters}

    # Remove 'self' parameter if accidentally included
    valid_params.pop('self', None)

    return network_class(**valid_params)


def get_available_networks() -> Dict[str, str]:
    """
    Returns a dictionary of available network types and their descriptions.

    Returns:
        Dict mapping network type names to descriptions
    """
    return {
        'SimpleMLP': 'Standard 2-layer MLP with configurable hidden dimensions',
        'LDPCOptimizedMLP': '5-layer MLP optimized for LDPC mathematical structure'
    }


def validate_network_config(network_type: str, network_config: Dict[str, Any]) -> bool:
    """
    Validates that the network configuration contains reasonable parameters.

    Args:
        network_type: Name of the network class
        network_config: Dictionary of network parameters

    Returns:
        True if configuration is valid

    Raises:
        ValueError: If configuration contains invalid parameters
    """
    # Check for reasonable hidden dimension values
    for key, value in network_config.items():
        if key.startswith('hidden_dim') and isinstance(value, int):
            if value <= 0:
                raise ValueError(f"Hidden dimension {key} must be positive, got {value}")
            if value > 2048:
                print(f"Warning: Large hidden dimension {key}={value} may cause memory issues")

    return True
