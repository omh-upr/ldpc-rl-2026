"""
Path utilities for robust project root detection and path resolution.

This module provides utilities to detect the project root and resolve paths
consistently, regardless of the current working directory.
"""

import os
from pathlib import Path
from typing import Optional, Union


def find_project_root(start_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Find the project root directory by looking for characteristic files/directories.
    
    Args:
        start_path: Directory to start searching from. If None, uses current file's location.
        
    Returns:
        Path object pointing to the project root
        
    Raises:
        RuntimeError: If project root cannot be found
    """
    if start_path is None:
        start_path = Path(__file__).parent
    else:
        start_path = Path(start_path)
    
    # Markers that indicate we've found the project root
    project_markers = [
        'config',  # config/ directory
        'src',     # src/ directory  
        'scripts', # scripts/ directory
        'pyproject.toml',  # Python project file
        'setup.py',        # Python setup file
        'CLAUDE.md',       # Project documentation
    ]
    
    current_path = start_path.resolve()
    
    # Search up the directory tree
    while current_path != current_path.parent:  # Stop at filesystem root
        # Check if we have enough markers to confirm this is the project root
        markers_found = []
        for marker in project_markers:
            marker_path = current_path / marker
            if marker_path.exists():
                markers_found.append(marker)
        
        # If we find at least 2 key markers, consider this the project root
        # Prioritize config, src, and scripts as strong indicators
        key_markers = ['config', 'src', 'scripts']
        key_markers_found = [m for m in markers_found if m in key_markers]
        
        if len(key_markers_found) >= 2 or len(markers_found) >= 3:
            return current_path
            
        current_path = current_path.parent
    
    # If we reach here, we couldn't find the project root
    raise RuntimeError(
        f"Could not find project root starting from {start_path}. "
        f"Expected to find directories like 'config/', 'src/', 'scripts/' "
        f"or files like 'pyproject.toml', 'CLAUDE.md'"
    )


def resolve_config_path(filename: str) -> Path:
    """
    Resolve a configuration file path relative to the project's config/ directory.
    
    Args:
        filename: Name of the configuration file (e.g., 'base_experiment.yml')
        
    Returns:
        Path object pointing to the configuration file
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
    """
    project_root = find_project_root()
    config_path = project_root / 'config' / filename
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Project root: {project_root}\n"
            f"Config directory: {project_root / 'config'}"
        )
    
    return config_path


def resolve_project_path(relative_path: Union[str, Path]) -> Path:
    """
    Resolve any path relative to the project root.
    
    Args:
        relative_path: Path relative to project root (e.g., 'data/output.h5')
        
    Returns:
        Path object pointing to the resolved file/directory
    """
    project_root = find_project_root()
    return project_root / relative_path


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure exists
        
    Returns:
        Path object pointing to the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path object pointing to the project root
    """
    return find_project_root()


def resolve_data_path(filename: str) -> Path:
    """
    Resolve a data file path relative to the project's data/ directory.
    
    Args:
        filename: Name of the data file (e.g., 'dataset.h5')
        
    Returns:
        Path object pointing to the data file
    """
    project_root = find_project_root()
    return project_root / 'data' / filename


def resolve_results_path(filename: str = '', config: dict = None) -> Path:
    """
    Resolve a results file path relative to the configured results directory.
    Creates the results directory if it doesn't exist.
    
    Args:
        filename: Name of the results file (optional)
        config: Configuration dictionary containing output_paths (optional)
        
    Returns:
        Path object pointing to the results file or directory
    """
    project_root = find_project_root()
    
    # Use configured path if available, otherwise default to 'results'
    if config and 'output_paths' in config and 'results_dir' in config['output_paths']:
        results_dir = project_root / config['output_paths']['results_dir']
    else:
        results_dir = project_root / 'results'
    
    ensure_directory(results_dir)
    
    if filename:
        return results_dir / filename
    else:
        return results_dir


def resolve_models_path(filename: str = '', config: dict = None) -> Path:
    """
    Resolve a models file path relative to the configured models directory.
    Creates the models directory if it doesn't exist.
    
    Args:
        filename: Name of the model file (optional)
        config: Configuration dictionary containing output_paths (optional)
        
    Returns:
        Path object pointing to the model file or directory
    """
    project_root = find_project_root()
    
    # Use configured path if available, otherwise default to 'models'
    if config and 'output_paths' in config and 'models_dir' in config['output_paths']:
        models_dir = project_root / config['output_paths']['models_dir']
    else:
        models_dir = project_root / 'models'
    
    ensure_directory(models_dir)
    
    if filename:
        return models_dir / filename
    else:
        return models_dir


def resolve_output_paths(config: dict) -> tuple[Path, Path]:
    """
    Resolve both results and models paths from configuration.
    
    Args:
        config: Configuration dictionary containing output_paths
        
    Returns:
        Tuple of (results_path, models_path)
    """
    results_path = resolve_results_path(config=config)
    models_path = resolve_models_path(config=config)
    return results_path, models_path


# Convenience functions for backward compatibility and ease of use
def config_path(filename: str) -> str:
    """Return configuration file path as string."""
    return str(resolve_config_path(filename))


def project_path(relative_path: str) -> str:
    """Return project-relative path as string."""
    return str(resolve_project_path(relative_path))


def data_path(filename: str) -> str:
    """Return data file path as string."""
    return str(resolve_data_path(filename))


def results_path(filename: str = '', config: dict = None) -> str:
    """Return results file path as string."""
    return str(resolve_results_path(filename, config))


def models_path(filename: str = '', config: dict = None) -> str:
    """Return models file path as string."""
    return str(resolve_models_path(filename, config))