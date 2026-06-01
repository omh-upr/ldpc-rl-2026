#!/usr/bin/env python3
"""
Expert Experience Generation Script

Entry point for generating expert experiences using any agent that conforms
to the BaseAgent interface. Uses configuration-driven workflow for maximum
flexibility and modularity.

Usage:
    PYTHONPATH=. python scripts/run_expert_experience.py \
        --config config/experiments_config/64x128/expert_experience_config.yml
"""

import yaml
import argparse
from pathlib import Path

from src.core.ldpc_code import LDPC_Code
from src.core.environment import LDPCEnvironment
from src.data.simple_agent_registry import SimpleAgentRegistry
from src.data.experience_generator import ExperienceGenerator
from src.data.robust_data_generator import RobustDataGenerator
from src.utils.path_utils import resolve_project_path


def load_configuration(config_path: str) -> dict:
    """
    Load and validate configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML configuration: {e}") from e
    
    # Validate required fields
    required_fields = ['code_id', 'agent_type', 'output_file', 'dataset']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required configuration field: {field}")
    
    # Ensure generation target is specified
    if 'num_experiences' not in config and 'num_episodes' not in config:
        raise ValueError("Either 'num_experiences' or 'num_episodes' must be specified")
    
    return config


def setup_components(config: dict):
    """
    Setup LDPC code, environment, agent, and data generator from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (ldpc_code, environment, agent, data_generator)
        
    Raises:
        ValueError: If agent type is not found
        RuntimeError: If component setup fails
    """
    # Setup LDPC Code
    print(f"Setting up LDPC code: {config['code_id']}")
    try:
        ldpc_code = LDPC_Code.from_default(config['code_id'])
        print(f"   Code dimensions: {ldpc_code.n}x{ldpc_code.k}")
    except Exception as e:
        raise RuntimeError(f"Failed to setup LDPC code '{config['code_id']}': {e}") from e
    
    # Setup Environment
    print(f"Setting up environment")
    try:
        env_config = config.get('environment', {})
        environment = LDPCEnvironment(ldpc_code, **env_config)
        print(f"   Environment: action_dim={environment.action_dim}, state_dim={environment.state_dim}")
    except Exception as e:
        raise RuntimeError(f"Failed to setup environment: {e}") from e
    
    # Setup Agent
    print(f"Setting up agent: {config['agent_type']}")
    try:
        registry = SimpleAgentRegistry()
        agent_params = config.get('agent_params', {})
        agent = registry.create_agent(config['agent_type'], ldpc_code, **agent_params)
        print(f"   Agent created successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to setup agent: {e}") from e
    
    # Setup Data Generator
    print(f"Setting up data generator")
    try:
        # Use dataset path from configuration
        dataset_path = resolve_project_path(config['dataset'])
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
        data_generator = RobustDataGenerator.from_h5(str(dataset_path), code=ldpc_code)
        print(f"   Dataset loaded: {dataset_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to setup data generator: {e}") from e
    
    return ldpc_code, environment, agent, data_generator


def run_generation(config: dict, ldpc_code, environment, agent, data_generator):
    """
    Run the experience generation process.
    
    Args:
        config: Configuration dictionary
        ldpc_code: LDPC code instance
        environment: LDPC environment instance
        agent: Agent instance
        data_generator: Data generator instance
        
    Raises:
        RuntimeError: If generation or saving fails
    """
    # Determine buffer size
    num_experiences = config.get('num_experiences')
    num_episodes = config.get('num_episodes')
    
    if num_experiences:
        buffer_size = num_experiences
        target_desc = f"{num_experiences:,} experiences"
    else:
        buffer_size = num_episodes * environment.max_steps  # Conservative estimate
        target_desc = f"{num_episodes:,} episodes"
    
    print(f"Starting generation: {target_desc}")
    print(f"   Buffer capacity: {buffer_size:,}")
    
    # Create generator and run
    try:
        generator = ExperienceGenerator(environment, agent, buffer_size, data_generator)
        total_experiences = generator.generate(
            num_episodes=num_episodes,
            num_experiences=num_experiences
        )
        print(f"Generation complete: {total_experiences:,} experiences collected")
    except Exception as e:
        raise RuntimeError(f"Experience generation failed: {e}") from e
    
    # Save to file
    output_path = resolve_project_path(config['output_file'])
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving to: {output_path}")
    try:
        generator.save(output_path)
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"File saved: {file_size_mb:.1f} MB")
    except Exception as e:
        raise RuntimeError(f"Failed to save experiences: {e}") from e


def main():
    """Main entry point for expert experience generation."""
    parser = argparse.ArgumentParser(
        description="Generate expert experiences using any BaseAgent-compatible agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic usage:
    PYTHONPATH=. python scripts/run_expert_experience.py \\
        --config config/experiments_config/64x128/expert_experience_config.yml
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="Path to expert experience YAML configuration file"
    )

    parser.add_argument(
        '--agent-type',
        type=str,
        default=None,
        help="Override agent_type from config file (e.g., omniscient, min_sum_expert)"
    )

    args = parser.parse_args()
    
    print("Expert Experience Generation Framework")
    print(f"   Configuration: {args.config}")
    
    try:
        # Load configuration
        config = load_configuration(args.config)

        # Override agent type if specified via CLI
        if args.agent_type:
            original_agent = config.get('agent_type', 'unknown')
            config['agent_type'] = args.agent_type
            print(f"Agent type override: {original_agent} -> {args.agent_type}")
            print(f"WARNING: output_file path in config may not reflect actual agent type")

        print("Configuration loaded and validated")

        # Setup components
        ldpc_code, environment, agent, data_generator = setup_components(config)
        print("Components setup complete")
        
        # Run generation
        run_generation(config, ldpc_code, environment, agent, data_generator)
        print("Expert experience generation successful!")
        
    except Exception as e:
        print(f"Expert experience generation failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()