"""
Agent Registry System

Centralized agent configuration, instantiation, and model loading management.
Handles RL agents with trained weights and baseline agents with proper initialization.
"""

import os
import torch
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Type
from pathlib import Path

from src.core.learning_agent import DQNAgent
from src.core.baseline_agents import RandomAgent, GreedyAgent, SimpleClassicalAgent, MinSumAgent, MinSumExpertAgent
from src.utils.path_utils import resolve_models_path


@dataclass
class AgentSpec:
    """Specification for an agent including configuration and metadata."""
    name: str
    display_name: str
    enabled: bool
    agent_class: Type
    config: Dict[str, Any] = field(default_factory=dict)
    requires_model: bool = False
    model_path: Optional[str] = None


class AgentRegistry:
    """
    Centralized agent configuration and instantiation system.
    
    Handles both RL agents (with model loading) and baseline agents,
    providing a unified interface for agent creation and management.
    """
    
    def __init__(self, experiment_config: Dict[str, Any], env=None, models_dir: str = None):
        """
        Initialize agent registry with experiment configuration.
        
        Args:
            experiment_config: Full experiment configuration dictionary
            env: LDPC environment instance for agent initialization
            models_dir: Directory containing saved models
        """
        self.config = experiment_config
        self.env = env
        self.models_dir = models_dir or resolve_models_path(config=experiment_config)
        self.device = experiment_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Agent class mapping
        self.agent_classes = {
            'rl': DQNAgent,
            'random': RandomAgent,
            'greedy': GreedyAgent,
            'simple_classical': SimpleClassicalAgent,
            'min_sum': MinSumAgent,
            'min_sum_expert': MinSumExpertAgent
        }
    
    def get_agent_specs(self) -> List[AgentSpec]:
        """
        Get all agent specifications from configuration.
        
        Returns:
            List of AgentSpec objects for enabled agents
        """
        specs = []
        
        # Add RL agent if weights were trained in this run OR a pretrained model is configured
        load_pretrained_enabled = self.config.get('load_pretrained_model', {}).get('enabled', False)
        include_rl_agent = self.config.get('run_training', False) or load_pretrained_enabled
        if include_rl_agent:
            rl_spec = self._create_rl_agent_spec()
            specs.append(rl_spec)
        
        # Add baseline agents from configuration
        baseline_specs = self._create_baseline_specs()
        specs.extend(baseline_specs)
        
        return specs
    
    def create_agent(self, spec: AgentSpec) -> Any:
        """
        Create an agent instance from specification.
        
        Args:
            spec: Agent specification
            
        Returns:
            Initialized agent instance
        """
        if spec.name == 'rl':
            return self._create_rl_agent(spec)
        else:
            return self._create_baseline_agent(spec)
    
    def _create_rl_agent_spec(self) -> AgentSpec:
        """Create specification for RL agent with model loading."""
        # Determine model path
        model_path = self._find_rl_model_path()
        
        return AgentSpec(
            name='rl',
            display_name='RL Agent',
            enabled=True,
            agent_class=DQNAgent,
            requires_model=True,
            model_path=model_path,
            config=self.config.get('training', {})
        )
    
    def _create_baseline_specs(self) -> List[AgentSpec]:
        """Create specifications for baseline agents."""
        specs = []
        
        # Get baseline configuration
        eval_config = self.config.get('evaluation', {})
        baseline_configs = eval_config.get('baseline_agents', [])
        
        # Default baseline agents if none configured
        if not baseline_configs:
            baseline_configs = [
                {'name': 'random', 'enabled': False},
                {'name': 'greedy', 'enabled': True}
            ]
        
        for baseline_config in baseline_configs:
            agent_name = baseline_config.get('name')
            enabled = baseline_config.get('enabled', False)
            
            if enabled and agent_name in self.agent_classes:
                spec = AgentSpec(
                    name=agent_name,
                    display_name=self._get_display_name(agent_name),
                    enabled=True,
                    agent_class=self.agent_classes[agent_name],
                    config=baseline_config
                )
                specs.append(spec)
        
        return specs
    
    def _create_rl_agent(self, spec: AgentSpec) -> DQNAgent:
        """Create and initialize RL agent with trained weights."""
        if not self.env:
            raise ValueError("Environment required for RL agent creation")
        
        # Get network configuration
        network_config = spec.config.get('network', {})
        
        # Create DQN agent
        agent = DQNAgent(
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim,
            network_config=network_config,
            device=self.config.get('device'),
            **self._get_rl_agent_params(spec.config)
        )
        
        # Load trained weights - CRITICAL: must exist for valid evaluation
        if spec.model_path and os.path.exists(spec.model_path):
            self._load_model_weights(agent, spec.model_path)
        else:
            # CRITICAL: Model must exist for RL agent evaluation - fail explicitly
            raise FileNotFoundError(
                f"Required RL model not found: {spec.model_path}\n"
                f"RL agent creation requires a trained model. Using randomly initialized weights would invalidate results.\n"
                f"This indicates either: 1) Model saving failed after training, or 2) Invalid evaluation configuration."
            )
        
        return agent
    
    def _create_baseline_agent(self, spec: AgentSpec) -> Any:
        """Create and initialize baseline agent."""
        if not self.env:
            raise ValueError("Environment required for baseline agent creation")
        
        agent_class = spec.agent_class
        
        # Create agent with LDPC code
        try:
            agent = agent_class(code=self.env.code)
            return agent
        except Exception as e:
            raise RuntimeError(f"Failed to create {spec.name} agent: {e}") from e
    
    def _get_rl_agent_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract RL agent parameters from configuration."""
        return {
            'learning_rate': config.get('learning_rate', 0.0001),
            'weight_decay': config.get('weight_decay', 0.0),
            'epsilon_start': 0.0,  # No exploration during evaluation
            'epsilon_decay': 1.0,   # No decay needed during evaluation
            'epsilon_min': 0.0      # Keep at zero for evaluation
        }
    
    def _find_rl_model_path(self) -> Optional[str]:
        """Find the saved RL model from training."""
        # Check if there's a specific model path configured
        pretrained_config = self.config.get('load_pretrained_model', {})
        if pretrained_config.get('enabled', False):
            model_path = pretrained_config.get('model_path')
            if model_path and os.path.exists(model_path):
                return model_path
        
        # Look for models in the models directory
        models_dir = Path(self.models_dir)
        
        # First priority: Look for model with current experiment timestamp
        current_timestamp = self.config.get('current_experiment_timestamp')
        if current_timestamp:
            timestamp_model = models_dir / f'ldpc_rl_agent_{current_timestamp}.pth'
            if timestamp_model.exists():
                print(f"Found current experiment model: {timestamp_model}")
                return str(timestamp_model)
        
        # Common model file patterns
        patterns = [
            'ldpc_rl_agent.pth',
            'best_model.pth',
            'final_model.pth'
        ]
        
        for pattern in patterns:
            model_path = models_dir / pattern
            if model_path.exists():
                return str(model_path)
        
        # Look for any .pth files in models directory
        pth_files = list(models_dir.glob('*.pth'))
        if pth_files:
            # Use the most recently modified model
            latest_model = max(pth_files, key=lambda p: p.stat().st_mtime)
            print(f"Using most recent model (no current experiment model found): {latest_model}")
            return str(latest_model)
        
        return None
    
    def _load_model_weights(self, agent: DQNAgent, model_path: str):
        """Load trained weights into RL agent."""
        try:
            checkpoint = torch.load(model_path, map_location=agent.device, weights_only=True)

            # Warn on device mismatch (informational — map_location handles remapping)
            if isinstance(checkpoint, dict):
                saved_device = None
                for key in ('network_state_dict', 'state_dict'):
                    sd = checkpoint.get(key)
                    if sd:
                        first_param = next(iter(sd.values()), None)
                        if first_param is not None and hasattr(first_param, 'device'):
                            saved_device = first_param.device
                        break
                if saved_device is not None and saved_device != agent.device:
                    print(f"Model saved on {saved_device}, loading onto {agent.device}")

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'network_state_dict' in checkpoint:
                    # Full agent checkpoint format
                    agent.network.load_state_dict(checkpoint['network_state_dict'])
                    agent.target_network.load_state_dict(checkpoint['network_state_dict'])
                elif 'state_dict' in checkpoint:
                    # Network-only checkpoint format
                    agent.network.load_state_dict(checkpoint['state_dict'])
                    agent.target_network.load_state_dict(checkpoint['state_dict'])
                else:
                    # Assume the dict is the state_dict itself
                    agent.network.load_state_dict(checkpoint)
                    agent.target_network.load_state_dict(checkpoint)
            else:
                # Assume it's a direct state_dict
                agent.network.load_state_dict(checkpoint)
                agent.target_network.load_state_dict(checkpoint)
            
            agent.network.eval()  # Set to evaluation mode
            agent.target_network.eval()
            print(f"Loaded RL model weights from {model_path}")
            
        except Exception as e:
            # CRITICAL: Model loading failure should fail the experiment, not silently continue
            raise RuntimeError(
                f"Failed to load RL model weights from {model_path}: {e}\n"
                f"Model loading failures indicate corrupted files, dimension mismatches, or other critical issues.\n"
                f"Using randomly initialized weights would invalidate evaluation results."
            ) from e
    
    def _get_display_name(self, agent_name: str) -> str:
        """Get human-readable display name for agent."""
        display_names = {
            'rl': 'RL Agent',
            'random': 'Random Agent',
            'greedy': 'Greedy Agent',
            'simple_classical': 'Simple Classical Agent',
            'min_sum': 'Min-Sum Agent',
            'min_sum_expert': 'Min-Sum Expert Agent'
        }
        return display_names.get(agent_name, agent_name.title() + ' Agent')
    
    def get_enabled_agent_names(self) -> List[str]:
        """Get list of enabled agent names."""
        specs = self.get_agent_specs()
        return [spec.name for spec in specs if spec.enabled]
    
    def validate_configuration(self) -> List[str]:
        """
        Validate agent configuration and return any warnings.
        
        Returns:
            List of warning messages
        """
        warnings = []
        
        # Check if weight-modifying steps were run but no model found
        weight_modifying_steps = self.config.get('run_training', False)
        if weight_modifying_steps:
            rl_spec = self._create_rl_agent_spec()
            if not rl_spec.model_path or not os.path.exists(rl_spec.model_path):
                step_names = []
                if self.config.get('run_training', False):
                    step_names.append("training")
                warnings.append(f"Weight-modifying steps ({', '.join(step_names)}) were enabled but no trained model found")
        
        # Check if any baseline agents are enabled
        baseline_specs = self._create_baseline_specs()
        if not baseline_specs:
            warnings.append("No baseline agents enabled for comparison")
        
        return warnings
