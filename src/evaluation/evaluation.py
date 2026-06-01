"""
Evaluation Framework for the RL LDPC Decoder

This module provides an abstract BaseEvaluator class and a concrete LDPCEvaluator
for measuring the performance of trained RL agents against baselines.

Updated to use agent-centric progressive saving with the modular metrics collection framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
from src.core.baseline_agents import RandomAgent, GreedyAgent, SimpleClassicalAgent, MinSumAgent
from src.data.robust_data_generator import RobustDataGenerator
from src.evaluation.episode_runner import EpisodeRunner
from src.evaluation.metrics_callbacks import EpisodeDataCollector
from src.evaluation.progressive_saver import EvaluationProgressiveSaver
from src.evaluation.agent_registry import AgentRegistry
from src.evaluation.episode_context import agent_evaluation_context

class BaseEvaluator(ABC):
    """
    Abstract base class for agent evaluation.
    """
    
    @abstractmethod
    def evaluate(self, agent, **kwargs) -> Dict[str, Any]:
        """
        Evaluate an agent and return performance metrics.
        
        Args:
            agent: The agent to evaluate
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary containing evaluation metrics
        """
        pass


class LDPCEvaluator(BaseEvaluator):
    """
    Handles the evaluation of the RL agent and comparison against baselines using RobustDataGenerator.
    
    Updated to use agent-centric progressive saving with modular metrics collection framework.
    """
    def __init__(self, env, data_generator: RobustDataGenerator, collectors: list = None, 
                 config: dict = None, base_filename: str = None, output_dir: str = "results"):
        self.env = env
        self.data_generator = data_generator
        self.config = config or {}
        
        # Validate that test split exists
        if hasattr(data_generator, 'h5_file') and data_generator.h5_file is not None:
            if 'splits' not in data_generator.h5_file:
                raise ValueError("RobustDataGenerator must have train/test split created. Call create_split() first.")
        else:
            raise ValueError("RobustDataGenerator must be loaded from HDF5 file with valid splits.")
        
        # Initialize the evaluation runner with default collectors if none provided
        if collectors is None:
            collectors = [EpisodeDataCollector()]
        
        # Convert list to dict for easier management
        if isinstance(collectors, list):
            self.collectors = {collector.__class__.__name__: collector for collector in collectors}
        else:
            self.collectors = collectors
        
        self.episode_runner = EpisodeRunner(env, list(self.collectors.values()))
        
        # Initialize progressive saver
        self.progressive_saver = EvaluationProgressiveSaver(
            base_filename=base_filename or "evaluation",
            output_dir=output_dir
        )
        
        # Initialize agent registry
        # Use proper path resolution instead of brittle string replacement
        from src.utils.path_utils import resolve_models_path
        models_dir = resolve_models_path(config=self.config)
        
        self.agent_registry = AgentRegistry(
            experiment_config=self.config,
            env=self.env,
            models_dir=str(models_dir)
        )
        
        # Storage for saved files
        self.saved_files = []

    def run_progressive_evaluation(self) -> List[str]:
        """
        Run evaluation using agent-centric progressive saving.
        
        Evaluates each agent separately and saves results progressively,
        preventing memory overflow and data loss.
        
        Returns:
            List of saved file paths
        """
        print("\nStarting Agent-Centric Progressive Evaluation")
        
        # Get all agent specifications
        agent_specs = self.agent_registry.get_agent_specs()
        
        if not agent_specs:
            raise ValueError(
                "No agents configured for evaluation. "
                "Enable RL evaluation (run_training=true or load_pretrained_model.enabled=true) "
                "and/or enable at least one baseline agent."
            )
        
        print(f"Found {len(agent_specs)} agents to evaluate:")
        for spec in agent_specs:
            print(f"   • {spec.display_name} ({spec.name})")
        
        # Evaluate each agent progressively
        for spec in agent_specs:
            print(f"\nEvaluating {spec.display_name}...")
            
            try:
                # Create agent instance
                agent = self.agent_registry.create_agent(spec)
                
                # Evaluate single agent with progressive saving
                saved_file = self._evaluate_single_agent_progressive(agent, spec)
                self.saved_files.append(saved_file)
                
                print(f"{spec.display_name} evaluation completed: {saved_file}")
                
            except Exception as e:
                print(f"Failed to evaluate {spec.display_name}: {e}")
                continue

        print(f"\nProgressive evaluation completed! Saved {len(self.saved_files)} files:")
        for file_path in self.saved_files:
            print(f"   {file_path}")
        
        return self.saved_files
    
    def _evaluate_single_agent_progressive(self, agent, agent_spec) -> str:
        """
        Evaluate a single agent with progressive saving.
        
        Args:
            agent: Agent instance to evaluate
            agent_spec: Agent specification object
            
        Returns:
            Path to saved file
        """
        # Start progressive saving for this agent
        flush_callback = self.progressive_saver.start_agent_evaluation(
            agent_spec.name, self.collectors
        )
        
        # Get test data iterator
        test_iterator = self._get_test_iterator()
        
        # Evaluate agent with episode context management
        with agent_evaluation_context(agent_spec.name, self.collectors) as ctx:
            episode_count = 0
            
            with torch.no_grad():
                for test_case in test_iterator:
                    # Run single episode using episode runner (automatically collects via metrics_manager)
                    episode = self.episode_runner.run_episode(
                        agent=agent,
                        original_codeword=test_case['original'],
                        noisy_codeword=test_case['noisy'],
                        error_info={'error_pattern': test_case['error'], 'meta': test_case['meta']},
                        episode_metadata={'agent_type': agent_spec.name, 
                                          'phase': test_case.get('phase', 1), 
                                          'episode_type': 'eval'}
                    )
                    
                    # Episode already collected by episode_runner - no manual collection needed
                    episode_count += 1
                    
                    # Progress reporting
                    if episode_count % 10000 == 0:
                        print(f"   Processed {episode_count} episodes...")

            print(f"   Total episodes: {episode_count}")
        
        # End progressive saving and get saved file path
        saved_file = self.progressive_saver.end_phase()
        
        return saved_file

    def _get_test_iterator(self):
        """
        Returns iterator for the full test dataset from RobustDataGenerator.
        
        Returns:
            Iterator over all test samples
        """
        return self.data_generator.get_iterator('test')

    def evaluate(self, agent, agent_type: str = 'agent') -> List[Dict[str, Any]]:
        """
        Evaluate agent performance on LDPC decoding tasks using the modular metrics framework.
        
        DEPRECATED: Use run_progressive_evaluation() for new implementations.
        This method is maintained for backward compatibility.
        
        Args:
            agent: The agent to evaluate
            agent_type: The type of agent being evaluated
            
        Returns:
            List of dictionaries with raw episode data
        """
        print("Warning: Using deprecated evaluate() method. Consider using run_progressive_evaluation().")
        
        # Get test iterator from RobustDataGenerator
        test_iterator = self._get_test_iterator()
        
        # Use the EpisodeRunner to perform evaluation
        with torch.no_grad():
            self.episode_runner.run_episodes_from_dataset(agent, test_iterator, agent_type=agent_type, episode_type='eval')
        
        return self.episode_runner.get_metrics().get('EpisodeDataCollector', [])

    def compare_performance(self, agent=None) -> Dict[str, Any]:
        """
        Compare agent performance against baselines.
        
        Updated to use progressive evaluation approach. If agent is provided,
        uses legacy mode for backward compatibility.
        
        Args:
            agent: The agent to evaluate (optional for progressive mode)
            
        Returns:
            Comprehensive comparison results or list of saved files for progressive mode
        """
        if agent is not None:
            # Legacy mode for backward compatibility
            print("Using legacy comparison mode. Consider using run_progressive_evaluation().")
            
            print("Evaluating RL Agent...")
            agent_results = self.evaluate(agent, agent_type='rl')
            
            print("Evaluating Baselines...")
            baseline_results = self._evaluate_baselines_legacy()
            
            return {
                'agent_episodes': agent_results,
                'baseline_episodes': baseline_results,
            }
        else:
            # New progressive mode
            return {
                'saved_files': self.run_progressive_evaluation(),
                'format': 'agent_separated'
            }
    
    def _evaluate_baselines_legacy(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Legacy baseline evaluation method for backward compatibility.
        """
        results = {}
        
        # Create baseline agents using agent registry
        agent_specs = self.agent_registry.get_agent_specs()
        baseline_specs = [spec for spec in agent_specs if spec.name != 'rl']
        
        for spec in baseline_specs:
            try:
                baseline_agent = self.agent_registry.create_agent(spec)
                test_iterator = self._get_test_iterator()
                
                # Reset metrics for each baseline evaluation
                self.episode_runner.reset_metrics()
                
                with torch.no_grad():
                    results[spec.name] = self.evaluate(baseline_agent, agent_type=spec.name)
            except Exception as e:
                print(f"Failed to evaluate baseline {spec.name}: {e}")
        
        return results
    
    def get_saved_files(self) -> List[str]:
        """Get list of files saved by progressive evaluation."""
        return self.saved_files.copy()
    
