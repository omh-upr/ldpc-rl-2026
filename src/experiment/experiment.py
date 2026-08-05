import abc
import yaml
import time
import os
import gc
import torch
import numpy as np
import random
from typing import Dict, Any
from src.utils.path_utils import resolve_project_path, resolve_config_path, resolve_results_path, resolve_models_path

# Import necessary project components
from src.core.ldpc_code import LDPC_Code
from src.core.environment import LDPCEnvironment
from src.core.learning_agent import DQNAgent
from src.training.replay_buffer import ReplayBuffer
from src.training.training import LDPCTrainer
from src.evaluation.evaluation import LDPCEvaluator
from src.data.robust_data_generator import RobustDataGenerator
from src.evaluation.metrics_manager import MetricsManager
from src.data.create_dataset import create_ldpc_dataset
from src.evaluation.episode_runner import EpisodeRunner
from src.evaluation.metrics_callbacks import (
    EpisodeDataCollector
)
from src.evaluation.progressive_saver import TrainingFlushSaver

class ExperimentBase(abc.ABC):
    """Abstract base class for an experiment."""
    def __init__(self, config: dict):
        self.config = config
        self.results = {}
        self._setup_seed()

    def _setup_seed(self):
        seed = self.config.get('seed', 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @abc.abstractmethod
    def run(self):
        """Execute the full experiment pipeline."""
        pass

    @abc.abstractmethod
    def _save_results(self):
        """Save all experiment artifacts."""
        pass


class LDPCExperiment(ExperimentBase):
    """
    Orchestrates a complete LDPC decoding experiment, from dataset creation
    to evaluation, based on a YAML configuration.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.ldpc_code = None
        self.env = None
        self.agent = None
        self.data_generator = None
        self.replay_buffer = None
        self.weights_modified = False  # Track if model weights were changed during experiment
        self.loaded_model_path = None  # Store path of loaded model for potential overwrite
        self.model_saved_after_training = False  # Track if model was saved immediately after training
        
        # Validate and normalize memory management configuration
        self._validate_memory_management_config()
        
        # Resolve output directories using configured paths
        results_dir = resolve_results_path(config=self.config)
        models_dir = resolve_models_path(config=self.config)
        self.metrics_manager = MetricsManager(
            experiment_name=self.config.get('experiment_name', 'ldpc_experiment'),
            output_dir=str(results_dir),
            models_dir=str(models_dir)
        )

    def run(self):
        """Executes the experiment pipeline based on the config."""
        start_time = time.time()
        print(f"Starting experiment: {self.metrics_manager.experiment_name}")

        if self.config.get('run_dataset_creation', False):
            self._create_dataset()

        self._setup_components()


        if self.config.get('run_training', False):
            self._train_agent()

        if self.config.get('run_evaluation', False):
            self._evaluate_agent()

        self._save_results()
        
        total_time = time.time() - start_time
        print(f"\nExperiment finished in {total_time / 60:.1f} minutes.")

    def _create_dataset(self):
        """Handles the dataset creation step."""
        print("\n--- 1. Dataset Creation ---")
        cfg = self.config['dataset_creation']
        
        # Resolve paths using path_utils
        config_path = resolve_project_path(cfg['config_path'])
        output_path = resolve_project_path(cfg['output_path'])
        
        if output_path.exists() and not cfg.get('force_recreate', False):
            print(f"Dataset already exists at {output_path}. Skipping creation.")
            return

        code_id = self.config.get('code_id', '56x80')  # Get code_id from config
        success = create_ldpc_dataset(
            config_path=str(config_path),
            output_path=str(output_path),
            test_size=cfg['test_size'],
            code_id=code_id
        )
        if not success:
            raise RuntimeError("Dataset creation failed.")
        print("Dataset creation complete.")


    def _setup_components(self):
        """Initializes shared components like the environment and agent."""
        print("\n--- 2. Initializing Core Components ---")
        code_id = self.config.get('code_id', '56x80')  # Default to '56x80' if not specified
        self.ldpc_code = LDPC_Code.from_default(code_id=code_id)
        self.env = LDPCEnvironment(code=self.ldpc_code, **self.config['environment'])
        
        agent_cfg = self.config['training']
        network_config = agent_cfg.get('network', {'type': 'SimpleMLP'})  # Default to SimpleMLP
        
        # Prepare agent parameters
        agent_params = {
            'state_dim': self.env.state_dim,
            'action_dim': self.env.action_dim,
            'learning_rate': agent_cfg['learning_rate'],
            'weight_decay': agent_cfg.get('weight_decay', 0.0),
            'epsilon_decay': agent_cfg['epsilon_decay'],
            'network_config': network_config,
            'device': self.config.get('device'),
            # Fail fast when an explicitly requested CUDA device is unavailable,
            # rather than silently completing a full run on CPU.
            'strict_device': self.config.get('strict_device', True)
        }
        
        # Add epsilon_start and epsilon_min if configured
        if 'epsilon_start' in agent_cfg:
            # Use first value if it's a list, otherwise use the single value
            epsilon_start = agent_cfg['epsilon_start']
            if isinstance(epsilon_start, (list, tuple)):
                agent_params['epsilon_start'] = epsilon_start[0] if epsilon_start else 1.0
            else:
                agent_params['epsilon_start'] = epsilon_start
        
        if 'epsilon_min' in agent_cfg:
            # Use first value if it's a list, otherwise use the single value
            epsilon_min = agent_cfg['epsilon_min']
            if isinstance(epsilon_min, (list, tuple)):
                agent_params['epsilon_min'] = epsilon_min[0] if epsilon_min else 0.05
            else:
                agent_params['epsilon_min'] = epsilon_min
        
        # Add epsilon_decay_type if configured
        if 'epsilon_decay_type' in agent_cfg:
            # Use first value if it's a list, otherwise use the single value
            epsilon_decay_type = agent_cfg['epsilon_decay_type']
            if isinstance(epsilon_decay_type, (list, tuple)):
                agent_params['epsilon_decay_type'] = epsilon_decay_type[0] if epsilon_decay_type else 'exponential'
            else:
                agent_params['epsilon_decay_type'] = epsilon_decay_type
        
        self.agent = DQNAgent(**agent_params)
        
        # Check if pregenerated expert experiences should be loaded
        replay_file = None
        expert_cfg = self.config.get('expert_experience', {})
        if expert_cfg.get('use_pregenerated_file', False):
            pregenerated_file = expert_cfg.get('pregenerated_file')
            if pregenerated_file:
                # Resolve path relative to project root
                replay_file_path = resolve_project_path(pregenerated_file)
                if replay_file_path.exists():
                    replay_file = str(replay_file_path)
                    print(f"Will load pregenerated expert experiences from: {replay_file}")
                else:
                    raise FileNotFoundError(
                        f"Pregenerated expert experience file not found: {replay_file_path}\n"
                        f"Set use_pregenerated_file=false to proceed without expert experiences, "
                        f"or generate the file using scripts/run_expert_experience.py"
                    )
        
        self.replay_buffer = ReplayBuffer(
            capacity=agent_cfg['replay_buffer_size'], 
            replay_file=replay_file,
            code=self.ldpc_code
        )
        
        print(f"Components initialized. Device: {self.agent.device}")
        
        # Load pre-trained model if configured
        self._load_pretrained_model()

        # Resolve dataset path using path_utils
        dataset_path = resolve_project_path(self.config['dataset_creation']['output_path'])
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please enable dataset_creation.")
        
        self.data_generator = RobustDataGenerator.from_h5(str(dataset_path), code=self.ldpc_code)
        if 'splits' not in self.data_generator.h5_file:
            self.data_generator.create_split(
                test_size=self.config['dataset_creation']['test_size'], 
                stratify=True, shuffle=True, seed=self.config['seed']
            )
        print(f"Dataset loaded from {dataset_path}")

    def _load_pretrained_model(self):
        """Load a pre-trained model if configured."""
        load_config = self.config.get('load_pretrained_model', {})
        
        if not load_config.get('enabled', False):
            return
            
        model_path = load_config.get('model_path')
        if not model_path:
            raise ValueError("load_pretrained_model.model_path must be specified when enabled=true")
        
        # Resolve model path using path_utils
        resolved_path = resolve_project_path(model_path)
        
        if not resolved_path.exists():
            raise FileNotFoundError(f"Pre-trained model not found at {resolved_path}")
        
        try:
            print(f"Loading pre-trained model from {resolved_path}")
            checkpoint = torch.load(str(resolved_path), map_location=self.agent.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'network_state_dict' in checkpoint:
                # Full agent checkpoint format (from DQNAgent.save())
                self.agent.load(str(resolved_path))
                print("Loaded full agent checkpoint (network + optimizer + training state)")
                
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                # Network-only checkpoint format (from MetricsManager.save_model())
                self.agent.network.load_state_dict(checkpoint['state_dict'])
                # Update target network to match
                self.agent.target_network.load_state_dict(checkpoint['state_dict'])
                metadata = checkpoint.get('metadata', {})
                print(f"Loaded network weights from checkpoint. Metadata: {metadata}")
                
            else:
                # Direct state_dict format
                self.agent.network.load_state_dict(checkpoint)
                self.agent.target_network.load_state_dict(checkpoint)
                print("Loaded network weights (direct state_dict format)")
            
            self.loaded_model_path = str(resolved_path)  # Store for potential overwrite
            
        except Exception as e:
            raise RuntimeError(f"Failed to load pre-trained model: {e}")


    def _train_agent(self):
        """Handles the main RL training step, potentially with a pre-filled buffer."""
        print("\n--- 3. Agent Training ---")
        if not self.config.get('run_training', False):
            print("Skipping training step as per configuration.")
            return
            
        cfg = self.config['training']
        training_collectors = self._create_collectors()

        # Pass memory management config to trainer
        memory_config = self.config.get('memory_management', {})

        # Get training_flush_saver (created by _create_collectors if progressive saving is enabled)
        training_flush_saver = getattr(self, 'training_flush_saver', None)

        trainer = LDPCTrainer(
            self.env, self.agent, self.replay_buffer, self.data_generator,
            collectors=training_collectors,
            memory_config=memory_config,
            training_flush_saver=training_flush_saver,
            **cfg
        )
        
        # Pass config and metrics_manager for incremental saving and memory management
        training_summary = trainer.train(
            num_epochs=cfg['num_epochs'],
            config=self.config,
            external_metrics_manager=self.metrics_manager
        )
        
        # After training completes, the data has been saved incrementally per phase
        # Only need to get remaining collector results (which should be minimal after cleanup)
        remaining_results = trainer.metrics_manager.get_collector('EpisodeDataCollector').get_results()
        self.results['training_episodes'] = remaining_results
        
        # Additional memory cleanup after training
        print("Performing post-training memory cleanup...")
        del trainer  # Remove trainer reference
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("   - CUDA cache cleared")
        print(f"   - Memory cleanup completed")
        
        self.weights_modified = True  # Mark that model weights have been changed
        
        # CRITICAL: Save model immediately after training for evaluation to use
        print("Saving trained model immediately after training...")
        model_path = self.metrics_manager.save_model(self.agent.network.state_dict(), {'architecture': 'DQN', 'post_training': True})
        print(f"Trained model saved to {model_path}")
        
        # Mark that we've already saved the timestamped model
        self.model_saved_after_training = True
        
        print("Training complete.")


    def _evaluate_agent(self):
        """Handles the final evaluation step with agent-centric progressive saving."""
        print("\n--- 4. Final Evaluation ---")
        
        # CRITICAL: Validate that a model exists before evaluation
        expected_model_path = self.metrics_manager.models_dir / f"ldpc_rl_agent_{self.metrics_manager.timestamp}.pth"
        
        # Check if weight-modifying steps ran
        weight_steps = []
        if self.config.get('run_training', False):
            weight_steps.append("training")
        
        if weight_steps:
            # Weight-modifying steps ran, model should exist
            if not expected_model_path.exists():
                raise FileNotFoundError(
                    f"Expected model not found: {expected_model_path}\n"
                    f"Weight-modifying steps ({', '.join(weight_steps)}) were enabled but model file is missing. "
                    f"This indicates a model saving failure."
                )
            print(f"Verified model exists: {expected_model_path}")
        else:
            # No weight-modifying steps ran, check if pre-trained model should be used
            load_config = self.config.get('load_pretrained_model', {})
            if load_config.get('enabled', False):
                print(f"Using pre-trained model as specified in configuration")
            else:
                # No model training and no pre-trained model = invalid configuration
                raise ValueError(
                    f"Invalid configuration: Evaluation enabled but no trained model available.\n"
                    f"Either enable training OR set load_pretrained_model.enabled=true.\n"
                    f"Current config: run_training={self.config.get('run_training', False)}, "
                    f"load_pretrained_model.enabled={load_config.get('enabled', False)}"
                )
        
        # Create collectors for evaluation with evaluation-specific memory settings
        evaluation_collectors = self._create_collectors(context='evaluation')
        
        # Create evaluator with progressive saving support and current experiment timestamp
        base_filename = f"{self.metrics_manager.experiment_name}_{self.metrics_manager.timestamp}_evaluation"
        
        # Pass current experiment timestamp to evaluator for correct model loading
        config_with_timestamp = self.config.copy()
        config_with_timestamp['current_experiment_timestamp'] = self.metrics_manager.timestamp
        
        evaluator = LDPCEvaluator(
            env=self.env,
            data_generator=self.data_generator,
            collectors=evaluation_collectors,
            config=config_with_timestamp,
            base_filename=base_filename,
            output_dir=str(self.metrics_manager.results_dir)
        )
        
        num_test_cases = self.data_generator.h5_file['splits'].attrs['num_test_samples']
        print(f"Evaluating on {num_test_cases} test cases...")
        
        # Run progressive evaluation (agent-by-agent)
        saved_files = evaluator.run_progressive_evaluation()
        
        # Store results in new format for compatibility
        self.results['evaluation_episodes'] = {
            'format': 'agent_separated',
            'saved_files': saved_files
        }
        
        # Mark that evaluation data is externally saved (don't create legacy files)
        self.results['evaluation_saved_externally'] = True
        
        print("Evaluation complete with progressive saving.")

    def _save_results(self):
        """Saves model, results, and config with intelligent overwrite logic."""
        print("\n--- 5. Saving Results & Model ---")
        
        # Check if we're using the new agent-separated evaluation format
        evaluation_data = self.results.get('evaluation_episodes', [])
        evaluation_saved_externally = self.results.get('evaluation_saved_externally', False)
        
        if evaluation_saved_externally:
            print("Evaluation data already saved via agent-level progressive saving")
            if isinstance(evaluation_data, dict) and 'saved_files' in evaluation_data:
                print(f"Agent evaluation files: {len(evaluation_data['saved_files'])}")
                for file_path in evaluation_data['saved_files']:
                    print(f"   {file_path}")
        
        # Save training and evaluation results to separate files
        training_data = self.results.get('training_episodes', [])
        saved_files = []
        
        # Check progressive saving settings (handle both old and new config structures)
        memory_config = self.config.get('memory_management', {})
        if 'training' in memory_config:
            # New structure
            enable_progressive = memory_config.get('training', {}).get('enable_progressive_saving', False)
        else:
            # Legacy structure
            enable_progressive = memory_config.get('enable_progressive_saving', False)
        
        if training_data:
            # Count episode types for detailed logging
            train_episodes = sum(1 for ep in training_data if ep.get('episode_type') == 'train')
            val_episodes = sum(1 for ep in training_data if ep.get('episode_type') == 'val')
            eval_episodes = sum(1 for ep in training_data if ep.get('episode_type') == 'eval')
            other_episodes = len(training_data) - train_episodes - val_episodes - eval_episodes            

            print(f"Found {len(training_data)} episodes in training_data memory:")
            print(f"   Training episodes: {train_episodes}")
            print(f"   Validation episodes: {val_episodes}")
            print(f"   Evaluation episodes: {eval_episodes}")
            if other_episodes > 0:
                print(f"   Other/unknown episodes: {other_episodes}")
            
            if not enable_progressive:
                print(f"Progressive saving is DISABLED - creating single training.json.gz file")
                training_path = self.metrics_manager.save_training_results(self.config, training_data)
                saved_files.append(f"Training: {training_path}")
                print(f"Training results saved to {training_path}")
            else:
                print(f"Progressive saving is ENABLED - skipping training.json.gz creation")
                print(f"   Training data should already be saved in separate phase files")
                if eval_episodes > 0:
                    print(f"   Note: Found {eval_episodes} evaluation episodes in training data - this indicates a data flow issue")
        else:
            print(f"No training data found in memory")
            if enable_progressive:
                print(f"   This is expected with progressive saving (data saved per phase)")
            else:
                print(f"   This is unusual for single-file saving - check if training actually occurred")
        
        # Save evaluation results if available and not already saved externally
        if evaluation_data and not evaluation_saved_externally:
            evaluation_path = self.metrics_manager.save_evaluation_results(self.config, evaluation_data)
            saved_files.append(f"Evaluation: {evaluation_path}")
            print(f"Evaluation results saved to {evaluation_path}")
        elif evaluation_saved_externally:
            print("Skipping legacy evaluation save (already saved via progressive system)")
        
        if saved_files:
            print(f"Results saved to {len(saved_files)} separate files")
        else:
            print("No results data to save")
        
        # Handle model saving based on weight modification and overwrite settings
        self._save_model_intelligently()

    def _save_model_intelligently(self):
        """Save model based on weight modification status and overwrite configuration."""
        if not self.weights_modified:
            print("No weight-modifying steps occurred - skipping model save")
            return
            
        load_config = self.config.get('load_pretrained_model', {})
        should_overwrite = load_config.get('overwrite', False) and self.loaded_model_path
        
        if should_overwrite:
            # Overwrite the original loaded model (this is separate from timestamped models)
            print(f"Overwriting loaded model at {self.loaded_model_path}")
            self.agent.save(self.loaded_model_path)
            print(f"Model overwritten at {self.loaded_model_path}")
        elif not self.model_saved_after_training:
            # Only create timestamped model if we haven't already saved it after training
            network_type = self.config.get('training', {}).get('network', {}).get('type', 'SimpleMLP')
            model_path = self.metrics_manager.save_model(self.agent.network.state_dict(), {'architecture': network_type})
            if self.loaded_model_path:
                print(f"New checkpoint saved to {model_path} (original preserved at {self.loaded_model_path})")
            else:
                print(f"Model saved to {model_path}")
        else:
            print("Timestamped model already saved after training - skipping duplicate save")

    def _create_collectors(self, context: str = 'training'):
        """Create metrics collectors with memory management from config.

        Args:
            context: Either 'training' or 'evaluation' to use appropriate memory settings
        """
        memory_config = self.config.get('memory_management', {})

        # Get context-specific memory settings
        if context in memory_config:
            context_config = memory_config[context]
            if context == 'training':
                max_episodes = context_config.get('max_episodes_per_phase', None)
            else:  # evaluation
                max_episodes = context_config.get('max_episodes_per_agent', None)
            flush_frequency = context_config.get('auto_flush_frequency', 1000)
            enable_progressive = context_config.get('enable_progressive_saving', True)
        else:
            # Fallback to legacy flat structure
            max_episodes = memory_config.get('max_episodes_per_phase', None)
            flush_frequency = memory_config.get('auto_flush_frequency', 1000)
            enable_progressive = memory_config.get('enable_progressive_saving', True)

        # For training context with progressive saving enabled, create flush callback
        # to save data immediately to disk when memory limits are hit
        flush_callback = None
        if context == 'training' and enable_progressive and max_episodes:
            self.training_flush_saver = TrainingFlushSaver(
                config=self.config,
                metrics_manager=self.metrics_manager
            )
            flush_callback = self.training_flush_saver.create_flush_callback()
            print(f"   Training flush callback: enabled (immediate disk write)")
        else:
            # Ensure training_flush_saver is None for evaluation or disabled contexts
            if context == 'training':
                self.training_flush_saver = None

        # Create EpisodeDataCollector with memory management settings and flush callback
        episode_collector = EpisodeDataCollector(
            max_episodes=max_episodes,
            memory_check_frequency=flush_frequency,
            flush_callback=flush_callback
        )

        collectors = [episode_collector]

        print(f"Enabled collectors: {[c.name for c in collectors]}")
        if max_episodes:
            if context == 'training':
                print(f"   Memory limit: {max_episodes:,} episodes per phase")
            else:
                print(f"   Memory limit: {max_episodes:,} episodes per agent")
            print(f"   Check frequency: every {flush_frequency:,} episodes")
        else:
            print(f"   Memory limit: unlimited ({context}-based saving only)")
        print(f"   Progressive saving: {'enabled' if enable_progressive else 'disabled'}")

        return collectors


    def _validate_memory_management_config(self):
        """
        Validate and normalize memory management configuration.
        Supports both legacy flat structure and new training/evaluation split.
        """
        # Ensure memory_management section exists
        if 'memory_management' not in self.config:
            print("Warning: No memory_management section found in config, using defaults")
            self.config['memory_management'] = {}
        
        memory_config = self.config['memory_management']
        
        # Check if using new structure (training/evaluation split)
        has_training_config = 'training' in memory_config
        has_evaluation_config = 'evaluation' in memory_config
        
        if has_training_config or has_evaluation_config:
            # New structure - validate both sections
            self._validate_context_memory_config('training', memory_config)
            self._validate_context_memory_config('evaluation', memory_config)
        else:
            # Legacy structure - migrate to new format
            print("Detected legacy memory configuration, migrating to new format...")
            self._migrate_legacy_memory_config(memory_config)
    
    def _validate_context_memory_config(self, context: str, memory_config: Dict[str, Any]):
        """Validate memory configuration for a specific context (training/evaluation)."""
        if context not in memory_config:
            memory_config[context] = {}
        
        context_config = memory_config[context]
        
        # Set context-specific defaults
        if context == 'training':
            defaults = {
                'max_episodes_per_phase': 5000,
                'auto_flush_frequency': 2500,
                'enable_progressive_saving': True
            }
        else:  # evaluation
            defaults = {
                'max_episodes_per_agent': 15000,
                'auto_flush_frequency': 5000,
                'enable_progressive_saving': True
            }
        
        # Apply defaults
        for key, default_value in defaults.items():
            if key not in context_config:
                context_config[key] = default_value
        
        # Validate consistency
        if context == 'training':
            max_episodes = context_config['max_episodes_per_phase']
        else:
            max_episodes = context_config['max_episodes_per_agent']
        
        flush_frequency = context_config['auto_flush_frequency']
        
        if flush_frequency > max_episodes and max_episodes > 0:
            print(f"Warning ({context}): auto_flush_frequency ({flush_frequency}) > max_episodes ({max_episodes})")
        
        # Log configuration
        print(f"Memory Management - {context.title()}:")
        if context == 'training':
            print(f"   Max episodes per phase: {max_episodes:,}")
        else:
            print(f"   Max episodes per agent: {max_episodes:,}")
        print(f"   Auto-flush frequency: {flush_frequency:,}")
        print(f"   Progressive saving: {'enabled' if context_config['enable_progressive_saving'] else 'disabled'}")
    
    def _migrate_legacy_memory_config(self, memory_config: Dict[str, Any]):
        """Migrate legacy flat memory configuration to new training/evaluation structure."""
        legacy_values = {
            'max_episodes_per_phase': memory_config.get('max_episodes_per_phase', 5000),
            'auto_flush_frequency': memory_config.get('auto_flush_frequency', 2500),
            'enable_progressive_saving': memory_config.get('enable_progressive_saving', True)
        }
        
        # Create new structure
        memory_config['training'] = {
            'max_episodes_per_phase': legacy_values['max_episodes_per_phase'],
            'auto_flush_frequency': legacy_values['auto_flush_frequency'],
            'enable_progressive_saving': legacy_values['enable_progressive_saving']
        }
        
        memory_config['evaluation'] = {
            'max_episodes_per_agent': 15000,  # Higher default for evaluation
            'auto_flush_frequency': 5000,     # Less frequent checks
            'enable_progressive_saving': True  # Always enable for evaluation
        }
        
        # Remove legacy keys
        for key in ['max_episodes_per_phase', 'auto_flush_frequency', 'enable_progressive_saving']:
            memory_config.pop(key, None)
        
        print("Memory configuration migrated to new training/evaluation structure")
