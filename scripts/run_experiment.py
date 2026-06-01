import argparse
import yaml
from src.experiment.experiment import LDPCExperiment

def main():
    parser = argparse.ArgumentParser(description="Run an LDPC RL experiment.")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="Path to the experiment configuration YAML file."
    )
    parser.add_argument(
        '--output-parent-dir',
        type=str,
        default=None,
        help="Parent directory for results and models (prepended to output_paths)"
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override output paths if parent directory specified
    if args.output_parent_dir:
        from pathlib import Path
        parent = Path(args.output_parent_dir)

        if 'output_paths' in config:
            original_results = config['output_paths'].get('results_dir', 'results/')
            original_models = config['output_paths'].get('models_dir', 'models/')

            # Prepend parent directory to relative paths
            config['output_paths']['results_dir'] = str(parent / original_results)
            config['output_paths']['models_dir'] = str(parent / original_models)

            print(f"Output directory override applied:")
            print(f"   Results: {original_results} → {config['output_paths']['results_dir']}")
            print(f"   Models:  {original_models} → {config['output_paths']['models_dir']}")

    # Create and run the experiment
    experiment = LDPCExperiment(config)
    experiment.run()

if __name__ == "__main__":
    main()
