import argparse
import os
from pathlib import Path
from src.analysis.analysis import LDPCAnalysis
from src.utils.path_utils import resolve_config_path, resolve_results_path, resolve_project_path
from src.utils.file_patterns import ExperimentFiles
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Run analysis on LDPC experiment results with database backend.")
    parser.add_argument(
        '--experiment_id',
        required=False,
        type=str,
        help="The timestamp ID of the experiment to analyze (e.g., '20250723_224425')."
    )
    parser.add_argument(
        '--analysis_id',
        required=False,
        type=str,
        help="The ID of the analysis to run (e.g., 'overall_performance_summary')."
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help="List all available analyses and exit."
    )
    parser.add_argument(
        '--category',
        type=str,
        help="Filter analyses by category (e.g., 'training_dynamics', 'behavioral_analysis')."
    )
    parser.add_argument(
        '--data-focus',
        type=str,
        choices=['training', 'evaluation', 'both'],
        help="Filter analyses by data focus (training, evaluation, both)."
    )
    parser.add_argument(
        '--complexity',
        type=str,
        choices=['simple', 'medium', 'complex'],
        help="Filter analyses by complexity level (simple, medium, complex)."
    )
    parser.add_argument(
        '--database-path',
        type=str,
        help="Custom path for database file (default: <results_dir>/{experiment_id}.db)."
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help="Results directory used for file discovery and default DB path (default: project's resolved results directory)."
    )
    parser.add_argument(
        '--rebuild-database',
        action='store_true',
        help="Force recreation of database schema and reload data."
    )
    parser.add_argument(
        '--data-only',
        action='store_true',
        help='Return only tabular data; skip plotting'
    )
    parser.add_argument(
        '--to-markdown',
        action='store_true',
        help='Convert output to markdown format (only when --data-only is used)'
    )

    args = parser.parse_args()

    # Validate --to-markdown flag usage
    if args.to_markdown and not args.data_only:
        print("Warning: --to-markdown requires --data-only; flag will be ignored")
        args.to_markdown = False

    try:
        definitions_path = resolve_config_path('analysis_definitions.json')
        if args.results_dir:
            results_dir = resolve_project_path(args.results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
        else:
            results_dir = resolve_results_path()
        
        # Initialize analyzer with database support
        database_path = None
        if args.experiment_id and not args.list:
            if args.database_path:
                database_path = args.database_path
            else:
                database_path = str(results_dir / f"{args.experiment_id}.db")
        
        analyzer = LDPCAnalysis(str(definitions_path), database_path)
        
    except FileNotFoundError as e:
        print(f"Error: Configuration file not found: {e}")
        print("Please ensure analysis_definitions.json exists in the config/ folder.")
        return

    if args.list:
        analyzer.list_analyses(
            category=args.category,
            data_focus=getattr(args, 'data_focus', None),
            complexity=args.complexity
        )
        return

    if not args.experiment_id or not args.analysis_id:
        parser.error("Both --experiment_id and --analysis_id are required unless --list is specified.")

    # Enhanced file discovery: support single, progressive, and flush training files
    def discover_training_files():
        """Discover training files with support for flush files.

        Uses ExperimentFiles.find_training_files() which properly handles:
        - Single file format (legacy): *_training.json.gz
        - Progressive format: *_training_phase{N}.json.gz
        - Flush files: *_training_phase{N}_flush{M}.json.gz

        Files are sorted by (phase, flush_num) for correct episode ordering.
        """
        training_files = ExperimentFiles.find_training_files(
            args.experiment_id, str(results_dir)
        )

        if not training_files:
            return []

        # Categorize files for informative output
        flush_files = [f for f in training_files if ExperimentFiles.is_flush_file(f)]
        phase_files = [f for f in training_files if not ExperimentFiles.is_flush_file(f)]

        if flush_files:
            print(f"Found {len(training_files)} training files ({len(flush_files)} flush + {len(phase_files)} phase-end):")
        else:
            print(f"Found {len(training_files)} training files:")

        for f in training_files:
            print(f"   - {Path(f).name}")

        return training_files
    
    def discover_evaluation_files():
        """Discover evaluation files with support for agent-separated format."""
        # Use the new file pattern utility
        eval_files_info = ExperimentFiles.find_evaluation_files(
            args.experiment_id, str(results_dir)
        )
        
        if eval_files_info.format_type == "agent_separated":
            print(f"Found {len(eval_files_info.files)} agent-separated evaluation files:")
            for agent_name, file_path in zip(eval_files_info.agent_names, eval_files_info.files):
                print(f"   - {agent_name}: {Path(file_path).name}")
            return eval_files_info.files
        elif eval_files_info.format_type == "legacy":
            print(f"Found legacy evaluation data: {Path(eval_files_info.files[0]).name}")
            return eval_files_info.files[0]
        else:
            print(f"No evaluation data found (training-only experiment)")
            return None
    
    # Check if database needs to be created/rebuilt
    db_path = Path(database_path)
    if args.rebuild_database or not db_path.exists():
        # Only discover files if we need to build/rebuild the database
        training_files = discover_training_files()
        evaluation_files = discover_evaluation_files()

        if not training_files:
            print(f"Error: Could not find training files for experiment {args.experiment_id}")
            print(f"   Tried patterns: *{args.experiment_id}_training.json.gz, *{args.experiment_id}_training_phase*.json.gz")
            print(f"   In directory: {results_dir}")
            return
        print("Creating database schema...")
        analyzer.create_database_schema()
        
        print("Loading experiment data into database...")
        
        # Extract experiment name from first training file
        first_training_file = Path(training_files[0])
        if len(training_files) == 1:
            # Single file format: remove _training suffix
            experiment_name = first_training_file.stem.replace('.json', '').replace(f"_{args.experiment_id}_training", "")
        else:
            # Progressive format: remove _training_phase{N} suffix (handles both phase and flush files)
            experiment_name = first_training_file.stem.replace('.json', '').split(f"_{args.experiment_id}_training_phase")[0]
        
        # Load experiment data with multiple training and evaluation files
        analyzer.load_experiment_data(
            training_files,    # Pass list of training files
            evaluation_files,  # Pass evaluation files (can be None, single file, or list)
            args.experiment_id,
            experiment_name
        )
        
        # Show database statistics
        stats = analyzer.get_database_stats()
        print("\nDatabase Statistics:")
        print(stats.to_string(index=False))
    else:
        print(f"Using existing database: {db_path}")
    
    # Run the requested analysis
    print(f"\nRunning analysis: {args.analysis_id}")
    result = analyzer.run_analysis(args.analysis_id, data_only=args.data_only)

    if isinstance(result, pd.DataFrame):
        print("\n--- Analysis Results ---")
        if args.data_only and args.to_markdown:
            print(result.to_markdown(index=False))
        else:
            print(result.to_string(index=False))
    else:
        print("\n--- Plot Generated ---")
        print("Displaying plot. Close the plot window to exit.")
        result.show()
        plt.show()


if __name__ == "__main__":
    main()
