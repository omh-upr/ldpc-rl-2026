import argparse
import os
import sys
from src.analysis.analysis import LDPCAnalysis
from src.analysis.experiment_loader import open_experiment
from src.utils.path_utils import resolve_config_path, resolve_results_path, resolve_project_path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    """
    Run one analysis and return the process exit status.

    Nonzero on every path that produces no analysis. `run_experiment_report.py`
    runs this as a subprocess and decides success by return code, so a failure
    that exits 0 becomes an analysis result: it embedded the error message in the
    report and called the report a success.
    """
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
        
    except FileNotFoundError as e:
        print(f"Error: Configuration file not found: {e}", file=sys.stderr)
        print("Please ensure analysis_definitions.json exists in the config/ folder.",
              file=sys.stderr)
        return 1

    if args.list:
        # Listing reads only the definitions, so it needs no database at all.
        LDPCAnalysis(str(definitions_path), None).list_analyses(
            category=args.category,
            data_focus=getattr(args, 'data_focus', None),
            complexity=args.complexity
        )
        return 0

    if not args.experiment_id or not args.analysis_id:
        parser.error("Both --experiment_id and --analysis_id are required unless --list is specified.")

    # One database per experiment unless the caller names one. A named database
    # is only reused for the run it already holds; open_experiment refuses the
    # rest rather than answering with another run's episodes.
    database_path = args.database_path or str(results_dir / f"{args.experiment_id}.db")

    try:
        analyzer = open_experiment(definitions_path, database_path,
                                   args.experiment_id, results_dir,
                                   rebuild=args.rebuild_database)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
