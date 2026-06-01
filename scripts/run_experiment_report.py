#!/usr/bin/env python3
"""
Automated Experiment Report Generator

This script generates a comprehensive analysis report for an LDPC experiment
by automatically discovering analyses from template placeholders and executing them.

Template-Driven Execution:
    Only analyses referenced in the template (e.g., {{analysis_id}}) are executed.
    This provides maximum flexibility and performance optimization.

Metadata Inclusion:
    By default, analysis titles and descriptions are automatically included from
    analysis_definitions.json. Use --no-metadata to exclude them.

Usage:
    # Default: include analysis metadata (title + description)
    PYTHONPATH=. python scripts/run_experiment_report.py --experiment-id 20251018_123456

    # Use custom template (only runs analyses in template)
    PYTHONPATH=. python scripts/run_experiment_report.py --experiment-id 20251018_123456 --template custom_template.md

    # Exclude metadata (data only)
    PYTHONPATH=. python scripts/run_experiment_report.py --experiment-id 20251018_123456 --no-metadata
"""

import argparse
import subprocess
import os
from pathlib import Path
from src.utils.path_utils import resolve_project_path


class ReportGenerator:
    """
    Generates a comprehensive experiment report by running multiple analyses
    and populating a Markdown template with the results.

    Analyses are automatically discovered from template placeholders (e.g., {{analysis_id}}).
    Only the analyses referenced in the template will be executed.
    """

    def __init__(
        self,
        experiment_id: str,
        template_path: str,
        include_metadata: bool = True,
        results_dir: str = "results",
    ):
        """
        Initialize the report generator.

        Args:
            experiment_id: Timestamp identifier of the experiment (e.g., '20251018_123456')
            template_path: Path to the Markdown template file
            include_metadata: If True, prepend title and description to analysis results (default: True)
            results_dir: Directory containing experiment outputs and where report/DB are stored
        """
        self.experiment_id = experiment_id
        self.template_path = template_path
        self.include_metadata = include_metadata
        self.results_dir = resolve_project_path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.report_path = str(self.results_dir / f"{self.experiment_id}_report.md")

    def _extract_analysis_ids_from_template(self) -> list[str]:
        """
        Parse template file and extract all {{placeholder}} analysis IDs.

        Returns:
            List of unique analysis IDs found in template, preserving order.
            Excludes 'experiment_id' which is a special placeholder.
        """
        import re

        try:
            with open(self.template_path, 'r') as f:
                template_content = f.read()

            # Find all {{placeholder}} patterns (valid Python identifiers)
            pattern = r'\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}'
            matches = re.findall(pattern, template_content)

            # Remove duplicates while preserving order, exclude 'experiment_id'
            seen = set()
            analysis_ids = []
            for match in matches:
                if match != 'experiment_id' and match not in seen:
                    seen.add(match)
                    analysis_ids.append(match)

            return analysis_ids

        except FileNotFoundError:
            print(f"Warning: Template not found at {self.template_path}")
            return []

    def _get_analysis_metadata(self, analysis_id: str) -> str:
        """
        Load title and description from analysis_definitions.json.

        Args:
            analysis_id: ID of the analysis to get metadata for

        Returns:
            Markdown-formatted header with title and description, or just the analysis_id if not found
        """
        import json

        definitions_path = Path("config/analysis_definitions.json")

        try:
            with open(definitions_path, 'r') as f:
                definitions = json.load(f)

            # Find matching analysis
            for analysis in definitions:
                if analysis['id'] == analysis_id:
                    title = analysis.get('title', analysis_id)
                    description = analysis.get('description', '')
                    return f"**{title}**\n\n{description}"

            # Fallback if not found
            return f"**{analysis_id}**"

        except Exception as e:
            # Graceful degradation if file not found or parse error
            print(f"Warning: Could not load metadata for '{analysis_id}': {e}")
            return f"**{analysis_id}**"

    def _run_analysis(self, analysis_id: str, rebuild_db: bool = False) -> str:
        """
        Run a single analysis and return markdown-formatted output.

        Args:
            analysis_id: ID of the analysis to run
            rebuild_db: Whether to rebuild the database (use True for first analysis)

        Returns:
            Markdown-formatted analysis results

        Raises:
            RuntimeError: If the analysis command fails
        """
        db_path = str(self.results_dir / f"{self.experiment_id}.db")
        cmd = [
            "python", "scripts/run_analysis.py",
            "--experiment_id", self.experiment_id,
            "--analysis_id", analysis_id,
            "--database-path", db_path,
            "--results-dir", str(self.results_dir),
            "--data-only", "--to-markdown"
        ]
        if rebuild_db:
            cmd.append("--rebuild-database")

        env = os.environ.copy()
        env["PYTHONPATH"] = "."

        result = subprocess.run(cmd, capture_output=True, text=True, env=env)

        if result.returncode != 0:
            raise RuntimeError(f"Analysis '{analysis_id}' failed:\n{result.stderr}")

        # Extract only the markdown table from stdout (skip progress messages)
        lines = result.stdout.strip().split('\n')
        # Find the "--- Analysis Results ---" marker and extract everything after it
        marker_found = False
        output_lines = []
        for line in lines:
            if "--- Analysis Results ---" in line:
                marker_found = True
                continue
            if marker_found:
                output_lines.append(line)

        table_content = '\n'.join(output_lines).strip() if output_lines else result.stdout.strip()

        # Optionally prepend metadata (title and description)
        if self.include_metadata:
            metadata_header = self._get_analysis_metadata(analysis_id)
            return f"{metadata_header}\n\n{table_content}"
        else:
            return table_content

    def generate_report(self):
        """
        Generate the complete experiment report.

        This method:
        1. Extracts required analyses from template placeholders
        2. Runs only the analyses referenced in the template
        3. Collects results in a dictionary
        4. Renders the template with analysis results
        5. Writes the final report to disk
        """
        # Extract required analyses from template
        required_analyses = self._extract_analysis_ids_from_template()

        if not required_analyses:
            print("No analysis placeholders found in template. Nothing to do.")
            print(f"    Template: {self.template_path}")
            print(f"    Expected format: {{{{analysis_id}}}}")
            return

        print(f"Starting report generation for experiment: {self.experiment_id}")
        print(f"Found {len(required_analyses)} analyses in template...\n")

        # Check if database already exists
        db_path = self.results_dir / f"{self.experiment_id}.db"
        if db_path.exists():
            print(f"Using existing database: {db_path}")
            rebuild_db_flag = False
        else:
            print(f"Database will be created on first analysis")
            rebuild_db_flag = True

        analysis_results = {}

        for i, analysis_id in enumerate(required_analyses, 1):
            print(f"[{i}/{len(required_analyses)}] Running analysis: {analysis_id}...", end=" ")
            try:
                output = self._run_analysis(analysis_id, rebuild_db=rebuild_db_flag)
                rebuild_db_flag = False  # Only rebuild on first analysis
                analysis_results[analysis_id] = output
                print("OK")
            except RuntimeError as e:
                print("FAILED")
                print(f"    Error: {str(e)}")
                analysis_results[analysis_id] = f"**ANALYSIS FAILED**\n\n```\n{e}\n```"

        print(f"\nRendering template and writing report...")
        self._render_and_write_report(analysis_results)
        print(f"Successfully generated report: {self.report_path}")

    def _render_and_write_report(self, results: dict):
        """
        Read the template, replace placeholders, and write the final report.

        Args:
            results: Dictionary mapping analysis_id to markdown-formatted results
        """
        try:
            with open(self.template_path, "r") as f:
                template_content = f.read()
        except FileNotFoundError:
            print(f"Error: Template file not found at {self.template_path}")
            print(f"    Please ensure the template exists or specify a valid path with --template")
            return

        # Add experiment_id for use as a placeholder in the template
        results['experiment_id'] = self.experiment_id

        # Replace all placeholders with actual results
        rendered_content = template_content
        for key, value in results.items():
            placeholder = f"{{{{{key}}}}}"  # e.g., {{overall_performance_summary}}
            rendered_content = rendered_content.replace(placeholder, value)

        # Ensure results directory exists
        Path(self.report_path).parent.mkdir(parents=True, exist_ok=True)

        # Write the final report
        with open(self.report_path, "w") as f:
            f.write(rendered_content)


def main():
    """Main entry point for the report generator script."""
    parser = argparse.ArgumentParser(
        description="Generate a comprehensive analysis report for an LDPC experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate report using default template (with metadata)
  PYTHONPATH=. python scripts/run_experiment_report.py --experiment-id 20251018_123456

  # Generate report using custom template (only runs analyses in template)
  PYTHONPATH=. python scripts/run_experiment_report.py --experiment-id 20251018_123456 --template custom_template.md

  # Generate report without analysis metadata (data only)
  PYTHONPATH=. python scripts/run_experiment_report.py --experiment-id 20251018_123456 --no-metadata

  # Use custom results directory (for output_paths.results_dir-aligned workflows)
  PYTHONPATH=. python scripts/run_experiment_report.py --experiment-id 20251018_123456 --results-dir my_results

Output:
  The generated report will be saved to <results_dir>/{experiment_id}_report.md
  (default results_dir: results/)

Template-Driven Execution:
  The script automatically discovers which analyses to run by parsing the template
  for {{placeholder}} patterns. Only referenced analyses are executed, improving
  performance and allowing flexible report customization.

Notes:
  - Analysis titles and descriptions from analysis_definitions.json are automatically
    included by default. Use --no-metadata to disable this.
  - Special placeholder {{experiment_id}} is always available and excluded from analysis execution.
        """
    )
    parser.add_argument(
        "--experiment-id",
        required=True,
        help="The timestamp ID of the experiment (e.g., '20251018_123456')"
    )
    parser.add_argument(
        "--template",
        default="config/report_template.md",
        help="Path to the report template file (default: config/report_template.md)"
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Exclude analysis titles and descriptions from output (default: include them)"
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Results directory containing experiment files and where report/DB should be written (default: results)"
    )
    args = parser.parse_args()

    generator = ReportGenerator(
        args.experiment_id,
        args.template,
        include_metadata=not args.no_metadata,
        results_dir=args.results_dir,
    )
    generator.generate_report()


if __name__ == "__main__":
    main()
