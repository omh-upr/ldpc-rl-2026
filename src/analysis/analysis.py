import duckdb
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gzip
import os
from pathlib import Path
from typing import Optional, List, Dict, Union
import tempfile
import csv

class LDPCAnalysis:
    """
    Database-first analysis system for LDPC experiment results.
    Transforms compressed JSON files into normalized DuckDB tables for efficient querying.
    """
    
    def __init__(self, definitions_path: str, database_path: Optional[str] = None):
        """
        Initialize the analysis engine with database-first approach.
        
        Args:
            definitions_path: Path to JSON file with analysis definitions
            database_path: Optional path for persistent database file
        """
        with open(definitions_path, 'r') as f:
            self.definitions = json.load(f)
        
        # Use persistent database if path provided, otherwise in-memory
        self.database_path = database_path
        self.con = duckdb.connect(database=database_path or ':memory:')
        
        print(f"LDPCAnalysis initialized with {'persistent' if database_path else 'in-memory'} database")
    
    def _load_evaluation_data(self, evaluation_json_paths: Union[str, Path, List[Union[str, Path]]]) -> List[Dict]:
        """
        Load evaluation data from single file or multiple agent-separated files.
        
        Args:
            evaluation_json_paths: Single file path or list of agent file paths
            
        Returns:
            List of evaluation episodes
        """
        from src.utils.file_patterns import ExperimentFiles
        
        # Normalize to list for unified processing
        if isinstance(evaluation_json_paths, (str, Path)):
            evaluation_json_paths = [evaluation_json_paths]
        
        evaluation_episodes = []
        
        # Determine if this is agent-separated format
        is_agent_separated = len(evaluation_json_paths) > 1 or any(
            ExperimentFiles.extract_agent_name(str(path)) is not None 
            for path in evaluation_json_paths
        )
        
        if is_agent_separated:
            print(f"Loading {len(evaluation_json_paths)} agent-separated evaluation files:")
            
            for eval_path in evaluation_json_paths:
                agent_name = ExperimentFiles.extract_agent_name(str(eval_path))
                print(f"   - Loading {agent_name or 'unknown'}: {Path(eval_path).name}")
                
                try:
                    episodes = self._load_single_evaluation_file(eval_path, agent_name)
                    evaluation_episodes.extend(episodes)
                    print(f"     Loaded {len(episodes)} episodes")
                    
                except Exception as e:
                    print(f"     Error loading {Path(eval_path).name}: {e}")
                    continue
        else:
            # Single legacy file
            eval_path = evaluation_json_paths[0]
            print(f"Loading legacy evaluation file: {Path(eval_path).name}")
            
            try:
                episodes = self._load_single_evaluation_file(eval_path)
                evaluation_episodes.extend(episodes)
                print(f"   Loaded {len(episodes)} evaluation episodes")
                
            except Exception as e:
                print(f"   Error loading evaluation file: {e}")
                print(f"   Continuing without evaluation data...")
        
        return evaluation_episodes
    
    def _load_single_evaluation_file(self, file_path: Union[str, Path], agent_name: str = None) -> List[Dict]:
        """
        Load evaluation episodes from a single file.
        
        Args:
            file_path: Path to evaluation file
            agent_name: Agent name to add to episodes (for agent-separated format)
            
        Returns:
            List of evaluation episodes
        """
        with gzip.open(file_path, 'rt') as f:
            evaluation_data = json.load(f)
        
        episodes = []
        
        # Handle different data structures
        if 'episodes' in evaluation_data:
            # New agent-separated format
            episodes = evaluation_data['episodes']
        elif 'evaluation_episodes' in evaluation_data:
            # Legacy format with nested structure
            eval_data = evaluation_data['evaluation_episodes']
            if isinstance(eval_data, dict):
                for category, category_episodes in eval_data.items():
                    if isinstance(category_episodes, list):
                        episodes.extend(category_episodes)
                    elif isinstance(category_episodes, dict):
                        for baseline_name, baseline_episodes in category_episodes.items():
                            if isinstance(baseline_episodes, list):
                                episodes.extend(baseline_episodes)
            elif isinstance(eval_data, list):
                episodes = eval_data
        
        # Add agent context if this is agent-separated format
        if agent_name:
            for episode in episodes:
                episode.setdefault('agent_context', {})['agent_name'] = agent_name
        
        return episodes
    
    def create_database_schema(self):
        """
        Create the normalized database schema for LDPC experiment data.
        """
        # Drop existing tables and sequences if they exist
        tables = ['steps', 'episodes', 'experiments']
        sequences = ['steps_seq', 'episodes_seq']
        
        for table in tables:
            self.con.execute(f"DROP TABLE IF EXISTS {table}")
        for seq in sequences:
            self.con.execute(f"DROP SEQUENCE IF EXISTS {seq}")
        
        # Create sequences for auto-increment behavior
        self.con.execute("CREATE SEQUENCE episodes_seq START 1")
        self.con.execute("CREATE SEQUENCE steps_seq START 1")
        
        # Create experiments table
        self.con.execute("""
            CREATE TABLE experiments (
                experiment_id VARCHAR PRIMARY KEY,
                experiment_name VARCHAR,
                configuration JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create episodes table with explicit primary key (no auto-increment)
        self.con.execute("""
            CREATE TABLE episodes (
                global_episode_id BIGINT PRIMARY KEY,
                experiment_id VARCHAR REFERENCES experiments(experiment_id),
                episode_type VARCHAR,
                original_episode_id INTEGER,
                agent_type VARCHAR,
                phase INTEGER,
                original_codeword INTEGER[],
                noisy_codeword INTEGER[],
                error_pattern INTEGER[],
                error_model VARCHAR,
                num_errors INTEGER,
                error_parameters VARCHAR,
                terminal_reason VARCHAR,
                episode_length INTEGER,
                success BOOLEAN
            )
        """)
        
        # Create steps table with compressed state representation
        self.con.execute("""
            CREATE TABLE steps (
                step_id BIGINT PRIMARY KEY DEFAULT nextval('steps_seq'),
                global_episode_id BIGINT REFERENCES episodes(global_episode_id),
                step_number INTEGER,
                action INTEGER,
                reward REAL,
                done BOOLEAN,
                success BOOLEAN,
                -- Compressed state (before action)
                codeword_weight INTEGER,
                syndrome_weight INTEGER,
                flip_weight INTEGER,
                penalty_sum INTEGER,
                step_count INTEGER,
                -- Compressed next_state (after action)
                next_codeword_weight INTEGER,
                next_syndrome_weight INTEGER,
                next_flip_weight INTEGER,
                next_penalty_sum INTEGER,
                next_step_count INTEGER,
                -- Training update (nested within step)
                training_loss REAL,
                training_epsilon REAL,
                q_values_mean REAL,
                q_values_std REAL,
                q_values_min REAL,
                q_values_max REAL,
                target_q_values_mean REAL,
                target_q_values_std REAL,
                target_q_values_min REAL,
                target_q_values_max REAL
            )
        """)
        
        print("Database schema created successfully (indexes will be created after data loading)")
    
    def _create_indexes(self):
        """Create indexes for common query patterns."""
        indexes = [
            "CREATE INDEX idx_episodes_agent_type ON episodes(agent_type)",
            "CREATE INDEX idx_episodes_phase ON episodes(phase)", 
            "CREATE INDEX idx_episodes_success ON episodes(success)",
            "CREATE INDEX idx_episodes_type ON episodes(episode_type)",
            "CREATE INDEX idx_steps_global_episode_id ON steps(global_episode_id)",
            "CREATE INDEX idx_episodes_agent_success ON episodes(agent_type, success)",
            "CREATE INDEX idx_episodes_phase_agent ON episodes(phase, agent_type)",
            "CREATE INDEX idx_episodes_type_agent ON episodes(episode_type, agent_type)",
            "CREATE INDEX idx_steps_training_loss ON steps(training_loss)",
            "CREATE INDEX idx_steps_syndrome_weight ON steps(syndrome_weight)"
        ]
        
        for index_sql in indexes:
            self.con.execute(index_sql)
    
    def load_experiment_data(self, training_json_paths: Union[str, Path, List[Union[str, Path]]], 
                           evaluation_json_paths: Optional[Union[str, Path, List[Union[str, Path]]]], 
                           experiment_id: str, experiment_name: str = None):
        """
        Load and transform experiment data from training and evaluation files into database.
        Supports both legacy single files and new agent-separated format.
        Uses a high-performance bulk loading approach for step data.
        
        Args:
            training_json_paths: Single file path (str) or list of file paths for progressive training data
            evaluation_json_paths: Single file path, list of agent files, or None if not available
            experiment_id: Unique experiment identifier
            experiment_name: Human-readable experiment name
        """
        print(f"Loading experiment {experiment_id}...")
        
        # Normalize training_json_paths to list for unified processing
        if isinstance(training_json_paths, (str, Path)):
            training_json_paths = [training_json_paths]
        
        # Load and merge training data from multiple files
        all_training_episodes = []
        latest_config = None
        
        for i, training_path in enumerate(training_json_paths):
            print(f"Loading training file {i+1}/{len(training_json_paths)}: {Path(training_path).name}")
            
            try:
                with gzip.open(training_path, 'rt') as f:
                    training_data = json.load(f)
                
                # Collect episodes from this phase
                phase_episodes = training_data.get('training_episodes', [])
                if not isinstance(phase_episodes, list):
                    print(f"   Warning: training_episodes is not a list in {Path(training_path).name}, skipping")
                    continue
                    
                all_training_episodes.extend(phase_episodes)
                
                # Keep the latest configuration (from last phase file)
                if 'experiment_configuration' in training_data:
                    latest_config = training_data['experiment_configuration']
                
                print(f"   Loaded {len(phase_episodes)} episodes from phase")
                
            except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
                print(f"   Error loading {Path(training_path).name}: {e}")
                print(f"   Skipping this training file and continuing...")
                continue
            except Exception as e:
                print(f"   Unexpected error loading {Path(training_path).name}: {e}")
                print(f"   Skipping this training file and continuing...")
                continue
        
        # Load evaluation data (optional) - supports both single file and agent-separated formats
        evaluation_episodes = []
        if evaluation_json_paths is not None:
            evaluation_episodes = self._load_evaluation_data(evaluation_json_paths)
        else:
            print("   No evaluation data to load")
            
        print(f"Total loaded: {len(all_training_episodes)} training episodes, {len(evaluation_episodes)} evaluation episodes")
        
        # Validate we have some data to process
        if len(all_training_episodes) == 0 and len(evaluation_episodes) == 0:
            raise ValueError(f"No valid episode data found for experiment {experiment_id}. Cannot proceed with database creation.")
        
        if len(all_training_episodes) == 0:
            print("   Warning: No training episodes found - this is unusual for most experiments")
        
        # Insert experiment metadata using latest configuration
        self._insert_experiment_metadata(
            experiment_id, 
            experiment_name or experiment_id,
            latest_config or {}
        )
        
        # Combine all episodes for unified processing
        all_episodes = all_training_episodes + evaluation_episodes
        
        # Use a temporary directory for intermediate CSV files
        with tempfile.TemporaryDirectory() as temp_dir:
            episodes_csv_path = Path(temp_dir) / "episodes.csv"
            steps_csv_path = Path(temp_dir) / "steps.csv"
            
            # 1. Transform and write all episode data to CSV
            all_global_ids = self._write_episodes_to_csv(all_episodes, experiment_id, episodes_csv_path)
            
            # 2. Bulk load episodes from CSV using DuckDB's fast COPY command
            print(f"Bulk loading episodes from {episodes_csv_path}...")
            episodes_columns = "(global_episode_id, experiment_id, episode_type, original_episode_id, agent_type, phase, original_codeword, noisy_codeword, error_pattern, error_model, num_errors, error_parameters, terminal_reason, episode_length, success)"
            self.con.execute(f"COPY episodes {episodes_columns} FROM '{episodes_csv_path}' (FORMAT CSV, HEADER TRUE, AUTO_DETECT FALSE, QUOTE '\"', ESCAPE '\"')")
            print("Episodes bulk load complete.")

            # Verify episode count after bulk load
            actual_stats = self.con.execute(
                "SELECT MIN(global_episode_id), MAX(global_episode_id), COUNT(*) FROM episodes WHERE experiment_id = ?",
                [experiment_id]
            ).fetchone()

            if actual_stats[2] != len(all_global_ids):
                print(f"WARNING: Database episode count ({actual_stats[2]}) != predicted count ({len(all_global_ids)})")

            # 3. Transform and write all step data to CSV
            self._write_steps_to_csv(all_episodes, all_global_ids, steps_csv_path)
            
            # 4. Bulk load steps from CSV using DuckDB's fast COPY command
            print(f"Bulk loading steps from {steps_csv_path}...")
            steps_columns = "(global_episode_id, step_number, action, reward, done, success, codeword_weight, syndrome_weight, flip_weight, penalty_sum, step_count, next_codeword_weight, next_syndrome_weight, next_flip_weight, next_penalty_sum, next_step_count, training_loss, training_epsilon, q_values_mean, q_values_std, q_values_min, q_values_max, target_q_values_mean, target_q_values_std, target_q_values_min, target_q_values_max)"
            self.con.execute(f"COPY steps {steps_columns} FROM '{steps_csv_path}' (FORMAT CSV, HEADER TRUE)")
            print("Steps bulk load complete.")

        # Create indexes after all data is loaded for optimal performance
        print("Creating indexes on loaded data...")
        self._create_indexes()
        print("Indexes created successfully.")

        print(f"Successfully loaded experiment {experiment_id} into database")

    def _extract_error_info(self, episode: Dict) -> tuple:
        """
        Extracts error information from episode data.
        Returns: (error_pattern, error_model, num_errors, error_parameters_json)

        Handles two different JSON formats:
        - Training format: flat structure with 'error_model' and 'parameters' at top level
        - Evaluation format: nested structure with 'meta' sub-dictionary containing 'error_model' and 'num_errors'
        """
        error_info = episode.get('error_info', {})
        error_pattern = error_info.get('error_pattern')

        # Check for nested 'meta' structure (evaluation data format)
        if 'meta' in error_info:
            meta = error_info.get('meta', {})
            error_model = meta.get('error_model')
            num_errors = meta.get('num_errors', 0)
            # Extract other parameters from meta (excluding the ones with dedicated columns)
            parameters = {k: v for k, v in meta.items() if k not in ['error_model', 'num_errors']}
        else:
            # Flat structure (training data format)
            error_model = error_info.get('error_model')
            parameters = error_info.get('parameters', {})
            num_errors = parameters.get('num_errors', 0)

        # Calculate num_errors from error_pattern if not already set
        if error_pattern and num_errors == 0:
            num_errors = sum(error_pattern)

        # Create error_parameters JSON, excluding num_errors since it has its own column
        error_parameters = {k: v for k, v in parameters.items() if k != 'num_errors'}
        error_parameters_json = json.dumps(error_parameters) if error_parameters else None

        return error_pattern, error_model, num_errors, error_parameters_json

    def _format_array_for_csv(self, arr) -> str:
        """Convert array to DuckDB INTEGER[] format."""
        if arr is None:
            return None
        if isinstance(arr, (list, tuple)):
            return f"[{','.join(map(str, arr))}]"
        return str(arr)

    def _format_json_for_csv(self, json_str) -> str:
        """Format JSON string for CSV - store as VARCHAR."""
        return json_str

    def _write_episodes_to_csv(self, all_episodes: List[Dict],
                             experiment_id: str, csv_path: Path) -> List[int]:
        """
        Transforms episode data and writes it to CSV file for bulk import.
        Returns list of explicitly assigned global_episode_ids.
        """
        header = [
            "global_episode_id",  # NEW: Explicit ID column as FIRST column
            "experiment_id", "episode_type", "original_episode_id", "agent_type", "phase",
            "original_codeword", "noisy_codeword", "error_pattern", "error_model",
            "num_errors", "error_parameters", "terminal_reason", "episode_length", "success"
        ]

        # Generate explicit global_episode_ids BEFORE writing CSV
        try:
            current_seq = self.con.execute("SELECT currval('episodes_seq')").fetchone()[0]
        except Exception:
            # Sequence not yet used - start from 0
            current_seq = 0

        # Generate explicit IDs for all episodes
        start_id = current_seq + 1
        all_global_ids = list(range(start_id, start_id + len(all_episodes)))

        total_episodes = 0
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL, escapechar=None)
            writer.writerow(header)
            
            # Write all episodes with explicit IDs
            for episode, global_id in zip(all_episodes, all_global_ids):
                error_pattern, error_model, num_errors, error_parameters_json = self._extract_error_info(episode)
                row = [
                    global_id,  # NEW: Explicit global_episode_id as FIRST value
                    experiment_id,
                    episode.get('episode_type', 'unknown'),
                    episode.get('episode_id', 0),
                    episode.get('agent_type'),
                    episode.get('phase'),
                    self._format_array_for_csv(episode.get('original_codeword')),
                    self._format_array_for_csv(episode.get('noisy_codeword')),
                    self._format_array_for_csv(error_pattern),
                    error_model,
                    num_errors,
                    self._format_json_for_csv(error_parameters_json),
                    episode.get('terminal_reason'),
                    len(episode.get('steps', [])),
                    episode.get('terminal_reason') == 'SUCCESS_STOP'
                ]
                writer.writerow(row)
                total_episodes += 1
        
        print(f"Wrote {total_episodes:,} episodes to temporary CSV file.")

        # Advance the sequence to account for IDs we just assigned
        # DuckDB doesn't have setval(), so we drop and recreate the sequence
        final_seq_value = start_id + len(all_episodes) - 1
        self.con.execute("DROP SEQUENCE IF EXISTS episodes_seq")
        self.con.execute(f"CREATE SEQUENCE episodes_seq START {final_seq_value}")
        # Call nextval() once to initialize currval to final_seq_value
        self.con.execute("SELECT nextval('episodes_seq')").fetchone()

        return all_global_ids

    def _write_steps_to_csv(self, all_episodes: List[Dict], all_global_ids: List[int], csv_path: Path):
        """
        Transforms step data from all episodes and writes it to a CSV file for bulk import.
        """
        header = [
            "global_episode_id", "step_number", "action", "reward", "done", "success",
            "codeword_weight", "syndrome_weight", "flip_weight", "penalty_sum", "step_count",
            "next_codeword_weight", "next_syndrome_weight", "next_flip_weight", "next_penalty_sum", "next_step_count",
            "training_loss", "training_epsilon", "q_values_mean", "q_values_std", "q_values_min", "q_values_max",
            "target_q_values_mean", "target_q_values_std", "target_q_values_min", "target_q_values_max"
        ]
        
        if len(all_episodes) != len(all_global_ids):
            print(f"WARNING: Episode/ID length mismatch: {len(all_episodes)} episodes vs {len(all_global_ids)} IDs")

        total_steps = 0
        steps_with_loss = 0
        episodes_processed = 0

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for episode, global_episode_id in zip(all_episodes, all_global_ids):
                episodes_processed += 1
                for step in episode.get('steps', []):
                    state = step.get('state', {})
                    next_state = step.get('next_state', {})
                    training_update = step.get('training_update', {})
                    q_values = training_update.get('q_values', {})
                    target_q_values = training_update.get('target_q_values', {})

                    loss_value = training_update.get('loss')
                    if loss_value is not None:
                        steps_with_loss += 1

                    row = [
                        global_episode_id,
                        step.get('step'),
                        step.get('action'),
                        step.get('reward'),
                        step.get('done'),
                        step.get('success'),
                        state.get('codeword_weight'),
                        state.get('syndrome_weight'),
                        state.get('flip_weight'),
                        state.get('penalty_sum'),
                        state.get('step_count'),
                        next_state.get('codeword_weight') if next_state else None,
                        next_state.get('syndrome_weight') if next_state else None,
                        next_state.get('flip_weight') if next_state else None,
                        next_state.get('penalty_sum') if next_state else None,
                        next_state.get('step_count') if next_state else None,
                        loss_value,
                        training_update.get('epsilon'),
                        q_values.get('mean'),
                        q_values.get('std'),
                        q_values.get('min'),
                        q_values.get('max'),
                        target_q_values.get('mean'),
                        target_q_values.get('std'),
                        target_q_values.get('min'),
                        target_q_values.get('max')
                    ]
                    writer.writerow(row)
                    total_steps += 1

        print(f"Wrote {total_steps:,} steps ({steps_with_loss:,} with loss data) to temporary CSV file.")

    def _insert_experiment_metadata(self, experiment_id: str, experiment_name: str, config: dict):
        """Insert experiment metadata into experiments table."""
        self.con.execute(
            "INSERT INTO experiments (experiment_id, experiment_name, configuration) VALUES (?, ?, ?)",
            [experiment_id, experiment_name, json.dumps(config)]
        )
    
    def query(self, sql_query: str) -> pd.DataFrame:
        """
        Execute SQL query on the normalized database.
        """
        return self.con.execute(sql_query).df()

    def list_analyses(self, category: str = None, data_focus: str = None, complexity: str = None):
        """
        List available analyses with optional filtering.
        
        Args:
            category: Filter by category (e.g., 'training_dynamics', 'behavioral_analysis')
            data_focus: Filter by data focus ('training', 'evaluation', 'both')  
            complexity: Filter by complexity ('simple', 'medium', 'complex')
        """
        filtered_analyses = self.definitions
        
        # Apply filters
        if category:
            filtered_analyses = [a for a in filtered_analyses if a.get('category') == category]
        if data_focus:
            filtered_analyses = [a for a in filtered_analyses if a.get('data_focus') == data_focus]
        if complexity:
            filtered_analyses = [a for a in filtered_analyses if a.get('complexity') == complexity]
        
        if not filtered_analyses:
            print("No analyses match the specified filters.")
            return
        
        # Group by category for organized display
        categories = {}
        for analysis in filtered_analyses:
            cat = analysis.get('category', 'uncategorized')
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(analysis)
        
        print(f"\n--- Available Analyses ({len(filtered_analyses)} found) ---")
        
        for category_name, analyses in sorted(categories.items()):
            print(f"\n{category_name.replace('_', ' ').title()}")
            print("=" * (len(category_name) + 4))
            
            # Group by subcategory
            subcategories = {}
            for analysis in analyses:
                subcat = analysis.get('subcategory', 'general')
                if subcat not in subcategories:
                    subcategories[subcat] = []
                subcategories[subcat].append(analysis)
            
            for subcat_name, subcat_analyses in sorted(subcategories.items()):
                print(f"\n  {subcat_name.replace('_', ' ').title()}")
                for analysis in subcat_analyses:
                    complexity_label = {"simple": "[S]", "medium": "[M]", "complex": "[C]"}.get(
                        analysis.get('complexity', 'simple'), '[?]'
                    )
                    data_label = {"training": "[T]", "evaluation": "[E]", "both": "[B]"}.get(
                        analysis.get('data_focus', 'both'), '[?]'
                    )

                    print(f"    {complexity_label}{data_label} {analysis['id']}")
                    print(f"      Title: {analysis['title']}")
                    print(f"      Description: {analysis['description']}")
                    
                    # Show prerequisites if they exist
                    if 'prerequisites' in analysis:
                        prereqs = analysis['prerequisites']
                        req_list = []
                        if prereqs.get('min_sample_size'):
                            req_list.append(f"min {prereqs['min_sample_size']} samples")
                        if prereqs.get('requires_training_data'):
                            req_list.append("training data")
                        if prereqs.get('requires_evaluation_data'):
                            req_list.append("evaluation data")
                        if prereqs.get('requires_multiple_phases'):
                            req_list.append("multiple phases")
                        if prereqs.get('requires_multiple_agents'):
                            req_list.append("multiple agents")
                        
                        if req_list:
                            print(f"      Requirements: {', '.join(req_list)}")
                    
                    print()
        
        print("\nUsage: python scripts/run_analysis.py --experiment_id <id> --analysis_id <analysis_id>")
        print("Filters: --category <cat> --data-focus <focus> --complexity <level>")
        
        # Show available filter values
        all_categories = sorted(set(a.get('category', 'uncategorized') for a in self.definitions))
        all_data_focus = sorted(set(a.get('data_focus', 'both') for a in self.definitions))
        all_complexity = sorted(set(a.get('complexity', 'simple') for a in self.definitions))
        
        print(f"\nAvailable Categories: {', '.join(all_categories)}")
        print(f"Available Data Focus: {', '.join(all_data_focus)}")
        print(f"Available Complexity: {', '.join(all_complexity)}")
    
    
    
    def run_analysis(self, analysis_id: str, params: dict = None, data_only: bool = False):
        """
        Run a specific analysis by ID using the normalized database.

        Args:
            analysis_id: ID of the analysis to run
            params: Optional parameters for query template formatting
            data_only: If True, return DataFrame regardless of result_type (skip plotting)
        """
        analysis = next((item for item in self.definitions if item["id"] == analysis_id), None)
        if not analysis:
            raise ValueError(f"Analysis with ID '{analysis_id}' not found.")

        print(f"Running analysis: {analysis['title']}")

        query = analysis['query_template']
        if params:
            query = query.format(**params)

        result_df = self.con.execute(query).df()

        # If data_only mode, return DataFrame immediately
        if data_only:
            return result_df

        if analysis['result_type'] == 'dataframe':
            return result_df
        elif analysis['result_type'] == 'plot':
            return self._generate_plot(result_df, analysis['plot_config'])
        else:
            raise ValueError(f"Unknown result_type: {analysis['result_type']}")
    
    def _generate_plot(self, df: pd.DataFrame, config: dict):
        """
        Generate enhanced plot from DataFrame based on configuration.
        Supports multiple plot types, multi-series, color schemes, and statistical overlays.
        """
        # Set color palette based on config
        color_scheme = config.get('color_scheme', 'tab10')
        if color_scheme in ['viridis', 'plasma', 'inferno', 'magma']:
            plt.cm.get_cmap(color_scheme)
        
        sns.set_theme(style="whitegrid" if config.get('show_grid', True) else "white")
        fig, ax = plt.subplots(figsize=(12, 8))
        
        plot_type = config.get('plot_type', 'line')
        x_axis = config['x_axis']
        y_axis = config['y_axis']
        color_by = config.get('color_by')
        
        # Generate plot based on type
        if plot_type == 'line':
            if color_by:
                sns.lineplot(data=df, x=x_axis, y=y_axis, hue=color_by, 
                           marker='o', ax=ax, palette=color_scheme)
            else:
                sns.lineplot(data=df, x=x_axis, y=y_axis, marker='o', ax=ax, color='steelblue')
                
        elif plot_type == 'bar':
            if color_by:
                sns.barplot(data=df, x=x_axis, y=y_axis, hue=color_by, ax=ax, palette=color_scheme)
            else:
                sns.barplot(data=df, x=x_axis, y=y_axis, ax=ax, color='steelblue')
        
        elif plot_type == 'scatter':
            size_by = config.get('size_by')
            if color_by and size_by:
                scatter = ax.scatter(df[x_axis], df[y_axis], c=df[color_by].astype('category').cat.codes, 
                                   s=df[size_by]*3, alpha=0.7, cmap=color_scheme)
                # Add color legend
                handles, labels = ax.get_legend_handles_labels()
                if color_by in df.columns:
                    unique_colors = df[color_by].unique()
                    for i, color_val in enumerate(unique_colors):
                        ax.scatter([], [], c=plt.cm.get_cmap(color_scheme)(i/len(unique_colors)), 
                                 label=str(color_val), s=50)
            elif color_by:
                sns.scatterplot(data=df, x=x_axis, y=y_axis, hue=color_by, ax=ax, 
                               palette=color_scheme, s=100, alpha=0.7)
            else:
                ax.scatter(df[x_axis], df[y_axis], alpha=0.7, s=50, color='steelblue')
        
        elif plot_type == 'heatmap':
            # For heatmap, we need to pivot the data
            value_axis = config.get('value_axis', y_axis)
            if color_by:
                pivot_df = df.pivot_table(values=value_axis, index=y_axis, columns=x_axis, fill_value=0)
            else:
                pivot_df = df.pivot_table(values=value_axis, index=y_axis, columns=x_axis, fill_value=0)
            
            sns.heatmap(pivot_df, annot=True, cmap=color_scheme, ax=ax, 
                       cbar_kws={'label': value_axis}, fmt='.2f')
        
        # Apply styling
        ax.set_title(config.get('title', ''), fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel(config.get('xlabel', ''), fontsize=12, fontweight='semibold')
        ax.set_ylabel(config.get('ylabel', ''), fontsize=12, fontweight='semibold')
        
        # Handle error bars for confidence intervals
        if config.get('error_bars') and config['error_bars'] in df.columns:
            if plot_type == 'bar':
                # Add error bars to bar plot
                for i, (idx, row) in enumerate(df.iterrows()):
                    ax.errorbar(i, row[y_axis], yerr=row[config['error_bars']], 
                              fmt='none', color='black', capsize=3, capthick=1)
        
        # Legend configuration
        if config.get('show_legend', True) and color_by:
            ax.legend(title=color_by, bbox_to_anchor=(1.05, 1), loc='upper left')
        elif not config.get('show_legend', True):
            ax.get_legend().remove() if ax.get_legend() else None
        
        # Grid configuration
        if config.get('show_grid', True):
            ax.grid(True, alpha=0.3, linestyle='--')
        else:
            ax.grid(False)
        
        # Logarithmic scale
        if config.get('log_y', False):
            ax.set_yscale('log')
        if config.get('log_x', False):
            ax.set_xscale('log')
        
        # Add trend line if requested
        if config.get('show_trend_line', False) and plot_type in ['line', 'scatter']:
            try:
                import numpy as np
                if len(df) > 1:
                    z = np.polyfit(df[x_axis].astype(float), df[y_axis].astype(float), 1)
                    p = np.poly1d(z)
                    ax.plot(df[x_axis], p(df[x_axis].astype(float)), 
                           linestyle='--', alpha=0.8, color='red', linewidth=2)
            except Exception:
                pass  # Skip trend line if data is not suitable
        
        # Rotate x-axis labels if needed
        if len(str(df[x_axis].iloc[0] if len(df) > 0 else '')) > 8:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    def get_database_stats(self) -> pd.DataFrame:
        """
        Get statistics about the loaded data.
        """
        stats_query = """
        SELECT 
            'experiments' as table_name, COUNT(*) as row_count FROM experiments
        UNION ALL
        SELECT 
            'episodes' as table_name, COUNT(*) as row_count FROM episodes
        UNION ALL
        SELECT 
            'steps' as table_name, COUNT(*) as row_count FROM steps
        ORDER BY table_name
        """
        return self.query(stats_query)