# CORE/data_pipeline.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Callable, Any, Tuple
import logging
from functools import partial
import concurrent.futures
from multiprocessing import cpu_count

from CORE.error_handling import DataError
from CORE.data_validation import DataValidator


class DataPipeline:
    """
    Data pipeline for preprocessing OHLCV data.

    This class provides a pipeline for preprocessing OHLCV data before analysis
    or storage. It supports various preprocessing steps such as resampling,
    filling missing values, calculating technical indicators, and more.

    Attributes:
        logger: Logger instance for logging pipeline operations
        validator: DataValidator instance for validating data
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the DataPipeline.

        Args:
            logger: Logger instance (None to create a new one)
        """
        if logger is None:
            from BOTS.loggerbot import Logger
            self.logger = Logger(
                name="DataPipeline", 
                tag="[PIPELINE]", 
                logfile="LOGS/data_pipeline.log", 
                console=True
            ).get_logger()
        else:
            self.logger = logger

        self.validator = DataValidator(logger=self.logger)
        self.logger.info("Initialized DataPipeline")

    def _can_process_in_parallel(self, steps: List[Dict[str, Any]]) -> List[List[int]]:
        """
        Identify which steps can be processed in parallel.

        Args:
            steps: List of preprocessing steps to analyze

        Returns:
            List of lists, where each inner list contains indices of steps that can be processed in parallel
        """
        # Steps that modify the DataFrame structure and must be processed sequentially
        sequential_types = {'validate', 'clean', 'resample', 'filter'}

        # Group steps into batches that can be processed in parallel
        batches = []
        current_batch = []

        for i, step in enumerate(steps):
            step_type = step.get('type', '')

            # If this is a sequential step or the first step, start a new batch
            if step_type in sequential_types or not current_batch:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []

                # Sequential steps always go in their own batch
                if step_type in sequential_types:
                    batches.append([i])
                    continue

            # Add non-sequential steps to the current batch
            current_batch.append(i)

        # Add the last batch if it's not empty
        if current_batch:
            batches.append(current_batch)

        return batches

    def _apply_step(self, df: pd.DataFrame, step: Dict[str, Any], step_index: int) -> pd.DataFrame:
        """
        Apply a single preprocessing step to a DataFrame.

        Args:
            df: DataFrame to process
            step: Preprocessing step to apply
            step_index: Index of the step (for error reporting)

        Returns:
            Processed DataFrame

        Raises:
            DataError: If there's an error during processing
        """
        try:
            step_type = step.get('type')
            if not step_type:
                raise DataError(f"Step {step_index} is missing 'type' field")

            self.logger.debug(f"Applying step {step_index}: {step_type}")
            result_df = df.copy()

            if step_type == 'validate':
                # Validate the DataFrame
                strict = step.get('strict', False)
                is_valid, errors = self.validator.validate_ohlcv_dataframe(result_df, strict=strict)
                if not is_valid:
                    error_msg = f"Data validation failed:"
                    for err in errors:
                        error_msg += f"\n- {err}"
                    raise DataError(error_msg)

            elif step_type == 'clean':
                # Clean the DataFrame
                drop_duplicates = step.get('drop_duplicates', True)
                fill_missing = step.get('fill_missing', True)
                sort_timestamps = step.get('sort_timestamps', True)
                result_df = self.validator.clean_ohlcv_dataframe(
                    result_df, 
                    drop_duplicates=drop_duplicates,
                    fill_missing=fill_missing,
                    sort_timestamps=sort_timestamps
                )

            elif step_type == 'resample':
                # Resample the DataFrame to a different timeframe
                rule = step.get('rule')
                if not rule:
                    raise DataError(f"Resample step {step_index} is missing 'rule' field")

                agg_dict = step.get('aggregation', {
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                })

                # Ensure timestamp is the index for resampling
                if 'timestamp' in result_df.columns:
                    result_df = result_df.set_index('timestamp')

                # Resample
                result_df = result_df.resample(rule).agg(agg_dict)

                # Reset index to get timestamp back as a column
                result_df = result_df.reset_index()

            elif step_type == 'filter':
                # Filter rows based on a condition
                column = step.get('column')
                if not column:
                    raise DataError(f"Filter step {step_index} is missing 'column' field")

                operator = step.get('operator', '>')
                value = step.get('value')
                if value is None:
                    raise DataError(f"Filter step {step_index} is missing 'value' field")

                # Apply the filter
                if operator == '>':
                    result_df = result_df[result_df[column] > value]
                elif operator == '>=':
                    result_df = result_df[result_df[column] >= value]
                elif operator == '<':
                    result_df = result_df[result_df[column] < value]
                elif operator == '<=':
                    result_df = result_df[result_df[column] <= value]
                elif operator == '==':
                    result_df = result_df[result_df[column] == value]
                elif operator == '!=':
                    result_df = result_df[result_df[column] != value]
                else:
                    raise DataError(f"Filter step {step_index} has invalid operator: {operator}")

            elif step_type == 'add_indicator':
                # Add a technical indicator
                indicator = step.get('indicator')
                if not indicator:
                    raise DataError(f"Add indicator step {step_index} is missing 'indicator' field")

                params = step.get('params', {})

                # Import indicators dynamically to avoid circular imports
                from BOTS.indicators import Indicators

                # Create indicators instance
                indicators = Indicators(result_df, self.logger)

                # Call the indicator method
                if hasattr(indicators, indicator) and callable(getattr(indicators, indicator)):
                    method = getattr(indicators, indicator)
                    indicator_result = method(**params)
                    if indicator_result is not None:
                        result_df = indicator_result
                else:
                    raise DataError(f"Unknown indicator: {indicator}")

            elif step_type == 'drop_columns':
                # Drop specified columns
                columns = step.get('columns', [])
                if not columns:
                    raise DataError(f"Drop columns step {step_index} is missing 'columns' field")

                # Drop only columns that exist
                columns_to_drop = [col for col in columns if col in result_df.columns]
                if columns_to_drop:
                    result_df = result_df.drop(columns=columns_to_drop)

            elif step_type == 'rename_columns':
                # Rename columns
                mapping = step.get('mapping', {})
                if not mapping:
                    raise DataError(f"Rename columns step {step_index} is missing 'mapping' field")

                # Rename only columns that exist
                columns_to_rename = {k: v for k, v in mapping.items() if k in result_df.columns}
                if columns_to_rename:
                    result_df = result_df.rename(columns=columns_to_rename)

            elif step_type == 'custom':
                # Apply a custom function
                func = step.get('function')
                if not func or not callable(func):
                    raise DataError(f"Custom step {step_index} is missing 'function' field or it's not callable")

                args = step.get('args', [])
                kwargs = step.get('kwargs', {})

                # Apply the custom function
                result_df = func(result_df, *args, **kwargs)

            else:
                raise DataError(f"Unknown step type: {step_type}")

            # Check if the step produced an empty DataFrame
            if result_df.empty:
                self.logger.warning(f"Step {step_index} ({step_type}) produced an empty DataFrame")

            return result_df

        except Exception as e:
            raise DataError(f"Error in preprocessing step {step_index} ({step_type}): {str(e)}") from e

    def _apply_parallel_steps(self, df: pd.DataFrame, steps: List[Dict[str, Any]], 
                             step_indices: List[int]) -> pd.DataFrame:
        """
        Apply multiple steps in parallel and merge the results.

        Args:
            df: DataFrame to process
            steps: List of all preprocessing steps
            step_indices: Indices of steps to apply in parallel

        Returns:
            DataFrame with all parallel steps applied
        """
        if len(step_indices) == 1:
            # If there's only one step, process it directly
            return self._apply_step(df, steps[step_indices[0]], step_indices[0])

        # ProcessPoolExecutor would require pickling ``self`` (which holds a logger),
        # causing failures on some platforms.  ThreadPoolExecutor avoids this and is
        # sufficient here because most pandas operations release the GIL.
        max_workers = min(len(step_indices), cpu_count())

        # Create a copy of the DataFrame for each parallel step
        # This is necessary because each step might modify the DataFrame differently
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(
                    self._apply_step, df.copy(), steps[i], i
                ): i for i in step_indices
            }

            # Collect results as they complete
            results = {}
            for future in concurrent.futures.as_completed(futures):
                step_index = futures[future]
                try:
                    results[step_index] = future.result()
                except Exception as e:
                    # If any step fails, cancel all remaining tasks
                    for f in futures:
                        f.cancel()
                    raise e

        # Merge the results from all parallel steps
        # Start with the original DataFrame
        result_df = df.copy()

        # For each step result, merge its columns into the result DataFrame
        for step_index in step_indices:
            step_df = results[step_index]

            # Get new or modified columns from this step
            new_columns = set(step_df.columns) - set(df.columns)
            modified_columns = set(col for col in step_df.columns if col in df.columns 
                                  and not step_df[col].equals(df[col]))

            # Add new columns and update modified ones
            for col in new_columns.union(modified_columns):
                result_df[col] = step_df[col]

        return result_df

    def process(self, df: pd.DataFrame, steps: List[Dict[str, Any]], 
                parallel: bool = True) -> pd.DataFrame:
        """
        Process a DataFrame through a series of preprocessing steps.

        Args:
            df: DataFrame to process
            steps: List of preprocessing steps to apply
            parallel: Whether to process independent steps in parallel

        Returns:
            Processed DataFrame

        Raises:
            DataError: If there's an error during processing
        """
        if df.empty:
            self.logger.warning("Empty DataFrame provided for processing")
            return df

        # Make a copy to avoid modifying the original
        result_df = df.copy()

        if not parallel:
            # Process steps sequentially
            for i, step in enumerate(steps):
                result_df = self._apply_step(result_df, step, i)
                if result_df.empty:
                    break
        else:
            # Group steps into batches that can be processed in parallel
            batches = self._can_process_in_parallel(steps)

            # Process each batch
            for batch in batches:
                if len(batch) == 1:
                    # Single step, process directly
                    result_df = self._apply_step(result_df, steps[batch[0]], batch[0])
                else:
                    # Multiple steps, process in parallel
                    result_df = self._apply_parallel_steps(result_df, steps, batch)

                # Check if the batch produced an empty DataFrame
                if result_df.empty:
                    self.logger.warning(f"Batch {batch} produced an empty DataFrame")
                    break

        self.logger.info(f"Preprocessing complete: {len(steps)} steps applied")
        return result_df

    def create_pipeline(self, steps: List[Dict[str, Any]]) -> Callable[[pd.DataFrame], pd.DataFrame]:
        """
        Create a reusable pipeline function from a list of steps.

        Args:
            steps: List of preprocessing steps to apply

        Returns:
            Function that takes a DataFrame and returns a processed DataFrame
        """
        return partial(self.process, steps=steps)


# Example usage
if __name__ == "__main__":
    # Create a sample DataFrame
    data = {
        'timestamp': pd.date_range(start='2023-01-01', periods=10, freq='1H'),
        'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'high': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
        'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
        'close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
    }
    df = pd.DataFrame(data)

    # Create a pipeline
    pipeline = DataPipeline()

    # Define preprocessing steps
    steps = [
        {'type': 'validate'},
        {'type': 'clean'},
        {'type': 'resample', 'rule': '2H'},
        {'type': 'add_indicator', 'indicator': 'sma', 'params': {'period': 3}}
    ]

    # Process the DataFrame
    result = pipeline.process(df, steps)
    print(result.head())

    # Create a reusable pipeline function
    process_func = pipeline.create_pipeline(steps)

    # Use the pipeline function
    result2 = process_func(df)
    print(result2.head())
