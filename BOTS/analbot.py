# BOTS/analbot.py - Analytic bot module for evaluating trading strategies
import os
import sys
import inspect
import time
import pandas as pd
import concurrent.futures
from functools import lru_cache, partial
from hashlib import md5
import pickle
import json
from multiprocessing import cpu_count
from typing import Dict, List, Any, Optional, Union, Type, Callable, TypeVar, cast, Tuple

sys.path.append(os.path.abspath("."))
from BOTS.loggerbot import Logger
from BOTS.indicators import Indicators
from STRATEGY.rsi import RSIonly_Strategy
from STRATEGY.base import BaseStrategy

# Type variable for strategy classes
T = TypeVar('T', bound=BaseStrategy)

class Analytic:
    def __init__(self, df: pd.DataFrame, data_name: str, output_file: str = "anal.csv", 
                 cache_dir: str = "DATA/cache") -> None:
        """
        Initialize the Analytic class with data and output settings.

        Args:
            df: DataFrame containing price data
            data_name: Name of the data (used for output file naming)
            output_file: Suffix for the output file name
            cache_dir: Directory for caching strategy results
        """
        self.df: pd.DataFrame = df
        self.logger = Logger(name="Analytic", tag="[ANAL]", logfile="LOGS/analytic.log", console=True).get_logger()
        self.indicators = Indicators(df, self.logger)
        self.output_file: str = output_file
        self.data_name: str = data_name
        self.output_path: str = f'DATA/{data_name}_{output_file}'
        self.cache_dir: str = cache_dir

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

    @staticmethod
    @lru_cache(maxsize=128)
    def _get_expected_columns(name: str, params_tuple: Tuple[Tuple[str, Any], ...]) -> List[str]:
        """
        Get the expected column names for a given indicator with specific parameters.

        This method is cached to avoid recalculating the same column names multiple times.

        Args:
            name: Name of the indicator
            params_tuple: Parameters for the indicator as a tuple of (key, value) pairs

        Returns:
            List of expected column names in the DataFrame
        """
        # Convert params_tuple back to a dictionary
        params = dict(params_tuple)

        defaults: Dict[str, Dict[str, Any]] = {
            "sma": {"period": 10},
            "ema": {"period": 10},
            "rsi": {"period": 14},
            "macd": {"window_fast": 12, "window_slow": 26},
            "bollinger_bands": {"period": 20},
        }

        def is_default(param: str, value: Any) -> bool:
            return defaults.get(name, {}).get(param) == value

        if name == "sma":
            suffix = f"_{params['period']}" if not is_default("period", params.get("period")) else ""
            return [f"sma{suffix}"]

        elif name == "ema":
            suffix = f"_{params['period']}" if not is_default("period", params.get("period")) else ""
            return [f"ema{suffix}"]

        elif name == "rsi":
            suffix = f"_{params['period']}" if not is_default("period", params.get("period")) else ""
            return [f"rsi{suffix}"]

        elif name == "macd":
            fast = params.get("window_fast")
            slow = params.get("window_slow")
            is_def = is_default("window_fast", fast) and is_default("window_slow", slow)
            suffix = f"_{fast}_{slow}" if not is_def else ""
            return [f"macd{suffix}"]

        elif name == "bollinger_bands":
            p = params["period"]
            suffix = f"_{p}" if not is_default("period", p) else ""
            return [f"bb_h{suffix}", f"bb_m{suffix}", f"bb_l{suffix}"]

        return []

    def _get_expected_columns_dict(self, name: str, params: Dict[str, Any]) -> List[str]:
        """
        Wrapper for _get_expected_columns that accepts a dictionary.

        Args:
            name: Name of the indicator
            params: Parameters for the indicator as a dictionary

        Returns:
            List of expected column names in the DataFrame
        """
        # Convert dictionary to a tuple of tuples for caching
        params_tuple = tuple(sorted(params.items()))
        return self._get_expected_columns(name, params_tuple)

    def _calculate_single_indicator(self, indicator_name: str, params: Dict[str, Any]) -> bool:
        """
        Calculate a single indicator with the given parameters.

        Args:
            indicator_name: Name of the indicator to calculate
            params: Parameters for the indicator

        Returns:
            bool: True if the indicator was calculated, False otherwise
        """
        self.logger.info(f"Calculating indicator {indicator_name} with parameters {params}")

        # Get the method from the indicators class
        method = getattr(self.indicators, indicator_name, None)
        if method is None:
            self.logger.warning(f"Indicator {indicator_name} not found.")
            return False

        # Check if the indicator is already calculated
        expected_columns = self._get_expected_columns_dict(indicator_name, params)
        missing = [col for col in expected_columns if col not in self.df.columns]
        if not missing:
            self.logger.info(f"Skipping {indicator_name} - already calculated.")
            return False

        # Filter only valid parameters for the method
        method_params = inspect.signature(method).parameters
        filtered_params = {k: v for k, v in params.items() if k in method_params}

        # Calculate the indicator
        try:
            method(inplace=True, **filtered_params)
            self.logger.info(f"Indicator {indicator_name} calculated successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Error calculating {indicator_name}: {e}")
            return False

    def make_calc(self, indicators: List[str], stratparams: Dict[str, Dict[str, Any]], 
              parallel: bool = True) -> None:
        """
        Calculate indicators based on strategy parameters.

        Args:
            indicators: List of indicator names to calculate
            stratparams: Dictionary of indicator parameters
            parallel: Whether to calculate indicators in parallel
        """
        self.logger.info(f"Calculating indicators: {stratparams}")

        # Filter indicators that are in stratparams
        indicators_to_calculate = [ind for ind in indicators if ind in stratparams]

        if not indicators_to_calculate:
            self.logger.warning("No indicators to calculate")
            return

        if not parallel or len(indicators_to_calculate) <= 1:
            # Calculate indicators sequentially
            for indicator_name in indicators_to_calculate:
                params = stratparams.get(indicator_name, {})
                self._calculate_single_indicator(indicator_name, params)
        else:
            # Calculate indicators in parallel
            # Use ThreadPoolExecutor since indicator calculations are mostly I/O-bound
            max_workers = min(len(indicators_to_calculate), cpu_count() * 2)

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(
                        self._calculate_single_indicator, 
                        indicator_name, 
                        stratparams.get(indicator_name, {})
                    ): indicator_name for indicator_name in indicators_to_calculate
                }

                # Process results as they complete
                for future in concurrent.futures.as_completed(futures):
                    indicator_name = futures[future]
                    try:
                        result = future.result()
                        if result:
                            self.logger.info(f"Parallel calculation of {indicator_name} completed successfully")
                    except Exception as e:
                        self.logger.error(f"Error in parallel calculation of {indicator_name}: {e}")
                        # Cancel remaining tasks if one fails
                        for f in futures:
                            f.cancel()

    def _save_results_to_csv(self) -> bool:
        """
        Save the analysis results to a CSV file.

        Returns:
            bool: True if the save was successful, False otherwise
        """
        try:
            self.df.to_csv(self.output_path, index=False)
            self.logger.info(f"Analysis data saved to {self.output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving to {self.output_path}: {e}")
            return False

    def _generate_cache_key(self, strategy_cls: Type[T], **params) -> str:
        """
        Generate a unique cache key for a strategy with specific parameters.

        Args:
            strategy_cls: Strategy class
            **params: Strategy parameters

        Returns:
            A unique cache key string
        """
        # Create a dictionary with strategy class name and parameters
        cache_dict = {
            "strategy_class": strategy_cls.__name__,
            "data_name": self.data_name,
            "params": params,
            # Include the last modified timestamp of the data file to invalidate cache when data changes
            "data_timestamp": os.path.getmtime(self.output_path) if os.path.exists(self.output_path) else 0
        }

        # Convert to a stable JSON string (sort keys for consistency)
        cache_json = json.dumps(cache_dict, sort_keys=True)

        # Create an MD5 hash of the JSON string
        cache_hash = md5(cache_json.encode()).hexdigest()

        return cache_hash

    def _get_cached_result(self, cache_key: str) -> Optional[int]:
        """
        Try to get a cached strategy result.

        Args:
            cache_key: The cache key for the strategy

        Returns:
            The cached result if available, None otherwise
        """
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")

        if not os.path.exists(cache_file):
            return None

        try:
            # Check if the cache file is recent enough (less than 1 day old)
            cache_age = time.time() - os.path.getmtime(cache_file)
            if cache_age > 86400:  # 24 hours in seconds
                self.logger.info(f"Cache file {cache_file} is too old, recalculating")
                return None

            with open(cache_file, 'rb') as f:
                result = pickle.load(f)
                self.logger.info(f"Using cached result for strategy (key: {cache_key})")
                return result
        except Exception as e:
            self.logger.warning(f"Error reading cache file {cache_file}: {e}")
            return None

    def _cache_result(self, cache_key: str, result: int) -> None:
        """
        Cache a strategy result.

        Args:
            cache_key: The cache key for the strategy
            result: The result to cache
        """
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            self.logger.info(f"Cached result for strategy (key: {cache_key})")
        except Exception as e:
            self.logger.warning(f"Error writing cache file {cache_file}: {e}")

    def make_strategy(self, strategy_cls: Type[T], inplace: bool = True, 
                  parallel: bool = True, use_cache: bool = True, **params) -> int:
        """
        Apply a trading strategy to the data.

        Args:
            strategy_cls: Strategy class to instantiate
            inplace: Whether to save results to file
            parallel: Whether to calculate indicators in parallel
            use_cache: Whether to use cached results if available
            **params: Parameters to pass to the strategy

        Returns:
            Signal value from the strategy
        """
        # Check cache first if enabled
        if use_cache:
            cache_key = self._generate_cache_key(strategy_cls, **params)
            cached_result = self._get_cached_result(cache_key)
            if cached_result is not None:
                return cached_result

        # Create strategy instance
        strategy = strategy_cls(**params)
        self.logger.info(f"Starting calculation for strategy {strategy.name}")

        # Get required indicators and calculate them
        indicators, stratparams = strategy.check_indicators()

        # Use batch processing for large datasets
        if len(self.df) > 10000:  # Threshold for "large" dataset
            self.logger.info(f"Using batch processing for large dataset ({len(self.df)} rows)")
            # Process in batches of 10,000 rows
            batch_size = 10000
            num_batches = (len(self.df) + batch_size - 1) // batch_size  # Ceiling division

            # Process each batch
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(self.df))
                self.logger.info(f"Processing batch {i+1}/{num_batches} (rows {start_idx}-{end_idx})")

                # Create a view of the dataframe for this batch
                batch_df = self.df.iloc[start_idx:end_idx].copy()

                # Create a temporary Analytic instance for this batch
                batch_indicators = Indicators(batch_df, self.logger)

                # Calculate indicators for this batch
                for indicator_name in indicators:
                    params = stratparams.get(indicator_name, {})
                    method = getattr(batch_indicators, indicator_name, None)
                    if method is not None:
                        method_params = inspect.signature(method).parameters
                        filtered_params = {k: v for k, v in params.items() if k in method_params}
                        try:
                            method(inplace=True, **filtered_params)
                        except Exception as e:
                            self.logger.error(f"Error calculating {indicator_name} for batch {i+1}: {e}")

                # Copy the calculated indicators back to the main dataframe
                for col in batch_df.columns:
                    if col not in self.df.columns:
                        self.df[col] = None
                    self.df.iloc[start_idx:end_idx, self.df.columns.get_loc(col)] = batch_df[col]
        else:
            # For smaller datasets, use the standard calculation method
            self.make_calc(indicators, stratparams, parallel=parallel)

        # Generate signals using the strategy
        start_time = time.time()
        result = strategy.get_signals(self.df)
        end_time = time.time()
        self.logger.info(f"Strategy signal generation took {end_time - start_time:.2f} seconds")

        # Save results if requested
        if inplace:
            self._save_results_to_csv()

        # Cache the result if caching is enabled
        if use_cache:
            cache_key = self._generate_cache_key(strategy_cls, **params)
            self._cache_result(cache_key, result)

        return result


if __name__ == "__main__":
    import pandas as pd
    # Get the path to the current file (equivalent to __file__)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Path one level up to the DATA folder
    csv_path = os.path.join(current_dir, "..", "DATA", "BTCUSDT_1h.csv")
    csv_path = os.path.abspath(csv_path)  # absolute path (just to be safe)

    # Load the data
    df = pd.read_csv(csv_path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Create an Analytic instance and run a strategy
    analyzer = Analytic(df, "BTCUSDT_1h")
    result = analyzer.make_strategy(RSIonly_Strategy, rsi={"period": 14, "lower": 20})
    print(result)
