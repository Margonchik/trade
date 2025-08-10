# API/birza_api.py

import os
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Optional, Any, Union, Coroutine
import logging
import asyncio

from API.data_parse import fetch_data

class BirzaAPI(ABC):
    """
    Abstract base class for cryptocurrency exchange API clients.

    This class defines the interface that all exchange-specific API clients
    must implement. It provides methods for fetching market data, placing orders,
    and retrieving account information.

    Attributes:
        logger: Logger instance for logging API operations
        exchange: Exchange client instance (implementation-specific)
    """

    def __init__(self, name: str, log_tag: str = "[API]", log_file: Optional[str] = None, console: bool = True):
        """
        Initialize the API client with logging configuration.

        Args:
            name: Name of the API client for logging
            log_tag: Tag to prepend to log messages
            log_file: Path to log file (None for no file logging)
            console: Whether to log to console
        """
        self.logger = self._setup_logger(name, log_tag, log_file, console)
        self.exchange = None

    def _setup_logger(self, name: str, tag: str, logfile: Optional[str], console: bool) -> logging.Logger:
        """
        Set up a logger for the API client.

        Args:
            name: Name of the logger
            tag: Tag to prepend to log messages
            logfile: Path to log file (None for no file logging)
            console: Whether to log to console

        Returns:
            Logger instance
        """
        # Import here to avoid circular imports
        from BOTS.loggerbot import Logger
        return Logger(name=name, tag=tag, logfile=logfile, console=console).get_logger()

    def _handle_error(self, operation: str, error: Exception, default_return: Any = None) -> Any:
        """
        Handle API operation errors consistently.

        Args:
            operation: Description of the operation that failed
            error: The exception that was raised
            default_return: Default value to return on error

        Returns:
            The default return value
        """
        self.logger.error(f"Error during {operation}: {error}")
        return default_return

    @abstractmethod
    def get_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 100) -> pd.DataFrame:
        """
        Fetch OHLCV (Open, High, Low, Close, Volume) candlestick data.

        Args:
            symbol (str): Trading pair symbol (e.g., "BTC/USDT")
            timeframe (str): Candlestick timeframe (e.g., "1m", "5m", "1h", "1d")
            limit (int): Maximum number of candles to fetch

        Returns:
            pd.DataFrame: DataFrame containing OHLCV data with columns:
                          [timestamp, open, high, low, close, volume]
        """
        pass

    @abstractmethod
    def place_order(self, symbol: str, side: str, qty: float,
                    order_type: str = "market", price: Optional[float] = None) -> Dict[str, Any]:
        """
        Place a trading order on the exchange.

        Args:
            symbol (str): Trading pair symbol (e.g., "BTC/USDT")
            side (str): Order side ("buy" or "sell")
            qty (float): Order quantity
            order_type (str): Order type ("market", "limit", etc.)
            price (Optional[float]): Order price (required for limit orders)

        Returns:
            Dict[str, Any]: Order information including order ID and status
        """
        pass

    @abstractmethod
    def get_balance(self) -> Dict[str, Any]:
        """
        Fetch account balance information.

        Returns:
            Dict[str, Any]: Account balance information for all assets
        """
        pass

    @abstractmethod
    def get_positions(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current positions for a specific symbol.

        Args:
            symbol (str): Trading pair symbol (e.g., "BTC/USDT")

        Returns:
            Dict[str, Any]: Position information including size, entry price, etc.
        """
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Fetch the status of a specific order.

        Args:
            order_id (str): Order ID to query

        Returns:
            Dict[str, Any]: Order status information
        """
        pass

    def download_candels_to_csv(self, symbol: str, start_date: str = "2023-01-01T00:00:00Z", 
                               timeframe: str = "1h", save_folder: str = "DATA") -> pd.DataFrame:
        """
        Download historical candle data and save to CSV.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            start_date: Start date for historical data in ISO format
            timeframe: Candlestick timeframe (e.g., "1m", "5m", "1h", "1d")
            save_folder: Folder to save CSV file (None to not save)

        Returns:
            DataFrame containing the downloaded data
        """
        exchange_name = self.__class__.__name__.replace("API", "").lower()
        self.logger.info(f"Downloading historical data for {symbol} from {start_date}, timeframe={timeframe}")

        try:
            df = fetch_data(exchange=exchange_name, symbol=symbol, start_date=start_date, timeframe=timeframe)

            if save_folder is not None:
                file_name = f'{symbol.replace("/", "")}_{timeframe}.csv'
                save_path = f'{save_folder}/{file_name}'

                try:    
                    os.makedirs(save_folder, exist_ok=True)
                    df.to_csv(save_path, index=False)
                    self.logger.info(f"Data saved to: {save_path}")
                except Exception as e:
                    return self._handle_error(f"saving data to {save_path}", e, df)

            return df
        except Exception as e:
            return self._handle_error(f"downloading data for {symbol}", e, pd.DataFrame())

    # Asynchronous API methods

    async def _handle_error_async(self, operation: str, error: Exception, default_return: Any = None) -> Any:
        """
        Handle API operation errors consistently in async context.

        Args:
            operation: Description of the operation that failed
            error: The exception that was raised
            default_return: Default value to return on error

        Returns:
            The default return value
        """
        self.logger.error(f"Error during async {operation}: {error}")
        return default_return

    @abstractmethod
    async def get_ohlcv_async(self, symbol: str, timeframe: str = "1m", limit: int = 100) -> pd.DataFrame:
        """
        Asynchronously fetch OHLCV (Open, High, Low, Close, Volume) candlestick data.

        Args:
            symbol (str): Trading pair symbol (e.g., "BTC/USDT")
            timeframe (str): Candlestick timeframe (e.g., "1m", "5m", "1h", "1d")
            limit (int): Maximum number of candles to fetch

        Returns:
            pd.DataFrame: DataFrame containing OHLCV data with columns:
                          [timestamp, open, high, low, close, volume]
        """
        pass

    @abstractmethod
    async def place_order_async(self, symbol: str, side: str, qty: float,
                    order_type: str = "market", price: Optional[float] = None) -> Dict[str, Any]:
        """
        Asynchronously place a trading order on the exchange.

        Args:
            symbol (str): Trading pair symbol (e.g., "BTC/USDT")
            side (str): Order side ("buy" or "sell")
            qty (float): Order quantity
            order_type (str): Order type ("market", "limit", etc.)
            price (Optional[float]): Order price (required for limit orders)

        Returns:
            Dict[str, Any]: Order information including order ID and status
        """
        pass

    @abstractmethod
    async def get_balance_async(self) -> Dict[str, Any]:
        """
        Asynchronously fetch account balance information.

        Returns:
            Dict[str, Any]: Account balance information for all assets
        """
        pass

    @abstractmethod
    async def get_positions_async(self, symbol: str) -> Dict[str, Any]:
        """
        Asynchronously fetch current positions for a specific symbol.

        Args:
            symbol (str): Trading pair symbol (e.g., "BTC/USDT")

        Returns:
            Dict[str, Any]: Position information including size, entry price, etc.
        """
        pass

    @abstractmethod
    async def get_order_status_async(self, order_id: str) -> Dict[str, Any]:
        """
        Asynchronously fetch the status of a specific order.

        Args:
            order_id (str): Order ID to query

        Returns:
            Dict[str, Any]: Order status information
        """
        pass

    async def download_candels_to_csv_async(self, symbol: str, start_date: str = "2023-01-01T00:00:00Z", 
                               timeframe: str = "1h", save_folder: str = "DATA") -> pd.DataFrame:
        """
        Asynchronously download historical candle data and save to CSV.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            start_date: Start date for historical data in ISO format
            timeframe: Candlestick timeframe (e.g., "1m", "5m", "1h", "1d")
            save_folder: Folder to save CSV file (None to not save)

        Returns:
            DataFrame containing the downloaded data
        """
        exchange_name = self.__class__.__name__.replace("API", "").lower()
        self.logger.info(f"Asynchronously downloading historical data for {symbol} from {start_date}, timeframe={timeframe}")

        try:
            # Note: fetch_data would need an async version for full async support
            # For now, we'll use the synchronous version in an executor
            loop = asyncio.get_running_loop()
            df = await loop.run_in_executor(
                None, 
                lambda: fetch_data(exchange=exchange_name, symbol=symbol, start_date=start_date, timeframe=timeframe)
            )

            if save_folder is not None:
                file_name = f'{symbol.replace("/", "")}_{timeframe}.csv'
                save_path = f'{save_folder}/{file_name}'

                try:    
                    os.makedirs(save_folder, exist_ok=True)
                    await loop.run_in_executor(None, lambda: df.to_csv(save_path, index=False))
                    self.logger.info(f"Data saved to: {save_path}")
                except Exception as e:
                    return await self._handle_error_async(f"saving data to {save_path}", e, df)

            return df
        except Exception as e:
            return await self._handle_error_async(f"downloading data for {symbol}", e, pd.DataFrame())
