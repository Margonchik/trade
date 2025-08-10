# API/binance_api.py

import os
import sys
import ccxt
import pandas as pd
from typing import Dict, Any, Optional

sys.path.append(os.path.abspath("."))

from API.birza_api import BirzaAPI

class BinanceAPI(BirzaAPI):
    """
    API client for the Binance cryptocurrency exchange.

    This class implements the BirzaAPI interface for the Binance exchange
    using the ccxt library.
    """

    def __init__(self, api_key: Optional[str], api_secret: Optional[str], testnet: bool = True):
        """
        Initialize the Binance API client.

        Args:
            api_key: API key for authentication (None for public API only)
            api_secret: API secret for authentication (None for public API only)
            testnet: Whether to use the testnet (sandbox) environment
        """
        super().__init__(name="binanceAPI", log_tag="[API]", log_file="LOGS/binanceAPI.log", console=True)

        # Initialize the ccxt exchange object
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        })

        # Set sandbox mode if testnet is True
        if testnet:
            self.exchange.set_sandbox_mode(True)

        try:
            self.logger.info("Initializing connection to Binance via ccxt...")
            if api_key and api_secret:
                assets = self.get_balance()
                self.logger.info(f"Connection established. Assets: {list(assets.keys())}")
            else:
                # For public API, just check if the exchange is accessible
                self.exchange.fetch_status()
                self.logger.info("Connection established (public API only).")
        except Exception as e:
            self.logger.exception(f"Error initializing BinanceAPI: {e}")
            raise

    def get_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 100) -> pd.DataFrame:
        """
        Fetch OHLCV candlestick data from Binance.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Candlestick timeframe (e.g., "1m", "5m", "1h", "1d")
            limit: Maximum number of candles to fetch

        Returns:
            DataFrame containing OHLCV data
        """
        try:
            self.logger.info(f"Fetching OHLCV: symbol={symbol}, timeframe={timeframe}, limit={limit}")
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.rename(columns={"timestamp": "time"}, inplace=True)
            return df
        except Exception as e:
            return self._handle_error(f"fetching OHLCV for {symbol}", e, pd.DataFrame())

    def place_order(self, symbol: str, side: str, qty: float, 
                   order_type: str = "market", price: Optional[float] = None) -> Dict[str, Any]:
        """
        Place a trading order on Binance.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            side: Order side ("buy" or "sell")
            qty: Order quantity
            order_type: Order type ("market", "limit", etc.)
            price: Order price (required for limit orders)

        Returns:
            Order information including order ID and status
        """
        try:
            self.logger.info(f"Creating order: {side.upper()} {qty} {symbol}, type={order_type.upper()}, price={price}")
            params = {}

            if order_type == "market":
                return self.exchange.create_market_order(symbol, side.lower(), qty, params)
            elif order_type == "limit":
                return self.exchange.create_limit_order(symbol, side.lower(), qty, price, params)
            else:
                raise ValueError("Invalid order type")
        except Exception as e:
            return self._handle_error(f"placing {order_type} order for {symbol}", e, {})

    def get_balance(self) -> Dict[str, Any]:
        """
        Fetch account balance information from Binance.

        Returns:
            Account balance information for all assets
        """
        try:
            balance = self.exchange.fetch_balance()
            return balance['total']
        except Exception as e:
            return self._handle_error("fetching balance", e, {})

    def get_positions(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current positions for a specific symbol.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")

        Returns:
            Position information
        """
        try:
            # Binance futures API supports positions
            if self.exchange.has['fetchPositions']:
                positions = self.exchange.fetch_positions([symbol])
                return {pos['symbol']: pos for pos in positions}
            else:
                self.logger.warning("The get_positions method is not fully supported for this Binance account type.")
                return {}
        except Exception as e:
            return self._handle_error(f"fetching positions for {symbol}", e, {})

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Fetch the status of a specific order from Binance.

        Args:
            order_id: Order ID to query

        Returns:
            Order status information
        """
        try:
            return self.exchange.fetch_order(order_id)
        except Exception as e:
            return self._handle_error(f"checking status of order {order_id}", e, {})


if __name__ == "__main__":
    bot = BinanceAPI(api_key=None, api_secret=None)
    bot.download_candles_to_csv("BTC/USDT", start_date="2023-05-05T00:00:00Z", timeframe="1h")
    df = bot.get_ohlcv("BTC/USDT")
