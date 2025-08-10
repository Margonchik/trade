# api/bybit_api.py

import os
import sys
import pandas as pd
import asyncio
import json
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()

import ccxt
import ccxt.async_support as ccxt_async
import ssl

sys.path.append(os.path.abspath("."))

from API.birza_api import BirzaAPI
from CORE.security import Security

class BybitAPI(BirzaAPI):
    """
    API client for the Bybit cryptocurrency exchange.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = True
    ):
        super().__init__(name="bybitAPI", log_tag="[API]", log_file="LOGS/bybitAPI.log", console=True)

        # если не передали явно — берём из окружения
        api_key = api_key or os.getenv("BYBIT_TOKEN")
        api_secret = api_secret or os.getenv("BYBIT_SECRET")

        if not api_key or not api_secret:
            raise ValueError("Bybit API key/secret not provided. Set BYBIT_TOKEN and BYBIT_SECRET in .env or pass them into constructor.")

        # синхронный клиент
        self.exchange = ccxt.bybit({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'rateLimit': 500,
            'options': {
                'verify': True,
                'timeout': 30000,
                'defaultType': 'spot'
            }
        })

        # асинхронный клиент
        self.async_exchange = ccxt_async.bybit({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'rateLimit': 500,
            'options': {
                'verify': True,
                'timeout': 30000,
                'defaultType': 'spot'
            }
        })

        if testnet:
            self.exchange.set_sandbox_mode(True)
            self.async_exchange.set_sandbox_mode(True)

        # проверяем баланс сразу и падаем, если ключи невалидные
        self.logger.info("Initializing connection to Bybit via ccxt...")
        assets = self.get_balance()
        self.logger.info(f"Connection established. Assets: {list(assets.keys())}")

    def get_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 100) -> pd.DataFrame:
        # ... без изменений ...
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
        # ... без изменений ...
        try:
            symbol = Security.sanitize_input(symbol)
            side = Security.sanitize_input(side).lower()
            self.logger.info(f"Creating order: {side.upper()} {qty} {symbol}, type={order_type.upper()}, price={price}")
            params = {}
            if order_type == "market":
                return self.exchange.create_market_order(symbol, side, qty, params)
            elif order_type == "limit":
                if price is None or price <= 0:
                    error_msg = "Price is required for limit orders and must be greater than 0"
                    self.logger.error(error_msg)
                    return {"error": error_msg, "status": "rejected"}
                return self.exchange.create_limit_order(symbol, side, qty, price, params)
            else:
                raise ValueError("Invalid order type")
        except Exception as e:
            return self._handle_error(f"placing {order_type} order for {symbol}", e, {})

    def get_balance(self) -> Dict[str, Any]:
        try:
            balance = self.exchange.fetch_balance(params={"type": "unified"})
            return balance['total']
        except Exception as e:
            return self._handle_error("fetching balance", e, {})

    def get_positions(self, symbol: str) -> Dict[str, Any]:
        if self.exchange.has.get('fetchPositions'):
            positions = self.exchange.fetch_positions([symbol])
            return {p['symbol']: p for p in positions}
        else:
            self.logger.warning("fetch_positions не поддерживается для данного типа аккаунта.")
            return {}

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        # ... без изменений ...
        try:
            return self.exchange.fetch_order(order_id)
        except Exception as e:
            return self._handle_error(f"checking status of order {order_id}", e, {})

    def download_candles_to_csv(self, symbol: str, start_date: str = "2023-01-01T00:00:00Z",
                                timeframe: str = "1h", save_folder: str = "DATA") -> pd.DataFrame:
        return super().download_candles_to_csv(symbol, start_date, timeframe, save_folder)

    # -------- async --------

    async def get_ohlcv_async(self, symbol: str, timeframe: str = "1m", limit: int = 100) -> pd.DataFrame:
        try:
            self.logger.info(f"Asynchronously fetching OHLCV: symbol={symbol}, timeframe={timeframe}, limit={limit}")
            ohlcv = await self.async_exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.rename(columns={"timestamp": "time"}, inplace=True)
            return df
        except Exception as e:
            return await self._handle_error_async(f"fetching OHLCV for {symbol}", e, pd.DataFrame())

    async def place_order_async(self, symbol: str, side: str, qty: float,
                   order_type: str = "market", price: Optional[float] = None) -> Dict[str, Any]:
        try:
            self.logger.info(f"Asynchronously creating order: {side.upper()} {qty} {symbol}, type={order_type.upper()}, price={price}")
            params = {}
            if order_type == "market":
                return await self.async_exchange.create_market_order(symbol, side.lower(), qty, params)
            elif order_type == "limit":
                return await self.async_exchange.create_limit_order(symbol, side.lower(), qty, price, params)
            else:
                raise ValueError("Invalid order type")
        except Exception as e:
            return await self._handle_error_async(f"placing {order_type} order for {symbol}", e, {})

    async def get_balance_async(self) -> Dict[str, Any]:
        try:
            balance = await self.async_exchange.fetch_balance(params={"type": "unified"})
            return balance['total']
        except Exception as e:
            return await self._handle_error_async("fetching balance", e, {})

    async def get_positions_async(self, symbol: str) -> Dict[str, Any]:
        self.logger.warning("The get_positions_async method is not supported by ccxt for Bybit. Returning empty dict.")
        return {}

    async def update_state(self, symbol="BTC/USDT", STATE_PATH= "DATA/static/state.json"):
        """
        Обновляет файл state.json с текущим балансом и символом.
        """
        try:
            balance = await self.get_balance_async()
            self.logger.info(f"Баланс: {balance}")
            quote_currency = symbol.split("/")[1]  # например USDT
            total = balance.get(quote_currency, 0)

            positions = []  # для spot пусто

            updated = datetime.now(timezone.utc).isoformat()

            state_data = {
                "balance": {
                    "total": total,
                    "currency": quote_currency
                },
                "positions": positions,
                "updated": updated
            }

            os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
            with open(STATE_PATH, "w", encoding="utf-8") as f:
                json.dump(state_data, f, indent=4)
            self.logger.info(f"Обновлено: {state_data}")
        except Exception as e:
            self.logger.error(f"Ошибка при обновлении: {e}")

    async def get_order_status_async(self, order_id: str) -> Dict[str, Any]:
        try:
            return await self.async_exchange.fetch_order(order_id)
        except Exception as e:
            return await self._handle_error_async(f"checking status of order {order_id}", e, {})

    async def download_candles_to_csv_async(self, symbol: str, start_date: str = "2023-01-01T00:00:00Z",
                                            timeframe: str = "1h", save_folder: str = "DATA") -> pd.DataFrame:
        return await super().download_candles_to_csv_async(symbol, start_date, timeframe, save_folder)

    async def close_async(self):
        if self.async_exchange:
            await self.async_exchange.close()


if __name__ == "__main__":
    bot = BybitAPI()
    bot.download_candles_to_csv("BTC/USDT", start_date="2025-05-05T00:00:00Z", timeframe="1h")
    # df = bot.get_ohlcv("BTC/USDT")
