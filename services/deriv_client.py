"""Deriv WebSocket API client with caching and auto-reconnect."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Optional

import websockets
from websockets.asyncio.client import ClientConnection

logger = logging.getLogger(__name__)

DERIV_WS_URL = "wss://ws.derivws.com/websockets/v3"

# Symbols suitable for the game (synthetic indices — always available, no market hours)
DEFAULT_SYMBOLS = [
    {"symbol": "R_100", "display_name": "Volatility 100 Index", "market": "synthetic_index"},
    {"symbol": "R_75", "display_name": "Volatility 75 Index", "market": "synthetic_index"},
    {"symbol": "R_50", "display_name": "Volatility 50 Index", "market": "synthetic_index"},
    {"symbol": "R_25", "display_name": "Volatility 25 Index", "market": "synthetic_index"},
    {"symbol": "R_10", "display_name": "Volatility 10 Index", "market": "synthetic_index"},
    {"symbol": "1HZ10V", "display_name": "Volatility 10 (1s) Index", "market": "synthetic_index"},
    {"symbol": "1HZ100V", "display_name": "Volatility 100 (1s) Index", "market": "synthetic_index"},
]


class DerivClient:
    """WebSocket client for the Deriv API."""

    def __init__(self, app_id: str = "1089") -> None:
        self.app_id = app_id
        self._ws: Optional[ClientConnection] = None
        self._req_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._connected = False
        self._recv_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        self._should_run = False

        # Caches
        self._candle_cache: dict[str, dict[str, Any]] = {}  # key -> {data, ts}
        self._tick_cache: dict[str, dict[str, Any]] = {}  # symbol -> latest tick
        self._tick_subscriptions: set[str] = set()
        self._symbol_cache: Optional[dict[str, Any]] = None
        self._symbol_cache_ts: float = 0

        # Cache TTLs
        self.candle_ttl = 60  # seconds
        self.symbol_ttl = 300  # 5 minutes

    @property
    def connected(self) -> bool:
        return self._connected and self._ws is not None

    async def connect(self) -> None:
        """Connect to Deriv WebSocket API."""
        self._should_run = True
        await self._do_connect()

    async def _do_connect(self) -> None:
        """Internal connection handler."""
        url = f"{DERIV_WS_URL}?app_id={self.app_id}"
        try:
            self._ws = await websockets.connect(url, ping_interval=30, ping_timeout=10)
            self._connected = True
            self._recv_task = asyncio.create_task(self._recv_loop())
            logger.info("Connected to Deriv WebSocket API (app_id=%s)", self.app_id)
        except Exception as e:
            logger.error("Failed to connect to Deriv: %s", e)
            self._connected = False
            if self._should_run:
                self._schedule_reconnect()

    async def disconnect(self) -> None:
        """Disconnect from Deriv WebSocket API."""
        self._should_run = False
        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()
        if self._ws:
            await self._ws.close()
            self._ws = None
        self._connected = False
        self._tick_subscriptions.clear()
        # Resolve any pending futures with errors
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(ConnectionError("Disconnected"))
        self._pending.clear()
        logger.info("Disconnected from Deriv")

    def _schedule_reconnect(self) -> None:
        """Schedule a reconnection attempt."""
        if not self._should_run:
            return
        if self._reconnect_task and not self._reconnect_task.done():
            return
        self._reconnect_task = asyncio.create_task(self._reconnect())

    async def _reconnect(self) -> None:
        """Reconnect with exponential backoff."""
        delay = 1
        while self._should_run and not self._connected:
            logger.info("Reconnecting to Deriv in %ds...", delay)
            await asyncio.sleep(delay)
            try:
                await self._do_connect()
                if self._connected:
                    # Re-subscribe to ticks
                    for symbol in list(self._tick_subscriptions):
                        self._tick_subscriptions.discard(symbol)
                        asyncio.create_task(self._subscribe_tick(symbol))
                    return
            except Exception as e:
                logger.error("Reconnect failed: %s", e)
            delay = min(delay * 2, 30)

    async def _recv_loop(self) -> None:
        """Receive messages from WebSocket."""
        try:
            async for message in self._ws:
                try:
                    data = json.loads(message)
                    self._handle_message(data)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON from Deriv")
        except websockets.ConnectionClosed:
            logger.warning("Deriv WebSocket connection closed")
        except Exception as e:
            logger.error("Recv loop error: %s", e)
        finally:
            self._connected = False
            if self._should_run:
                self._schedule_reconnect()

    def _handle_message(self, data: dict[str, Any]) -> None:
        """Route incoming messages to pending futures or tick cache."""
        req_id = data.get("req_id")

        # Handle tick stream updates (no req_id for subscription pushes)
        if "tick" in data:
            tick = data["tick"]
            symbol = tick.get("symbol", "")
            self._tick_cache[symbol] = {
                "symbol": symbol,
                "quote": tick.get("quote"),
                "epoch": tick.get("epoch"),
                "ask": tick.get("ask"),
                "bid": tick.get("bid"),
            }

        # Resolve pending request futures
        if req_id is not None and req_id in self._pending:
            fut = self._pending.pop(req_id)
            if not fut.done():
                if "error" in data:
                    fut.set_exception(
                        RuntimeError(data["error"].get("message", "Deriv API error"))
                    )
                else:
                    fut.set_result(data)

    async def _send(self, payload: dict[str, Any], timeout: float = 10.0) -> dict[str, Any]:
        """Send a request and wait for the response."""
        if not self.connected:
            raise ConnectionError("Not connected to Deriv")

        self._req_id += 1
        req_id = self._req_id
        payload["req_id"] = req_id

        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = fut

        try:
            await self._ws.send(json.dumps(payload))
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise TimeoutError(f"Deriv request timed out (req_id={req_id})")

    # --- Public API Methods ---

    async def get_candles(
        self,
        symbol: str = "R_100",
        count: int = 100,
        granularity: int = 60,
    ) -> list[dict[str, Any]]:
        """Fetch historical OHLC candles.

        Args:
            symbol: Deriv symbol (e.g. R_100, R_50)
            count: Number of candles (max 5000)
            granularity: Candle period in seconds (60, 120, 300, 600, 900, 1800, 3600, etc.)
        """
        cache_key = f"{symbol}_{count}_{granularity}"
        cached = self._candle_cache.get(cache_key)
        if cached and (time.time() - cached["ts"]) < self.candle_ttl:
            return cached["data"]

        resp = await self._send({
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": count,
            "end": "latest",
            "granularity": granularity,
            "style": "candles",
        })

        candles_raw = resp.get("candles", [])
        candles = [
            {
                "open": float(c["open"]),
                "high": float(c["high"]),
                "low": float(c["low"]),
                "close": float(c["close"]),
                "timestamp": int(c["epoch"]),
            }
            for c in candles_raw
        ]

        self._candle_cache[cache_key] = {"data": candles, "ts": time.time()}
        return candles

    async def _subscribe_tick(self, symbol: str) -> None:
        """Subscribe to live tick stream for a symbol."""
        if symbol in self._tick_subscriptions:
            return
        try:
            await self._send({
                "ticks": symbol,
                "subscribe": 1,
            })
            self._tick_subscriptions.add(symbol)
            logger.info("Subscribed to tick stream for %s", symbol)
        except Exception as e:
            logger.error("Failed to subscribe to %s ticks: %s", symbol, e)

    async def get_tick(self, symbol: str = "R_100") -> Optional[dict[str, Any]]:
        """Get the latest tick for a symbol. Subscribes on first call."""
        if symbol not in self._tick_subscriptions:
            await self._subscribe_tick(symbol)
            # Wait briefly for the first tick to arrive
            for _ in range(10):
                if symbol in self._tick_cache:
                    break
                await asyncio.sleep(0.2)

        return self._tick_cache.get(symbol)

    async def get_symbols(self) -> list[dict[str, Any]]:
        """Get available trading symbols from Deriv, with cache."""
        if self._symbol_cache and (time.time() - self._symbol_cache_ts) < self.symbol_ttl:
            return self._symbol_cache["symbols"]

        try:
            resp = await self._send({"active_symbols": "brief", "product_type": "basic"})
            symbols_raw = resp.get("active_symbols", [])
            # Filter to synthetic indices (always available, best for demo)
            symbols = [
                {
                    "symbol": s["symbol"],
                    "display_name": s.get("display_name", s["symbol"]),
                    "market": s.get("market", "unknown"),
                    "market_display_name": s.get("market_display_name", ""),
                }
                for s in symbols_raw
                if s.get("market") == "synthetic_index"
            ]
            if symbols:
                self._symbol_cache = {"symbols": symbols}
                self._symbol_cache_ts = time.time()
                return symbols
        except Exception as e:
            logger.warning("Failed to fetch symbols from Deriv: %s", e)

        # Fallback to hardcoded list
        return DEFAULT_SYMBOLS
