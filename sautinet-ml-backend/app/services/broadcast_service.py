"""
SentiKenya WebSocket Broadcast Service
========================================
Manages WebSocket connections and broadcasts processed sentiment data
to connected dashboard clients in real-time.
"""

import asyncio
import json
import logging
from typing import Set, Dict, Any, Optional
from datetime import datetime
from collections import deque

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections with channel-based broadcasting.

    Channels:
    - "feed": All processed posts (live feed)
    - "counties": County-level aggregated updates
    - "alerts": High-priority sentiment alerts
    - "stats": System statistics updates
    """

    def __init__(self, max_buffer: int = 100):
        self._connections: Dict[str, Set[WebSocket]] = {
            "feed": set(),
            "counties": set(),
            "alerts": set(),
            "stats": set(),
            "all": set(),  # Subscribes to everything
        }
        self._message_buffer: deque = deque(maxlen=max_buffer)
        self._stats = {
            "total_connections": 0,
            "total_messages_sent": 0,
            "peak_connections": 0,
        }

    async def connect(self, websocket: WebSocket, channels: list = None):
        """Accept a WebSocket connection and subscribe to channels."""
        await websocket.accept()

        if not channels:
            channels = ["all"]

        for channel in channels:
            if channel in self._connections:
                self._connections[channel].add(websocket)

        self._stats["total_connections"] += 1
        active = self._get_active_count()
        self._stats["peak_connections"] = max(self._stats["peak_connections"], active)

        logger.info(f"WebSocket connected (channels: {channels}, active: {active})")

        # Send recent buffer to new connection
        for msg in list(self._message_buffer)[-10:]:
            try:
                await websocket.send_json(msg)
            except Exception:
                break

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket from all channels."""
        for channel_connections in self._connections.values():
            channel_connections.discard(websocket)
        logger.info(f"WebSocket disconnected (active: {self._get_active_count()})")

    async def broadcast(self, channel: str, data: dict):
        """
        Broadcast a message to all subscribers of a channel.

        Also sends to "all" channel subscribers.
        """
        message = {
            "channel": channel,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Buffer the message
        self._message_buffer.append(message)

        # Get target connections
        targets = set()
        if channel in self._connections:
            targets.update(self._connections[channel])
        targets.update(self._connections.get("all", set()))

        if not targets:
            return

        # Broadcast to all targets
        disconnected = set()
        for websocket in targets:
            try:
                await websocket.send_json(message)
                self._stats["total_messages_sent"] += 1
            except Exception:
                disconnected.add(websocket)

        # Clean up disconnected
        for ws in disconnected:
            self.disconnect(ws)

    async def broadcast_processed_post(self, processed_post_data: dict):
        """Broadcast a processed post to the feed channel."""
        await self.broadcast("feed", {
            "event": "new_post",
            **processed_post_data,
        })

    async def broadcast_county_update(self, county_data: dict):
        """Broadcast county sentiment update."""
        await self.broadcast("counties", {
            "event": "county_update",
            **county_data,
        })

    async def broadcast_alert(self, alert_data: dict):
        """Broadcast a sentiment alert (spike, anomaly, etc.)."""
        await self.broadcast("alerts", {
            "event": "sentiment_alert",
            **alert_data,
        })

    async def broadcast_stats(self, stats_data: dict):
        """Broadcast system statistics update."""
        await self.broadcast("stats", {
            "event": "stats_update",
            **stats_data,
        })

    def _get_active_count(self) -> int:
        """Count unique active connections across all channels."""
        all_connections = set()
        for connections in self._connections.values():
            all_connections.update(connections)
        return len(all_connections)

    def get_stats(self) -> dict:
        """Return connection manager statistics."""
        return {
            **self._stats,
            "active_connections": self._get_active_count(),
            "buffer_size": len(self._message_buffer),
            "channels": {
                name: len(conns) for name, conns in self._connections.items()
            },
        }
