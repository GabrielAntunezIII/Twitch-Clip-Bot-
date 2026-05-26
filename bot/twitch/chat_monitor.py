import asyncio
import json
import logging
import os
import time
from collections import deque
from typing import Callable

import httpx
import websockets

import config
from bot.twitch.auth import validate_token, refresh_access_token

logger = logging.getLogger(__name__)

EVENTSUB_WS_URL = "wss://eventsub.twitch.tv/ws"
EVENTSUB_API_URL = "https://api.twitch.tv/helix/eventsub/subscriptions"

# Keywords that signal a hype moment in chat
HYPE_KEYWORDS = {"pog", "pogchamp", "clip", "clip it", "lol", "omg", "wtf", "lets go", "letsgo"}


class ChatMonitor:
    def __init__(self, on_hype_spike: Callable[[], None]):
        self._on_hype_spike = on_hype_spike
        self._message_times: deque[float] = deque()
        self._last_clip_time: float = 0.0
        self._session_id: str | None = None

    def _record_message(self, text: str) -> None:
        now = time.monotonic()
        self._message_times.append(now)
        self._prune_window(now)

        count = len(self._message_times)
        is_hype = self._is_hype_message(text)
        logger.debug("chat msg | window=%d hype_keyword=%s", count, is_hype)

        if count >= config.HYPE_SPIKE_THRESHOLD or is_hype:
            self._maybe_trigger(now)

    def _prune_window(self, now: float) -> None:
        cutoff = now - config.HYPE_WINDOW_SECONDS
        while self._message_times and self._message_times[0] < cutoff:
            self._message_times.popleft()

    def _is_hype_message(self, text: str) -> bool:
        lower = text.lower()
        return any(kw in lower for kw in HYPE_KEYWORDS)

    def _maybe_trigger(self, now: float) -> None:
        if now - self._last_clip_time < config.HYPE_COOLDOWN_SECONDS:
            return
        self._last_clip_time = now
        logger.info("Hype spike detected — triggering clip")
        self._on_hype_spike()

    async def _subscribe(self, session_id: str) -> None:
        headers = {
            "Authorization": f"Bearer {config.TWITCH_ACCESS_TOKEN}",
            "Client-Id": config.TWITCH_CLIENT_ID,
            "Content-Type": "application/json",
        }
        payload = {
            "type": "channel.chat.message",
            "version": "1",
            "condition": {
                "broadcaster_user_id": config.TWITCH_BROADCASTER_ID,
                "user_id": config.TWITCH_BROADCASTER_ID,
            },
            "transport": {"method": "websocket", "session_id": session_id},
        }
        async with httpx.AsyncClient() as client:
            r = await client.post(EVENTSUB_API_URL, headers=headers, json=payload)
            r.raise_for_status()
            logger.info("Subscribed to channel.chat.message (status %s)", r.status_code)

    async def run(self) -> None:
        self._ensure_valid_token()
        while True:
            try:
                await self._connect()
            except Exception as exc:
                logger.warning("WebSocket error: %s — reconnecting in 5s", exc)
                await asyncio.sleep(5)

    @staticmethod
    def _ensure_valid_token() -> None:
        if validate_token(config.TWITCH_ACCESS_TOKEN):
            return
        logger.info("Access token expired — refreshing")
        refresh_token = os.environ.get("TWITCH_REFRESH_TOKEN", "")
        if not refresh_token:
            raise SystemExit("Token expired and no TWITCH_REFRESH_TOKEN in .env. Run: python -m bot.twitch.auth")
        new_token = refresh_access_token(config.TWITCH_CLIENT_ID, config.TWITCH_CLIENT_SECRET, refresh_token)
        os.environ["TWITCH_ACCESS_TOKEN"] = new_token
        config.TWITCH_ACCESS_TOKEN = new_token

    async def _connect(self) -> None:
        async with websockets.connect(EVENTSUB_WS_URL) as ws:
            logger.info("Connected to Twitch EventSub WebSocket")
            async for raw in ws:
                msg = json.loads(raw)
                await self._handle(msg)

    async def _handle(self, msg: dict) -> None:
        metadata = msg.get("metadata", {})
        msg_type = metadata.get("message_type")

        if msg_type == "session_welcome":
            self._session_id = msg["payload"]["session"]["id"]
            await self._subscribe(self._session_id)

        elif msg_type == "notification":
            event = msg.get("payload", {}).get("event", {})
            text = event.get("message", {}).get("text", "")
            self._record_message(text)

        elif msg_type == "session_keepalive":
            pass

        elif msg_type == "session_reconnect":
            reconnect_url = msg["payload"]["session"]["reconnect_url"]
            logger.info("Reconnect requested: %s", reconnect_url)
