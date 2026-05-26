import asyncio
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable

import httpx
import websockets

import config
from bot.twitch.auth import refresh_access_token, validate_token

logger = logging.getLogger(__name__)

EVENTSUB_WS_URL = "wss://eventsub.twitch.tv/ws"
EVENTSUB_API_URL = "https://api.twitch.tv/helix/eventsub/subscriptions"

# Weighted keyword scores — higher = stronger hype signal
_KEYWORD_SCORES: dict[str, int] = {
    "pogchamp": 4, "pog": 3, "clip": 4, "clip it": 5, "clipped": 4,
    "letsgo": 3, "lets go": 3, "let's go": 3,
    "omg": 1, "wtf": 1, "lol": 1, "lmao": 1,
    "insane": 2, "no way": 2, "what": 1,
}


@dataclass
class HypeEvent:
    trigger: str          # "rate" | "score"
    message_count: int    # messages in current window
    keyword_score: int    # cumulative keyword score in current window
    timestamp: float      # monotonic time


HypeCallback = Callable[[HypeEvent], Awaitable[None] | None]


class ChatMonitor:
    def __init__(self, on_hype_spike: HypeCallback):
        self._on_hype_spike = on_hype_spike
        self._message_times: deque[float] = deque()
        self._keyword_scores: deque[tuple[float, int]] = deque()  # (time, score)
        self._last_clip_time: float = 0.0
        self._session_id: str | None = None
        self._bot_user_id: str | None = None
        self._keepalive_timeout: int = 10

    # ── Hype detection ────────────────────────────────────────────────────────

    def _record_message(self, text: str) -> None:
        now = time.monotonic()
        self._message_times.append(now)

        score = self._score_message(text)
        if score:
            self._keyword_scores.append((now, score))

        self._prune_window(now)

        count = len(self._message_times)
        total_score = sum(s for _, s in self._keyword_scores)

        logger.debug("chat | window_msgs=%d keyword_score=%d", count, total_score)

        trigger: str | None = None
        if count >= config.HYPE_RATE_THRESHOLD:
            trigger = "rate"
        elif total_score >= config.HYPE_SCORE_THRESHOLD:
            trigger = "score"

        if trigger:
            self._maybe_trigger(HypeEvent(trigger, count, total_score, now))

    def _prune_window(self, now: float) -> None:
        cutoff = now - config.HYPE_WINDOW_SECONDS
        while self._message_times and self._message_times[0] < cutoff:
            self._message_times.popleft()
        while self._keyword_scores and self._keyword_scores[0][0] < cutoff:
            self._keyword_scores.popleft()

    @staticmethod
    def _score_message(text: str) -> int:
        lower = text.lower()
        return max(
            (score for kw, score in _KEYWORD_SCORES.items() if kw in lower),
            default=0,
        )

    def _maybe_trigger(self, event: HypeEvent) -> None:
        if event.timestamp - self._last_clip_time < config.HYPE_COOLDOWN_SECONDS:
            return
        self._last_clip_time = event.timestamp
        logger.info(
            "Hype spike! trigger=%s msgs=%d score=%d",
            event.trigger, event.message_count, event.keyword_score,
        )
        result = self._on_hype_spike(event)
        if asyncio.iscoroutine(result):
            asyncio.ensure_future(result)

    # ── EventSub subscription ─────────────────────────────────────────────────

    async def _fetch_bot_user_id(self) -> str:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                "https://api.twitch.tv/helix/users",
                headers={
                    "Authorization": f"Bearer {config.TWITCH_ACCESS_TOKEN}",
                    "Client-Id": config.TWITCH_CLIENT_ID,
                },
            )
            r.raise_for_status()
            return r.json()["data"][0]["id"]

    async def _subscribe(self, session_id: str) -> None:
        if not self._bot_user_id:
            self._bot_user_id = await self._fetch_bot_user_id()

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
                "user_id": self._bot_user_id,
            },
            "transport": {"method": "websocket", "session_id": session_id},
        }
        async with httpx.AsyncClient() as client:
            r = await client.post(EVENTSUB_API_URL, headers=headers, json=payload)
            r.raise_for_status()
            logger.info("Subscribed to channel.chat.message as user %s", self._bot_user_id)

    # ── Connection lifecycle ──────────────────────────────────────────────────

    async def run(self) -> None:
        self._ensure_valid_token()
        url = EVENTSUB_WS_URL
        while True:
            try:
                url = await self._connect(url)
            except Exception as exc:
                logger.warning("WebSocket error: %s — reconnecting in 5s", exc)
                url = EVENTSUB_WS_URL
                await asyncio.sleep(5)

    async def _connect(self, url: str) -> str:
        """Connect to EventSub WebSocket. Returns a new URL if Twitch requests a reconnect."""
        async with websockets.connect(url) as ws:
            logger.info("Connected to %s", url)
            reconnect_url: str | None = None

            async def _read():
                nonlocal reconnect_url
                async for raw in ws:
                    result = await self._handle(json.loads(raw))
                    if result:
                        reconnect_url = result
                        return

            keepalive_deadline = time.monotonic() + self._keepalive_timeout + config.EVENTSUB_KEEPALIVE_BUFFER
            read_task = asyncio.ensure_future(_read())

            while not read_task.done():
                if time.monotonic() > keepalive_deadline:
                    logger.warning("Keepalive timeout — reconnecting")
                    read_task.cancel()
                    break
                await asyncio.sleep(1)

            if reconnect_url:
                return reconnect_url

        return EVENTSUB_WS_URL

    async def _handle(self, msg: dict) -> str | None:
        """Process one EventSub message. Returns a reconnect URL if Twitch requested one."""
        msg_type = msg.get("metadata", {}).get("message_type")

        if msg_type == "session_welcome":
            session = msg["payload"]["session"]
            self._session_id = session["id"]
            self._keepalive_timeout = session.get("keepalive_timeout_seconds", 10)
            await self._subscribe(self._session_id)

        elif msg_type == "notification":
            event = msg.get("payload", {}).get("event", {})
            text = event.get("message", {}).get("text", "")
            self._record_message(text)

        elif msg_type == "session_keepalive":
            logger.debug("keepalive received")

        elif msg_type == "session_reconnect":
            return msg["payload"]["session"]["reconnect_url"]

        return None

    @staticmethod
    def _ensure_valid_token() -> None:
        if validate_token(config.TWITCH_ACCESS_TOKEN):
            return
        logger.info("Access token expired — refreshing")
        refresh_token = os.environ.get("TWITCH_REFRESH_TOKEN", "")
        if not refresh_token:
            raise SystemExit(
                "Token expired and no TWITCH_REFRESH_TOKEN in .env. Run: python -m bot.twitch.auth"
            )
        new_token = refresh_access_token(
            config.TWITCH_CLIENT_ID, config.TWITCH_CLIENT_SECRET, refresh_token
        )
        os.environ["TWITCH_ACCESS_TOKEN"] = new_token
        config.TWITCH_ACCESS_TOKEN = new_token
