import asyncio
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

IRC_WS_URL = "wss://irc-ws.chat.twitch.tv:443"

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


HypeCallback = Callable[["HypeEvent", list[str]], Awaitable[None] | None]


class ChatMonitor:
    def __init__(self, on_hype_spike: HypeCallback):
        self._on_hype_spike = on_hype_spike
        self._message_times: deque[float] = deque()
        self._keyword_scores: deque[tuple[float, int]] = deque()  # (time, score)
        self._recent_messages: deque[str] = deque(maxlen=config.AI_RECENT_MSG_COUNT)
        self._last_clip_time: float = 0.0

    # ── Hype detection ────────────────────────────────────────────────────────

    def _record_message(self, text: str) -> None:
        logger.info("chat: %s", text)
        now = time.monotonic()
        self._message_times.append(now)
        if text:
            self._recent_messages.append(text)

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
            self._maybe_trigger(HypeEvent(trigger, count, total_score, now), list(self._recent_messages))

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

    def _maybe_trigger(self, event: HypeEvent, recent_messages: list[str]) -> None:
        if event.timestamp - self._last_clip_time < config.HYPE_COOLDOWN_SECONDS:
            return
        self._last_clip_time = event.timestamp
        logger.info(
            "Hype spike! trigger=%s msgs=%d score=%d",
            event.trigger, event.message_count, event.keyword_score,
        )
        result = self._on_hype_spike(event, recent_messages)
        if asyncio.iscoroutine(result):
            asyncio.ensure_future(result)

    # ── IRC connection ────────────────────────────────────────────────────────

    async def _fetch_bot_login(self) -> str:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                "https://api.twitch.tv/helix/users",
                headers={
                    "Authorization": f"Bearer {config.TWITCH_ACCESS_TOKEN}",
                    "Client-Id": config.TWITCH_CLIENT_ID,
                },
            )
            r.raise_for_status()
            return r.json()["data"][0]["login"]

    async def run(self) -> None:
        self._ensure_valid_token()
        bot_login = await self._fetch_bot_login()
        channel = config.TWITCH_CHANNEL.lstrip("#").lower()

        while True:
            try:
                await self._connect(bot_login, channel)
            except Exception as exc:
                logger.warning("IRC WebSocket error: %s — reconnecting in 5s", exc)
                await asyncio.sleep(5)

    async def _connect(self, bot_login: str, channel: str) -> None:
        async with websockets.connect(IRC_WS_URL, ping_interval=None) as ws:
            logger.info("Connected to %s as %s, joining #%s", IRC_WS_URL, bot_login, channel)
            await ws.send("CAP REQ :twitch.tv/membership twitch.tv/tags twitch.tv/commands")
            await ws.send(f"PASS oauth:{config.TWITCH_ACCESS_TOKEN}")
            await ws.send(f"NICK {bot_login}")
            await ws.send(f"JOIN #{channel}")

            async for raw in ws:
                for line in raw.strip().split("\r\n"):
                    if line:
                        logger.debug("< %s", line)
                        if await self._handle_line(ws, line):
                            return  # reconnect requested

    async def _handle_line(self, ws, line: str) -> bool:
        """Handle one IRC line. Returns True if the caller should reconnect."""
        # Strip IRCv3 tags first so all checks below work on the plain IRC line
        if line.startswith("@"):
            _, _, line = line.partition(" ")

        if line.startswith("PING"):
            pong = line.replace("PING", "PONG", 1)
            await ws.send(pong)
            logger.debug("> %s", pong)
            return False

        if " NOTICE " in line:
            logger.warning("Twitch NOTICE: %s", line)
            return False

        if " RECONNECT" in line:
            logger.warning("Twitch requested reconnect")
            return True

        # :user!user@user.tmi.twitch.tv PRIVMSG #channel :message text
        if " PRIVMSG " not in line:
            return False

        _, _, rest = line.partition(" PRIVMSG ")
        _, _, text = rest.partition(" :")
        self._record_message(text)
        return False

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
