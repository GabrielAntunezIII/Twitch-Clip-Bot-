import logging

import anthropic

import config

logger = logging.getLogger(__name__)

_SYSTEM = (
    "Generate a punchy 2-line TikTok title for a Twitch clip.\n"
    "Line 1: attention hook — 1-6 words with 1-2 emojis. Max 28 characters including emojis.\n"
    "Line 2: short teaser or reaction — 3-7 words with 1 emoji. Max 28 characters including emojis.\n"
    "Separate the two lines with a literal newline. No hashtags. No surrounding quotes.\n"
    "Match the tone to the clip — shocked, unhinged, funny, dramatic, or wholesome.\n"
    "Examples:\n"
    "😱 HE ACTUALLY SAID THAT\n💀 chat is not okay\n\n"
    "🔥 SHE STARTED CRYING LIVE\n😭 most emotional moment\n\n"
    "💀 BRO JUST GOT DESTROYED\n😂 streamer lost it\n\n"
    "Respond with the two lines only."
)


class TitleGenerator:
    def __init__(self) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)

    async def generate(self, reason: str, transcript: str) -> str:
        prompt = f"Clip reason: {reason}\nTranscript: {transcript or '(none)'}"
        try:
            resp = await self._client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=64,
                system=_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip().strip("\"'")
            lines = [l.strip() for l in raw.splitlines() if l.strip()]
            title = "\n".join(lines[:2])
            logger.info("Generated title: %r", title)
            return title
        except Exception as exc:
            logger.warning("Title generation failed (%s) — using reason as fallback", exc)
            return reason[:28]
