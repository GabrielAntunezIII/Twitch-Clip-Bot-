import logging

import anthropic

import config

logger = logging.getLogger(__name__)

_SYSTEM = (
    "Generate a short, punchy, clickbait TikTok title for a Twitch clip. "
    "Maximum 60 characters. No hashtags. No surrounding quotes. "
    "Make it dramatic or intriguing — something viewers can't scroll past. "
    "Respond with the title text only."
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
            title = resp.content[0].text.strip().strip("\"'")
            logger.info("Generated title: %r", title)
            return title[:80]
        except Exception as exc:
            logger.warning("Title generation failed (%s) — using reason as fallback", exc)
            return reason[:60]
