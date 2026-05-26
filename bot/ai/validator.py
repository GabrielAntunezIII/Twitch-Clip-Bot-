import logging
from dataclasses import dataclass

import anthropic

import config

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a Twitch clip quality detector for a bot that auto-uploads to TikTok.

Given a snapshot of chat messages around a hype spike, decide whether the moment
is genuinely worth clipping and uploading.

Clip-worthy moments:
- Big plays, clutch moments, near-misses, impressive skill
- Streamer does something funny, surprising, or emotional
- Community in-joke explodes in chat (lots of relevant emotes/phrases)
- Viewer count milestones, raids, large donations

NOT clip-worthy:
- Generic chat spam / bot flooding (all same message, no variety)
- Low-effort meme floods unrelated to on-stream content
- False positive from a single user spamming
- Routine gameplay with unremarkable chat

Respond with JSON only, no markdown fences:
{"worth_clipping": true|false, "confidence": 0.0-1.0, "reason": "one sentence"}"""


@dataclass
class ValidationResult:
    worth_clipping: bool
    confidence: float
    reason: str


class ClipValidator:
    def __init__(self) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)

    async def validate(
        self,
        trigger: str,
        message_count: int,
        keyword_score: int,
        recent_messages: list[str],
    ) -> ValidationResult:
        sample = recent_messages[-40:] if len(recent_messages) > 40 else recent_messages
        chat_block = "\n".join(f"  {m}" for m in sample) or "  (no messages captured)"

        user_content = (
            f"Hype trigger: {trigger}\n"
            f"Messages in window: {message_count}\n"
            f"Keyword score: {keyword_score}\n\n"
            f"Recent chat (up to 40 messages):\n{chat_block}\n\n"
            "Is this worth clipping?"
        )

        try:
            response = await self._client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=128,
                system=[
                    {
                        "type": "text",
                        "text": _SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_content}],
            )
        except Exception as exc:
            logger.warning("AI validation failed (%s) — defaulting to clip", exc)
            return ValidationResult(worth_clipping=True, confidence=0.5, reason="validation unavailable")

        raw = response.content[0].text.strip()
        try:
            import json
            data = json.loads(raw)
            return ValidationResult(
                worth_clipping=bool(data["worth_clipping"]),
                confidence=float(data.get("confidence", 0.5)),
                reason=str(data.get("reason", "")),
            )
        except Exception:
            logger.warning("Could not parse AI response: %r", raw)
            return ValidationResult(worth_clipping=True, confidence=0.5, reason="parse error")
