import asyncio
import logging

from bot.twitch.chat_monitor import ChatMonitor, HypeEvent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


async def on_hype_spike(event: HypeEvent) -> None:
    # Placeholder — will trigger stream capture + AI validation
    logger.info(
        "on_hype_spike: trigger=%s msgs=%d score=%d",
        event.trigger, event.message_count, event.keyword_score,
    )


async def main() -> None:
    monitor = ChatMonitor(on_hype_spike=on_hype_spike)
    await monitor.run()


if __name__ == "__main__":
    asyncio.run(main())
