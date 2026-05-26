import asyncio
import logging

from bot.twitch.chat_monitor import ChatMonitor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def on_hype_spike() -> None:
    # Placeholder — will trigger stream capture + AI validation
    logger.info("on_hype_spike called — clip pipeline not yet wired up")


async def main() -> None:
    monitor = ChatMonitor(on_hype_spike=on_hype_spike)
    await monitor.run()


if __name__ == "__main__":
    asyncio.run(main())
