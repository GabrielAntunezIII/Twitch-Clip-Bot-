import asyncio
import logging

from bot.ai.validator import ClipValidator
from bot.capture.stream_capture import StreamCapture
from bot.process.video_processor import VideoProcessor
from bot.twitch.chat_monitor import ChatMonitor, HypeEvent
from bot.upload.tiktok_uploader import TikTokUploader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


async def main() -> None:
    capture = StreamCapture()
    processor = VideoProcessor()
    uploader = TikTokUploader()
    validator = ClipValidator()

    await capture.start()
    await uploader.start()

    async def on_hype_spike(event: HypeEvent, recent_messages: list[str]) -> None:
        logger.info(
            "Hype spike! trigger=%s msgs=%d score=%d — validating with AI",
            event.trigger, event.message_count, event.keyword_score,
        )

        result = await validator.validate(
            trigger=event.trigger,
            message_count=event.message_count,
            keyword_score=event.keyword_score,
            recent_messages=recent_messages,
        )
        logger.info(
            "AI validation: worth_clipping=%s confidence=%.2f reason=%s",
            result.worth_clipping, result.confidence, result.reason,
        )
        if not result.worth_clipping:
            return

        raw_clip = await capture.clip()
        if not raw_clip:
            return

        tiktok_clip = await processor.process(raw_clip)
        if not tiktok_clip:
            return

        success = await uploader.upload(tiktok_clip)
        if success:
            logger.info("Uploaded to TikTok: %s", tiktok_clip.name)
            tiktok_clip.unlink(missing_ok=True)

    monitor = ChatMonitor(on_hype_spike=on_hype_spike)
    try:
        await monitor.run()
    finally:
        await capture.stop()
        await uploader.stop()


if __name__ == "__main__":
    asyncio.run(main())
