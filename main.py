import asyncio
import logging

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

    await capture.start()
    await uploader.start()

    async def on_hype_spike(event: HypeEvent) -> None:
        logger.info(
            "Hype spike! trigger=%s msgs=%d score=%d — clipping",
            event.trigger, event.message_count, event.keyword_score,
        )

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
