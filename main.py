import argparse
import asyncio
import logging
from logging.handlers import RotatingFileHandler

from bot.ai.validator import ClipValidator
from bot.capture.stream_capture import StreamCapture
from bot.process.video_processor import VideoProcessor
from bot.twitch.chat_monitor import ChatMonitor, HypeEvent
from bot.upload.tiktok_uploader import TikTokUploader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Dedicated clip log — one line per valid clip, file only
_clip_logger = logging.getLogger("clip_log")
_clip_logger.propagate = False
_clip_handler = RotatingFileHandler("bot.log", maxBytes=1_000_000, backupCount=10, encoding="utf-8")
_clip_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
_clip_logger.addHandler(_clip_handler)


async def main(dry_run: bool = False) -> None:
    capture = StreamCapture()
    processor = VideoProcessor()
    validator = ClipValidator()

    await capture.start()

    uploader = None
    if not dry_run:
        uploader = TikTokUploader()
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

        tiktok_clip = await processor.process(raw_clip, reason=result.reason)
        if not tiktok_clip:
            raw_clip.unlink(missing_ok=True)
            return

        size_mb = tiktok_clip.stat().st_size / 1e6
        _clip_logger.info(
            "CLIP | file=%s | size=%.1fMB | confidence=%.2f | reason=%s",
            tiktok_clip.name, size_mb, result.confidence, result.reason,
        )

        if dry_run:
            logger.info("[dry-run] Skipping TikTok upload: %s", tiktok_clip.name)
            return

        success = await uploader.upload(tiktok_clip)
        if success:
            logger.info("Uploaded to TikTok: %s", tiktok_clip.name)
        else:
            logger.warning("Upload failed — deleting clip: %s", tiktok_clip.name)
        tiktok_clip.unlink(missing_ok=True)

    monitor = ChatMonitor(on_hype_spike=on_hype_spike)
    try:
        await monitor.run()
    finally:
        await capture.stop()
        if uploader is not None:
            await uploader.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Skip TikTok upload entirely")
    args = parser.parse_args()
    asyncio.run(main(dry_run=args.dry_run))
