import asyncio
import logging
import os
import shutil
import tempfile
import time
from collections import deque
from pathlib import Path

import config

logger = logging.getLogger(__name__)


class StreamCapture:
    """
    Maintains a rolling disk buffer of a live Twitch stream using streamlink + ffmpeg.

    Segments are written to a temp directory. Old segments beyond BUFFER_DURATION
    are pruned automatically. Call clip() to extract a finished clip file.
    """

    def __init__(self):
        self._segment_dir = Path(tempfile.mkdtemp(prefix="twitch_buf_"))
        self._segments: deque[tuple[float, Path]] = deque()  # (recorded_at, path)
        self._ffmpeg: asyncio.subprocess.Process | None = None
        self._watcher_task: asyncio.Task | None = None
        self._clip_lock = asyncio.Lock()

    # ── Startup ───────────────────────────────────────────────────────────────

    async def start(self) -> None:
        stream_url = await self._resolve_stream_url()
        logger.info("Stream URL resolved, starting capture buffer")
        Path(config.CLIPS_DIR).mkdir(exist_ok=True)
        await self._start_ffmpeg(stream_url)
        self._watcher_task = asyncio.create_task(self._watch_segments())

    async def _resolve_stream_url(self) -> str:
        proc = await asyncio.create_subprocess_exec(
            "streamlink",
            "--stream-url",
            f"twitch.tv/{config.TWITCH_CHANNEL}",
            config.STREAM_QUALITY,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"streamlink failed: {stderr.decode().strip()}"
            )
        url = stdout.decode().strip()
        if not url:
            raise RuntimeError("streamlink returned an empty URL — is the channel live?")
        return url

    async def _start_ffmpeg(self, stream_url: str) -> None:
        segment_pattern = str(self._segment_dir / "%Y%m%d_%H%M%S.ts")
        self._ffmpeg = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-hide_banner", "-loglevel", "warning",
            "-i", stream_url,
            "-c", "copy",
            "-f", "segment",
            "-segment_time", str(config.SEGMENT_DURATION),
            "-strftime", "1",
            "-reset_timestamps", "1",
            segment_pattern,
            stderr=asyncio.subprocess.PIPE,
        )
        logger.info("ffmpeg capture started (pid=%d)", self._ffmpeg.pid)

    # ── Segment tracking ──────────────────────────────────────────────────────

    async def _watch_segments(self) -> None:
        """Poll segment dir, register new files, prune old ones."""
        seen: set[Path] = set()
        while True:
            await asyncio.sleep(1)
            for path in sorted(self._segment_dir.glob("*.ts")):
                if path not in seen:
                    seen.add(path)
                    self._segments.append((time.monotonic(), path))
                    logger.debug("New segment: %s", path.name)
            self._prune_old_segments()

    def _prune_old_segments(self) -> None:
        cutoff = time.monotonic() - config.BUFFER_DURATION
        while self._segments and self._segments[0][0] < cutoff:
            _, old_path = self._segments.popleft()
            try:
                old_path.unlink()
            except FileNotFoundError:
                pass

    # ── Clip extraction ───────────────────────────────────────────────────────

    async def clip(self) -> Path | None:
        """
        Wait CLIP_DURATION_AFTER seconds (to capture post-spike reaction),
        then assemble the last CLIP_DURATION_BEFORE + CLIP_DURATION_AFTER seconds
        into a single .ts file in CLIPS_DIR.
        """
        async with self._clip_lock:
            logger.info(
                "Waiting %ds for post-spike content...", config.CLIP_DURATION_AFTER
            )
            await asyncio.sleep(config.CLIP_DURATION_AFTER)

            total_duration = config.CLIP_DURATION_BEFORE + config.CLIP_DURATION_AFTER
            segments = self._segments_for_duration(total_duration)

            if not segments:
                logger.warning("No buffered segments available for clip")
                return None

            return await self._concat_segments(segments)

    def _segments_for_duration(self, duration: float) -> list[Path]:
        """Return the most-recent segments covering at least `duration` seconds.

        The newest segment is excluded because ffmpeg is likely still writing to
        it; including a partial .ts corrupts the concat output.
        """
        now = time.monotonic()
        cutoff = now - duration
        candidates = [
            path
            for recorded_at, path in self._segments
            if recorded_at >= cutoff and path.exists()
        ]
        return candidates[:-1] if len(candidates) > 1 else candidates

    async def _concat_segments(self, segments: list[Path]) -> Path | None:
        concat_list = self._segment_dir / "concat.txt"
        concat_list.write_text(
            "\n".join(f"file '{p.resolve()}'" for p in segments)
        )

        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = Path(config.CLIPS_DIR) / f"clip_{ts}.ts"

        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-hide_banner", "-loglevel", "warning",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            str(out_path),
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()

        if proc.returncode != 0:
            logger.error("ffmpeg concat failed: %s", stderr.decode().strip())
            return None

        logger.info("Clip saved: %s (from %d segments)", out_path, len(segments))
        return out_path

    # ── Shutdown ──────────────────────────────────────────────────────────────

    async def stop(self) -> None:
        if self._watcher_task:
            self._watcher_task.cancel()
        if self._ffmpeg and self._ffmpeg.returncode is None:
            self._ffmpeg.terminate()
            await self._ffmpeg.wait()
        shutil.rmtree(self._segment_dir, ignore_errors=True)
        logger.info("Stream capture stopped, buffer cleaned up")
