"""
Phase 4 — Video processor

Takes a raw .ts clip and produces a TikTok-ready vertical MP4:
  1. Blurred 9:16 background with the original 16:9 video centered over it
  2. Whisper transcription → bold word-chunked captions burned in
  3. Audio loudness normalisation (EBU R128)
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import config

logger = logging.getLogger(__name__)


# ── Caption data ──────────────────────────────────────────────────────────────

@dataclass
class _Word:
    text: str
    start: float
    end: float


def _chunk_words(words: list[_Word], per_cue: int) -> list[list[_Word]]:
    """Group words into cues of at most `per_cue` words."""
    return [words[i : i + per_cue] for i in range(0, len(words), per_cue)]


# ── ASS subtitle generation ───────────────────────────────────────────────────

def _ts(seconds: float) -> str:
    """Format seconds as ASS timestamp H:MM:SS.cc"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _write_ass(words: list[_Word], out_path: Path) -> None:
    w, h = config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT

    header = f"""\
[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 1

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{config.CAPTION_FONT},{config.CAPTION_SIZE},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,2,0,1,5,2,2,40,40,{config.CAPTION_MARGIN_BOTTOM},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    for cue in _chunk_words(words, config.WORDS_PER_CAPTION):
        start = _ts(cue[0].start)
        end = _ts(cue[-1].end)
        text = " ".join(w.text.upper() for w in cue)
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.debug("ASS captions written: %s (%d cues)", out_path.name, len(lines) - 1)


# ── Whisper transcription (runs in thread pool) ───────────────────────────────

def _transcribe_sync(clip_path: Path, model_name: str) -> list[_Word]:
    import whisper  # imported here so startup isn't blocked if whisper isn't installed

    logger.info("Loading Whisper model '%s'...", model_name)
    model = whisper.load_model(model_name)
    logger.info("Transcribing %s", clip_path.name)
    result = model.transcribe(str(clip_path), word_timestamps=True)

    words: list[_Word] = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            word = w["word"].strip()
            if word:
                words.append(_Word(word, w["start"], w["end"]))
    logger.info("Transcription complete: %d words", len(words))
    return words


# ── ffmpeg render ─────────────────────────────────────────────────────────────

async def _render(
    raw: Path,
    ass: Path | None,
    out: Path,
) -> bool:
    w, h = config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT

    # Background: scale source to fill full 9:16 frame, then blur
    bg = f"[0:v]scale=-1:{h},crop={w}:{h},boxblur=luma_radius={config.BLUR_RADIUS}:luma_power=5[bg]"

    # Foreground: scale source to fit width, keep aspect ratio
    fg = f"[0:v]scale={w}:-2[fg]"

    # Overlay fg centered vertically over bg
    overlay = "[bg][fg]overlay=0:(H-h)/2"

    # Optionally burn in captions
    if ass:
        escaped = str(ass).replace("\\", "/").replace(":", "\\:")
        video_filter = f"{overlay},subtitles='{escaped}'[vout]"
    else:
        video_filter = f"{overlay}[vout]"

    filter_complex = f"{bg};{fg};{video_filter}"

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-i", str(raw),
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "0:a",
        # audio loudness normalisation (EBU R128)
        "-af", "loudnorm=I=-14:TP=-1:LRA=11",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        "-y", str(out),
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()

    if proc.returncode != 0:
        logger.error("ffmpeg render failed:\n%s", stderr.decode().strip())
        return False

    logger.info("Rendered: %s (%.1f MB)", out.name, out.stat().st_size / 1e6)
    return True


# ── Public interface ──────────────────────────────────────────────────────────

class VideoProcessor:
    """Converts a raw .ts clip into a TikTok-ready vertical MP4."""

    async def process(self, raw_clip: Path) -> Path | None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = Path(config.CLIPS_DIR) / f"tiktok_{ts}.mp4"
        ass_path = raw_clip.with_suffix(".ass")

        # Transcribe in thread pool (CPU-bound)
        words: list[_Word] = []
        try:
            loop = asyncio.get_running_loop()
            words = await loop.run_in_executor(
                None, _transcribe_sync, raw_clip, config.WHISPER_MODEL
            )
        except Exception as exc:
            logger.warning("Whisper transcription failed (%s) — rendering without captions", exc)

        if words:
            _write_ass(words, ass_path)
        else:
            ass_path = None

        try:
            success = await _render(raw_clip, ass_path, out_path)
        finally:
            if ass_path and ass_path.exists():
                ass_path.unlink()

        if not success:
            return None

        raw_clip.unlink(missing_ok=True)
        return out_path
