"""
Downloads a sample of videos from the scraped dataset via yt-dlp and extracts
audio-loudness-spike, scene-change, duration, and transcript/sentiment features
for model training.

Reads training/data-collection/dataset/metadata.json, samples videos round-robin across
channels (so the batch isn't dominated by whichever channel has the most
records), downloads each with yt-dlp into <output_dir>/videos/, then runs
ffmpeg/ffprobe/Whisper to compute:
  - duration_seconds        (ffprobe)
  - audio_spike_count       (ebur128 momentary loudness jumps >= --spike-db)
  - scene_change_count      (ffmpeg scene-detection filter, score > --scene-threshold)
  - transcript_word_count   (Whisper transcription word count)
  - transcript_keyword_score (hype-keyword hits, weighted — see _KEYWORD_SCORES)
  - sentiment_compound/pos/neg (VADER lexicon sentiment over the transcript)

Writes one row per successfully processed video to <output_dir>/features.csv
and features.json, keyed by video_id so it can be joined back to
metadata.json for labels (views/likes/etc).

Usage:
  python training/feature-extraction/extract_features.py --sample-size 25
"""

import argparse
import csv
import json
import logging
import random
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

TRAINING_ROOT = Path(__file__).resolve().parent.parent  # training/
REPO_ROOT = TRAINING_ROOT.parent                          # true repo root
sys.path.insert(0, str(REPO_ROOT))
import config  # noqa: E402

METADATA_PATH = TRAINING_ROOT / "data-collection" / "dataset" / "metadata.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "dataset"

SCENE_CHANGE_THRESHOLD = 0.2  # ffmpeg 'scene' score cutoff for a cut/motion spike — see
# training/feature-extraction/README.md for the 84-video sample that set this (0.4 zeroed ~45% of clips)
AUDIO_SPIKE_DB = 6.0          # minimum jump in momentary loudness to count as a spike

# Hype keywords for spoken transcripts — adapted from bot/twitch/chat_monitor.py's
# _KEYWORD_SCORES, dropping chat-only terms ("pogchamp", "clip it") nobody says aloud.
_KEYWORD_SCORES: dict[str, int] = {
    "let's go": 3, "lets go": 3, "oh my god": 2, "no way": 2,
    "insane": 2, "crazy": 2, "what": 1, "omg": 1, "wtf": 1,
}

_sentiment_analyzer = SentimentIntensityAnalyzer()


@dataclass
class VideoFeatures:
    video_id: str
    channel_id: str
    duration_seconds: float
    audio_spike_count: int
    scene_change_count: int
    transcript_word_count: int
    transcript_keyword_score: int
    sentiment_compound: float
    sentiment_pos: float
    sentiment_neg: float


def _sample_videos(metadata: list[dict], n: int, seed: int) -> list[dict]:
    """Round-robin sample across channels so no single prolific channel dominates
    the test batch."""
    by_channel: dict[str, list[dict]] = {}
    for row in metadata:
        by_channel.setdefault(row["channel_id"], []).append(row)

    rng = random.Random(seed)
    for rows in by_channel.values():
        rng.shuffle(rows)

    sampled = []
    channel_iters = {cid: iter(rows) for cid, rows in by_channel.items()}
    while len(sampled) < n and channel_iters:
        for cid in list(channel_iters):
            try:
                sampled.append(next(channel_iters[cid]))
            except StopIteration:
                del channel_iters[cid]
                continue
            if len(sampled) >= n:
                break
    return sampled


def download_video(video_id: str, url: str, videos_dir: Path) -> Path | None:
    out_path = videos_dir / f"{video_id}.mp4"
    if out_path.exists():
        logger.debug("Already downloaded: %s", video_id)
        return out_path

    cmd = [
        "yt-dlp",
        "-f", "mp4[height<=720]/mp4/best",
        "-o", str(out_path),
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("yt-dlp failed for %s:\n%s", video_id, result.stderr.strip()[-500:])
        return None
    return out_path


def _ffprobe_duration(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def _count_scene_changes(path: Path, threshold: float) -> int:
    """Counts frames ffmpeg's scene-detection filter flags above `threshold` —
    a proxy for cut frequency / motion intensity."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "info",
        "-i", str(path),
        "-vf", f"select='gt(scene,{threshold})',metadata=print",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return len(re.findall(r"lavfi\.scene_score", result.stderr))


def _count_audio_spikes(path: Path, spike_db: float) -> int:
    """Runs the ebur128 momentary-loudness filter and counts local jumps of
    >= spike_db between consecutive samples — a proxy for hype/reaction beats.

    framelog must be "info" (not "verbose") to match -loglevel info below —
    ebur128's verbose level (40) is more detailed than info (32), so
    -loglevel info silently drops verbose-level frame logs.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "info",
        "-i", str(path),
        "-filter_complex", "ebur128=peak=none:framelog=info",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    levels = [float(m) for m in re.findall(r"M:\s*(-?\d+(?:\.\d+)?)", result.stderr)]
    return sum(
        1
        for prev, curr in zip(levels, levels[1:])
        if curr > -70 and (curr - prev) >= spike_db
    )


def _transcribe(whisper_model, path: Path) -> str:
    """Whisper transcription, text only (no word timestamps — those are only
    needed for caption rendering in bot/process/video_processor.py, not here)."""
    result = whisper_model.transcribe(str(path), word_timestamps=False)
    return result.get("text", "").strip()


def _keyword_score(text: str) -> int:
    lower = text.lower()
    return sum(score for kw, score in _KEYWORD_SCORES.items() if kw in lower)


def extract_features(
    video_id: str,
    channel_id: str,
    videos_dir: Path,
    scene_threshold: float,
    spike_db: float,
    whisper_model,
) -> VideoFeatures | None:
    path = videos_dir / f"{video_id}.mp4"
    if not path.exists():
        return None

    transcript = ""
    try:
        transcript = _transcribe(whisper_model, path)
    except Exception as exc:
        logger.warning("%s: Whisper transcription failed (%s) — transcript features will be 0", video_id, exc)

    sentiment = _sentiment_analyzer.polarity_scores(transcript) if transcript else {"compound": 0.0, "pos": 0.0, "neg": 0.0}

    return VideoFeatures(
        video_id=video_id,
        channel_id=channel_id,
        duration_seconds=_ffprobe_duration(path),
        audio_spike_count=_count_audio_spikes(path, spike_db),
        scene_change_count=_count_scene_changes(path, scene_threshold),
        transcript_word_count=len(transcript.split()),
        transcript_keyword_score=_keyword_score(transcript),
        sentiment_compound=sentiment["compound"],
        sentiment_pos=sentiment["pos"],
        sentiment_neg=sentiment["neg"],
    )


def _write_outputs(records: list[VideoFeatures], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "features.json"
    json_path.write_text(json.dumps([asdict(r) for r in records], indent=2), encoding="utf-8")

    csv_path = output_dir / "features.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in fields(VideoFeatures)])
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))

    logger.info("Wrote %d feature rows to %s and %s", len(records), csv_path, json_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=25)
    parser.add_argument("--metadata-path", default=str(METADATA_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scene-threshold", type=float, default=SCENE_CHANGE_THRESHOLD)
    parser.add_argument("--spike-db", type=float, default=AUDIO_SPIKE_DB)
    parser.add_argument("--whisper-model", default=config.WHISPER_MODEL, help="tiny | base | small | medium | large")
    args = parser.parse_args()

    metadata_path = Path(args.metadata_path)
    if not metadata_path.exists():
        logger.error("No metadata found at %s — run training/data-collection/youtube_scraper.py or tiktok_scraper.py first", metadata_path)
        sys.exit(1)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sample = _sample_videos(metadata, args.sample_size, args.seed)
    logger.info("Sampled %d videos across %d channels", len(sample), len({r["channel_id"] for r in sample}))

    import whisper  # imported here so --help doesn't pull in torch/whisper unnecessarily

    logger.info("Loading Whisper model '%s'...", args.whisper_model)
    whisper_model = whisper.load_model(args.whisper_model)

    output_dir = Path(args.output_dir)
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for row in sample:
        video_id = row["video_id"]
        url = row.get("url") or f"https://www.youtube.com/watch?v={video_id}"
        if download_video(video_id, url, videos_dir) is None:
            continue
        features = extract_features(
            video_id, row["channel_id"], videos_dir, args.scene_threshold, args.spike_db, whisper_model
        )
        if features:
            records.append(features)
            logger.info(
                "%s: duration=%.1fs audio_spikes=%d scene_changes=%d words=%d keyword_score=%d sentiment=%.2f",
                video_id, features.duration_seconds, features.audio_spike_count, features.scene_change_count,
                features.transcript_word_count, features.transcript_keyword_score, features.sentiment_compound,
            )

    _write_outputs(records, output_dir)
    logger.info("Extracted features for %d/%d sampled videos", len(records), len(sample))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
