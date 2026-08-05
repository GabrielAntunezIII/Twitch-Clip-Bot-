"""
Computes the same audio-spike/scene-change/transcript/sentiment features as
extract_features.py, but for fixed-length WINDOWS of a source VOD (built by
training/alignment/build_windows.py) rather than whole downloaded clips.

Reuses extract_features() / VideoFeatures from extract_features.py
unchanged -- the only difference is how the .mp4 lands in videos_dir: here
it's a precise time-slice of a source video pulled via
`yt-dlp --download-sections`, rather than the whole file.

Usage:
  python training/feature-extraction/extract_window_features.py --sample-size 300
"""

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

TRAINING_ROOT = Path(__file__).resolve().parent.parent  # training/
REPO_ROOT = TRAINING_ROOT.parent                          # true repo root
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(TRAINING_ROOT / "feature-extraction"))
import config  # noqa: E402
from dataclasses import asdict  # noqa: E402
from extract_features import (  # noqa: E402
    SCENE_CHANGE_THRESHOLD,
    AUDIO_SPIKE_DB,
    extract_features,
)

WINDOWS_METADATA_PATH = TRAINING_ROOT / "data-collection" / "dataset" / "togi_windows_metadata.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "dataset" / "togi_windows"


def _format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def download_window(
    window_id: str, url: str, start_seconds: float, end_seconds: float, videos_dir: Path, sleep_seconds: float
) -> Path | None:
    out_path = videos_dir / f"{window_id}.mp4"
    if out_path.exists():
        logger.debug("Already downloaded: %s", window_id)
        return out_path

    section = f"*{_format_timestamp(start_seconds)}-{_format_timestamp(end_seconds)}"
    cmd = [
        "yt-dlp",
        "-f", "mp4[height<=720]/mp4/best",
        "--download-sections", section,
        "-o", str(out_path),
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Sleep after every real network hit (success or failure) -- not on cache
    # hits above -- since it's the request volume/rate that trips YouTube's
    # bot detection, not the local feature-extraction work.
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    if not out_path.exists():
        logger.warning("yt-dlp failed for window %s:\n%s", window_id, result.stderr.strip()[-500:])
        if "Sign in to confirm you" in result.stderr:
            logger.error("Hit YouTube bot detection at window %s -- stopping early to avoid making it worse.", window_id)
            raise SystemExit(2)
        return None
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Only process up to N windows that don't already have features (default: all remaining). "
             "Safe to re-run repeatedly in small batches -- already-extracted windows are skipped.",
    )
    parser.add_argument("--windows-path", default=str(WINDOWS_METADATA_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--scene-threshold", type=float, default=SCENE_CHANGE_THRESHOLD)
    parser.add_argument("--spike-db", type=float, default=AUDIO_SPIKE_DB)
    parser.add_argument("--whisper-model", default=config.WHISPER_MODEL)
    parser.add_argument(
        "--sleep-seconds", type=float, default=4.0,
        help="Delay after every yt-dlp invocation. YouTube's bot detection appears to be volume/rate-triggered, "
             "so this matters more than any single window's download time.",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=15,
        help="Write features.json to disk after this many newly-extracted windows, so a crash or bot-detection "
             "abort only loses a few minutes of work instead of the whole run.",
    )
    args = parser.parse_args()

    windows_path = Path(args.windows_path)
    if not windows_path.exists():
        logger.error("No windows metadata found at %s — run training/alignment/build_windows.py first", windows_path)
        sys.exit(1)

    windows = json.loads(windows_path.read_text(encoding="utf-8"))

    output_dir = Path(args.output_dir)
    features_json_path = output_dir / "features.json"
    existing_features = []
    already_done_ids = set()
    if features_json_path.exists():
        existing_features = json.loads(features_json_path.read_text(encoding="utf-8"))
        already_done_ids = {row["video_id"] for row in existing_features}
        logger.info("Found %d already-extracted windows in %s -- these will be skipped", len(already_done_ids), features_json_path)

    remaining_windows = [w for w in windows if w["video_id"] not in already_done_ids]
    if args.sample_size:
        windows = remaining_windows[: args.sample_size]
    else:
        windows = remaining_windows
    logger.info(
        "Processing %d windows this run (%d positive, %d negative); %d remain unprocessed after this run",
        len(windows), sum(w["label"] == 1 for w in windows), sum(w["label"] == 0 for w in windows),
        len(remaining_windows) - len(windows),
    )

    import whisper  # imported here so --help doesn't pull in torch/whisper unnecessarily

    logger.info("Loading Whisper model '%s'...", args.whisper_model)
    whisper_model = whisper.load_model(args.whisper_model)

    output_dir = Path(args.output_dir)
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    merged_by_id = {row["video_id"]: row for row in existing_features}
    csv_path = output_dir / "features.csv"

    def _flush() -> None:
        merged = list(merged_by_id.values())
        features_json_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        if merged:
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
                writer.writeheader()
                writer.writerows(merged)
        logger.info("Checkpoint: features.json now has %d rows.", len(merged))

    n_new = 0
    n_since_checkpoint = 0
    try:
        for row in windows:
            window_id = row["video_id"]
            if download_window(window_id, row["url"], row["start_seconds"], row["end_seconds"], videos_dir, args.sleep_seconds) is None:
                continue
            features = extract_features(
                window_id, row["channel_id"], videos_dir, args.scene_threshold, args.spike_db, whisper_model
            )
            if features:
                record = asdict(features)
                record["label"] = row["label"]
                merged_by_id[window_id] = record
                n_new += 1
                n_since_checkpoint += 1
                logger.info(
                    "%s (label=%d): duration=%.1fs audio_spikes=%d scene_changes=%d words=%d keyword_score=%d sentiment=%.2f",
                    window_id, row["label"], features.duration_seconds, features.audio_spike_count,
                    features.scene_change_count, features.transcript_word_count,
                    features.transcript_keyword_score, features.sentiment_compound,
                )
                if n_since_checkpoint >= args.checkpoint_every:
                    _flush()
                    n_since_checkpoint = 0
    finally:
        # Always flush on the way out -- including SystemExit from a bot-detection
        # abort -- so an interrupted run never loses more than checkpoint_every
        # windows' worth of work.
        _flush()

    logger.info(
        "Extracted features for %d new windows this run. features.json now has %d rows total (%d before this run).",
        n_new, len(merged_by_id), len(existing_features),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
