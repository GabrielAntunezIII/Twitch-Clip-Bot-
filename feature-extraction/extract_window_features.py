"""
Computes the same audio-spike/scene-change/transcript/sentiment features as
extract_features.py, but for fixed-length WINDOWS of a source VOD (built by
alignment/build_windows.py) rather than whole downloaded clips.

Reuses extract_features() / VideoFeatures from extract_features.py
unchanged -- the only difference is how the .mp4 lands in videos_dir: here
it's a precise time-slice of a source video pulled via
`yt-dlp --download-sections`, rather than the whole file.

Usage:
  python feature-extraction/extract_window_features.py --sample-size 300
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "feature-extraction"))
import config  # noqa: E402
from extract_features import (  # noqa: E402
    SCENE_CHANGE_THRESHOLD,
    AUDIO_SPIKE_DB,
    extract_features,
    _write_outputs,
)

WINDOWS_METADATA_PATH = REPO_ROOT / "data-collection" / "dataset" / "togi_windows_metadata.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "dataset" / "togi_windows"


def _format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def download_window(window_id: str, url: str, start_seconds: float, end_seconds: float, videos_dir: Path) -> Path | None:
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
    if not out_path.exists():
        logger.warning("yt-dlp failed for window %s:\n%s", window_id, result.stderr.strip()[-500:])
        return None
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample-size", type=int, default=None, help="Only process the first N windows (default: all)")
    parser.add_argument("--windows-path", default=str(WINDOWS_METADATA_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--scene-threshold", type=float, default=SCENE_CHANGE_THRESHOLD)
    parser.add_argument("--spike-db", type=float, default=AUDIO_SPIKE_DB)
    parser.add_argument("--whisper-model", default=config.WHISPER_MODEL)
    args = parser.parse_args()

    windows_path = Path(args.windows_path)
    if not windows_path.exists():
        logger.error("No windows metadata found at %s — run alignment/build_windows.py first", windows_path)
        sys.exit(1)

    windows = json.loads(windows_path.read_text(encoding="utf-8"))
    if args.sample_size:
        windows = windows[: args.sample_size]
    logger.info(
        "Processing %d windows (%d positive, %d negative)",
        len(windows), sum(w["label"] == 1 for w in windows), sum(w["label"] == 0 for w in windows),
    )

    import whisper  # imported here so --help doesn't pull in torch/whisper unnecessarily

    logger.info("Loading Whisper model '%s'...", args.whisper_model)
    whisper_model = whisper.load_model(args.whisper_model)

    output_dir = Path(args.output_dir)
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    records = []
    labels_by_id = {}
    for row in windows:
        window_id = row["video_id"]
        if download_window(window_id, row["url"], row["start_seconds"], row["end_seconds"], videos_dir) is None:
            continue
        features = extract_features(
            window_id, row["channel_id"], videos_dir, args.scene_threshold, args.spike_db, whisper_model
        )
        if features:
            records.append(features)
            labels_by_id[window_id] = row["label"]
            logger.info(
                "%s (label=%d): duration=%.1fs audio_spikes=%d scene_changes=%d words=%d keyword_score=%d sentiment=%.2f",
                window_id, row["label"], features.duration_seconds, features.audio_spike_count,
                features.scene_change_count, features.transcript_word_count,
                features.transcript_keyword_score, features.sentiment_compound,
            )

    _write_outputs(records, output_dir)

    # Fold labels directly into the features file so train.py's --label-source explicit
    # mode doesn't need a second join against the (much larger) windows metadata file.
    features_json_path = output_dir / "features.json"
    features = json.loads(features_json_path.read_text(encoding="utf-8"))
    for row in features:
        row["label"] = labels_by_id[row["video_id"]]
    features_json_path.write_text(json.dumps(features, indent=2), encoding="utf-8")

    logger.info("Extracted features for %d/%d windows", len(records), len(windows))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
