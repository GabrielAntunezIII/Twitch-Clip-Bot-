"""
Turns confident clip->source matches (alignment/cross_reference.py) into a
labeled window dataset: fixed-length (45s, matching config.CLIP_DURATION_BEFORE
+ CLIP_DURATION_AFTER -- the same span the live bot actually clips) positive
windows centered on each matched clip, plus sampled negative windows drawn
from elsewhere in the same source video.

Output schema is deliberately compatible with model-training/train.py's
--label-source explicit mode: each row already carries a binary `label`,
so no engagement-percentile derivation is needed downstream.

Usage:
  python alignment/build_windows.py --min-score 0.15
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
MATCHES_PATH = REPO_ROOT / "data-collection" / "dataset" / "togi_clip_matches.json"
SOURCES_PATH = REPO_ROOT / "data-collection" / "dataset" / "togi_sources_metadata.json"
OUTPUT_PATH = REPO_ROOT / "data-collection" / "dataset" / "togi_windows_metadata.json"

WINDOW_DURATION = config.CLIP_DURATION_BEFORE + config.CLIP_DURATION_AFTER
NEG_BUFFER_SECONDS = 60  # minimum gap between a negative window and any positive window


def _positive_windows(matches: list[dict]) -> list[dict]:
    windows = []
    for m in matches:
        clip_center = (m["offset_start_seconds"] + m["offset_end_seconds"]) / 2
        start = max(0.0, clip_center - WINDOW_DURATION / 2)
        windows.append(
            {
                "source_video_id": m["matched_source_video_id"],
                "source_channel_id": m["matched_source_channel_id"],
                "start_seconds": start,
                "end_seconds": start + WINDOW_DURATION,
                "label": 1,
                "source_clip_id": m["clip_video_id"],
                "match_score": m["match_score"],
            }
        )
    return windows


def _overlaps_any(start: float, end: float, intervals: list[tuple[float, float]], buffer_s: float) -> bool:
    return any(start < (e + buffer_s) and end > (s - buffer_s) for s, e in intervals)


def _negative_windows(
    positives_by_source: dict[str, list[dict]],
    source_duration: dict[str, int],
    negative_ratio: float,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    negatives = []
    for source_id, positives in positives_by_source.items():
        duration = source_duration.get(source_id)
        if not duration or duration < WINDOW_DURATION * 2:
            continue
        positive_intervals = [(p["start_seconds"], p["end_seconds"]) for p in positives]
        target = int(round(len(positives) * negative_ratio))
        max_start = duration - WINDOW_DURATION
        attempts = 0
        found = 0
        while found < target and attempts < target * 50:
            attempts += 1
            start = rng.uniform(0, max_start)
            end = start + WINDOW_DURATION
            if _overlaps_any(start, end, positive_intervals, NEG_BUFFER_SECONDS):
                continue
            negatives.append(
                {
                    "source_video_id": source_id,
                    "source_channel_id": positives[0]["source_channel_id"],
                    "start_seconds": start,
                    "end_seconds": end,
                    "label": 0,
                    "source_clip_id": None,
                    "match_score": None,
                }
            )
            positive_intervals.append((start, end))  # avoid stacking negatives on each other too
            found += 1
    return negatives


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matches-path", default=str(MATCHES_PATH))
    parser.add_argument("--sources-path", default=str(SOURCES_PATH))
    parser.add_argument("--output-path", default=str(OUTPUT_PATH))
    parser.add_argument("--min-score", type=float, default=0.15, help="Drop matches below this normalized correlation score")
    parser.add_argument("--negative-ratio", type=float, default=1.5, help="Negative windows sampled per positive window")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    matches = json.loads(Path(args.matches_path).read_text(encoding="utf-8"))
    sources = json.loads(Path(args.sources_path).read_text(encoding="utf-8"))
    source_by_id = {s["video_id"]: s for s in sources}
    source_duration = {s["video_id"]: s["duration_seconds"] for s in sources}

    confident = [m for m in matches if m["match_score"] >= args.min_score]
    logger.info(
        "%d/%d matches pass --min-score %.3f", len(confident), len(matches), args.min_score
    )

    positives = _positive_windows(confident)
    positives_by_source: dict[str, list[dict]] = {}
    for p in positives:
        positives_by_source.setdefault(p["source_video_id"], []).append(p)

    negatives = _negative_windows(positives_by_source, source_duration, args.negative_ratio, args.seed)

    rows = []
    for w in positives + negatives:
        source = source_by_id[w["source_video_id"]]
        start_int = int(w["start_seconds"])
        window_id = f"{w['source_video_id']}_{start_int}"
        rows.append(
            {
                "video_id": window_id,
                "channel_id": w["source_video_id"],
                "url": source["url"],
                "start_seconds": w["start_seconds"],
                "end_seconds": w["end_seconds"],
                "label": w["label"],
                "source_clip_id": w["source_clip_id"],
                "match_score": w["match_score"],
            }
        )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    logger.info(
        "Wrote %d windows (%d positive, %d negative) across %d source videos to %s",
        len(rows), len(positives), len(negatives), len(positives_by_source), output_path,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
