"""
Turns confident clip->source matches (training/alignment/cross_reference.py) into a
labeled window dataset: fixed-length (45s, matching config.CLIP_DURATION_BEFORE
+ CLIP_DURATION_AFTER -- the same span the live bot actually clips) positive
windows centered on each matched clip, plus sampled negative windows drawn
from elsewhere in the same source video.

A matched clip only becomes a positive window if it also performed -- the
program this dataset is meant to compete in pays clippers by views, so a
window is only as "clip-worthy" as the views the clip cut from it actually
got. Performance is views_per_day (raw view_count divided by days since the
clip was posted, since the pool spans mid-2024 to now and raw view_count
would just reward old clips having more time to accumulate views), ranked
by percentile across all confident matches. Matches below --min-views-percentile
are dropped rather than relabeled 0 -- an underperforming clip doesn't mean
the moment was bad (could be a weak edit/caption/post-time instead), so it's
excluded as ambiguous rather than treated as a confident negative.

Output schema is deliberately compatible with training/model-training/train.py's
--label-source explicit mode: each row already carries a binary `label`,
so no engagement-percentile derivation is needed downstream.

Usage:
  python training/alignment/build_windows.py --min-score 0.15 --min-views-percentile 0.5
"""

import argparse
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parent.parent  # training/
REPO_ROOT = TRAINING_ROOT.parent                          # true repo root
sys.path.insert(0, str(REPO_ROOT))
import config  # noqa: E402

logger = logging.getLogger(__name__)

MATCHES_PATH = TRAINING_ROOT / "data-collection" / "dataset" / "togi_clip_matches.json"
SOURCES_PATH = TRAINING_ROOT / "data-collection" / "dataset" / "togi_sources_metadata.json"
CLIPS_METADATA_PATH = TRAINING_ROOT / "data-collection" / "dataset" / "tiktok_metadata.json"
OUTPUT_PATH = TRAINING_ROOT / "data-collection" / "dataset" / "togi_windows_metadata.json"

WINDOW_DURATION = config.CLIP_DURATION_BEFORE + config.CLIP_DURATION_AFTER
NEG_BUFFER_SECONDS = 60  # minimum gap between a negative window and any positive window


def _views_per_day(clips_metadata_path: Path) -> dict[str, float]:
    clips = json.loads(clips_metadata_path.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc)
    result = {}
    for c in clips:
        published = datetime.fromisoformat(c["published_at"])
        days = max((now - published).total_seconds() / 86400, 1.0)
        result[c["video_id"]] = c["view_count"] / days
    return result


def _percentile_ranks(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=values.get)
    n = len(ordered)
    return {video_id: (i + 1) / n for i, video_id in enumerate(ordered)}


def _positive_windows(matches: list[dict], views_per_day: dict[str, float], views_percentile: dict[str, float]) -> list[dict]:
    windows = []
    for m in matches:
        clip_center = (m["offset_start_seconds"] + m["offset_end_seconds"]) / 2
        start = max(0.0, clip_center - WINDOW_DURATION / 2)
        clip_id = m["clip_video_id"]
        windows.append(
            {
                "source_video_id": m["matched_source_video_id"],
                "source_channel_id": m["matched_source_channel_id"],
                "start_seconds": start,
                "end_seconds": start + WINDOW_DURATION,
                "label": 1,
                "source_clip_id": clip_id,
                "match_score": m["match_score"],
                "views_per_day": views_per_day[clip_id],
                "views_percentile": views_percentile[clip_id],
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
                    "views_per_day": None,
                    "views_percentile": None,
                }
            )
            positive_intervals.append((start, end))  # avoid stacking negatives on each other too
            found += 1
    return negatives


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matches-path", default=str(MATCHES_PATH))
    parser.add_argument("--sources-path", default=str(SOURCES_PATH))
    parser.add_argument("--clips-metadata-path", default=str(CLIPS_METADATA_PATH))
    parser.add_argument("--output-path", default=str(OUTPUT_PATH))
    parser.add_argument("--min-score", type=float, default=0.15, help="Drop matches below this normalized correlation score")
    parser.add_argument(
        "--min-views-percentile", type=float, default=0.5,
        help="Drop matches whose clip's views_per_day percentile (among confident matches) falls below this -- "
             "an underperforming clip is ambiguous, not necessarily a bad moment, so it's excluded rather than labeled 0",
    )
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

    views_per_day = _views_per_day(Path(args.clips_metadata_path))
    n_before_view_filter = len(confident)
    confident = [m for m in confident if m["clip_video_id"] in views_per_day]
    if len(confident) < n_before_view_filter:
        logger.warning(
            "%d/%d confident matches had no clip view data — dropped",
            n_before_view_filter - len(confident), n_before_view_filter,
        )

    percentiles = _percentile_ranks({m["clip_video_id"]: views_per_day[m["clip_video_id"]] for m in confident})
    top_performing = [m for m in confident if percentiles[m["clip_video_id"]] >= args.min_views_percentile]
    logger.info(
        "%d/%d confident matches (with view data) pass --min-views-percentile %.2f",
        len(top_performing), len(confident), args.min_views_percentile,
    )

    positives = _positive_windows(top_performing, views_per_day, percentiles)
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
                "views_per_day": w["views_per_day"],
                "views_percentile": w["views_percentile"],
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
