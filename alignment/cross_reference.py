"""
Cross-references clips from a clipping page against a pool of candidate
source videos to find exactly where each clip was cut from.

For each clip, computes an onset-strength audio fingerprint (see
audio_fingerprint.py) and brute-force correlates it against every cached
candidate source fingerprint. Source fingerprints are cached to disk, so
this is cheap to re-run once the pool has been fingerprinted once —
correlation itself (FFT-based, scipy) is fast even for a full VOD-length
envelope; downloading/decoding audio is the real cost, and only happens
once per video_id.

No date-based pruning: YouTube's flat-playlist channel listing (used by
vod_scraper.py) doesn't expose upload timestamps, so candidates can't be
filtered by publish-date proximity to clips. Brute-forcing all clip x
candidate pairs is still tractable since correlation is cheap once
fingerprints are cached (~230k pairs for the full TOGI dataset estimated
under an hour).

Usage:
  # Validation run on a small sample first -- see the plan's Phase B checkpoint
  python alignment/cross_reference.py --limit 25

  # Full run
  python alignment/cross_reference.py
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.signal import correlate

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from alignment.audio_fingerprint import FPS, get_envelope  # noqa: E402

logger = logging.getLogger(__name__)

CLIPS_METADATA_PATH = REPO_ROOT / "data-collection" / "dataset" / "tiktok_metadata.json"
SOURCES_METADATA_PATH = REPO_ROOT / "data-collection" / "dataset" / "togi_sources_metadata.json"
CACHE_DIR = REPO_ROOT / "alignment" / "cache"
OUTPUT_PATH = REPO_ROOT / "data-collection" / "dataset" / "togi_clip_matches.json"

MIN_CANDIDATE_MULTIPLE = 1.0  # candidate must be at least as long as the clip to search within it


def _normalize(env: np.ndarray) -> np.ndarray:
    std = env.std()
    if std < 1e-8:
        return env - env.mean()
    return (env - env.mean()) / std


def best_offset(candidate_env: np.ndarray, clip_env: np.ndarray) -> tuple[int, float] | None:
    """Returns (best_start_frame, normalized_score) for aligning clip_env within
    candidate_env, or None if candidate is shorter than the clip."""
    if len(candidate_env) < len(clip_env):
        return None
    c = _normalize(candidate_env)
    q = _normalize(clip_env)
    corr = correlate(c, q, mode="valid", method="fft")
    corr /= len(q)
    best_idx = int(np.argmax(corr))
    return best_idx, float(corr[best_idx])


def match_clip(clip: dict, sources: list[dict], audio_cache: Path, envelope_cache: Path) -> dict | None:
    clip_env = get_envelope(clip["video_id"], clip["url"], audio_cache, envelope_cache)
    if clip_env is None:
        logger.warning("%s: could not fingerprint clip audio", clip["video_id"])
        return None

    best_match = None
    for source in sources:
        source_env = get_envelope(source["video_id"], source["url"], audio_cache, envelope_cache)
        if source_env is None:
            continue
        result = best_offset(source_env, clip_env)
        if result is None:
            continue
        best_idx, score = result
        if best_match is None or score > best_match["match_score"]:
            offset_start = best_idx / FPS
            best_match = {
                "clip_video_id": clip["video_id"],
                "matched_source_video_id": source["video_id"],
                "matched_source_channel_id": source["channel_id"],
                "offset_start_seconds": offset_start,
                "offset_end_seconds": offset_start + clip["duration_seconds"],
                "match_score": score,
            }
    return best_match


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips-path", default=str(CLIPS_METADATA_PATH))
    parser.add_argument("--sources-path", default=str(SOURCES_METADATA_PATH))
    parser.add_argument("--output-path", default=str(OUTPUT_PATH))
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--title-filter", default="togi", help="Only cross-reference clips whose title contains this (case-insensitive)")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N clips -- use for the Phase B validation checkpoint")
    args = parser.parse_args()

    clips = json.loads(Path(args.clips_path).read_text(encoding="utf-8"))
    sources = json.loads(Path(args.sources_path).read_text(encoding="utf-8"))

    if args.title_filter:
        clips = [c for c in clips if args.title_filter.lower() in c["title"].lower()]
    if args.limit:
        clips = clips[: args.limit]

    logger.info("Cross-referencing %d clips against %d candidate source videos", len(clips), len(sources))

    cache_dir = Path(args.cache_dir)
    audio_cache = cache_dir / "audio"
    envelope_cache = cache_dir / "envelopes"

    output_path = Path(args.output_path)
    existing = {}
    if output_path.exists():
        existing = {m["clip_video_id"]: m for m in json.loads(output_path.read_text(encoding="utf-8"))}

    matches = dict(existing)
    for i, clip in enumerate(clips):
        if clip["video_id"] in existing:
            continue
        match = match_clip(clip, sources, audio_cache, envelope_cache)
        if match:
            matches[clip["video_id"]] = match
            logger.info(
                "[%d/%d] %s -> %s @ %.1fs (score=%.3f)",
                i + 1, len(clips), clip["video_id"], match["matched_source_video_id"],
                match["offset_start_seconds"], match["match_score"],
            )
        else:
            logger.warning("[%d/%d] %s: no match found", i + 1, len(clips), clip["video_id"])

        if (i + 1) % 10 == 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(list(matches.values()), indent=2), encoding="utf-8")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(list(matches.values()), indent=2), encoding="utf-8")

    scores = [m["match_score"] for m in matches.values()]
    if scores:
        logger.info(
            "Wrote %d matches to %s. Score distribution: min=%.3f p25=%.3f median=%.3f p75=%.3f max=%.3f",
            len(matches), output_path, min(scores),
            float(np.percentile(scores, 25)), float(np.median(scores)),
            float(np.percentile(scores, 75)), max(scores),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
