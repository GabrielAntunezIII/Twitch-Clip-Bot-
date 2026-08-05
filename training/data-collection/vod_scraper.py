"""
Pulls video metadata for one or more YouTube channels via yt-dlp's
flat-playlist extraction, for use as CANDIDATE SOURCE VIDEOS in
training/alignment/cross_reference.py — i.e. full-length uploads a short clip might
have been cut from, not the clips themselves.

Unlike youtube_scraper.py (which targets a channel's Shorts and needs
YOUTUBE_API_KEY for view/like stats), this only needs id/title/duration/
publish-date/url, all available directly from yt-dlp's flat-playlist
listing with no API key. Results across channels are merged (deduped by
video_id) into training/data-collection/dataset/<output-name>.csv/.json, so repeated
runs across different channel lists accumulate into the same dataset.

Usage:
  python training/data-collection/vod_scraper.py \\
      --channels UCbZkHHK3mX0p8LiYcLgRxng UCb2y3fAB0mt6Chk7fTko7eg \\
      --output-dir training/data-collection/dataset \\
      --output-name togi_sources
"""

import argparse
import csv
import json
import logging
import sys
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

import yt_dlp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logger = logging.getLogger(__name__)


@dataclass
class SourceVideoMetadata:
    video_id: str
    channel_id: str
    channel_title: str
    title: str
    published_at: str
    duration_seconds: int
    url: str


def _to_iso8601(timestamp: int | None) -> str:
    if not timestamp:
        return ""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def collect_channel(channel_id: str, max_results: int) -> list[SourceVideoMetadata]:
    channel_url = f"https://www.youtube.com/channel/{channel_id}/videos"
    ydl_opts = {
        "extract_flat": True,
        "quiet": True,
        "no_warnings": True,
        "playlistend": max_results,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)

    entries = [e for e in (info.get("entries") or []) if e]
    channel_title = info.get("channel") or info.get("uploader") or channel_id
    records = []
    for entry in entries:
        video_id = entry["id"]
        records.append(
            SourceVideoMetadata(
                video_id=video_id,
                channel_id=channel_id,
                channel_title=entry.get("channel") or channel_title,
                title=(entry.get("title") or "").strip(),
                published_at=_to_iso8601(entry.get("timestamp")),
                duration_seconds=int(entry.get("duration") or 0),
                url=entry.get("url") or f"https://www.youtube.com/watch?v={video_id}",
            )
        )
    logger.info("%s (%s): %d videos found", channel_title, channel_id, len(records))
    return records


def _load_existing(json_path: Path) -> list[SourceVideoMetadata]:
    if not json_path.exists():
        return []
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return [SourceVideoMetadata(**item) for item in data]


def _write_outputs(records: list[SourceVideoMetadata], output_dir: Path, output_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{output_name}_metadata.json"
    json_path.write_text(json.dumps([asdict(r) for r in records], indent=2), encoding="utf-8")

    csv_path = output_dir / f"{output_name}_metadata.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in fields(SourceVideoMetadata)])
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))

    logger.info("Wrote %d records to %s and %s", len(records), csv_path, json_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--channels", nargs="+", required=True, help="YouTube channel IDs (UCxxxx)")
    parser.add_argument("--output-dir", default="training/data-collection/dataset")
    parser.add_argument("--output-name", default="vod_sources", help="Basename for output files, e.g. 'togi_sources'")
    parser.add_argument("--max-per-channel", type=int, default=500)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    json_path = output_dir / f"{args.output_name}_metadata.json"

    new_records: list[SourceVideoMetadata] = []
    for channel_id in args.channels:
        try:
            records = collect_channel(channel_id, args.max_per_channel)
        except Exception as exc:
            logger.error("Failed to collect channel %s: %s", channel_id, exc)
            continue
        new_records.extend(records)

    merged = {r.video_id: r for r in _load_existing(json_path)}
    merged.update({r.video_id: r for r in new_records})

    _write_outputs(list(merged.values()), output_dir, args.output_name)
    logger.info(
        "This run: %d records across %d channel(s). Dataset total: %d records.",
        len(new_records), len(args.channels), len(merged),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
