"""
Pulls video metadata for one or more TikTok profiles via yt-dlp's flat-playlist
extraction (no official API needed — yt-dlp scrapes the public profile page).

Unlike youtube_scraper.py, view/like/comment/share counts come back directly
from the flat-playlist listing itself, so no per-video follow-up request is
needed. Results across profiles are merged (deduped by video_id) into
data-collection/dataset/tiktok_metadata.csv and tiktok_metadata.json, so
repeated runs across different handles accumulate into the same dataset.

Usage:
  python data-collection/tiktok_scraper.py \\
      --handles stoffer.edits \\
      --output-dir data-collection/dataset \\
      --max-per-channel 500
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


@dataclass
class TikTokVideoMetadata:
    video_id: str
    channel_id: str
    channel_title: str
    title: str
    published_at: str
    view_count: int
    like_count: int
    comment_count: int
    repost_count: int
    duration_seconds: int
    url: str


def _to_iso8601(timestamp: int | None) -> str:
    if not timestamp:
        return ""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def collect_handle(handle: str, max_results: int) -> list[TikTokVideoMetadata]:
    """Flat-playlist extraction returns per-video stats directly — TikTok's
    profile page embeds them, unlike YouTube where playlistItems only gives
    IDs and a separate videos.list call is needed for stats."""
    profile_url = f"https://www.tiktok.com/@{handle}"
    ydl_opts = {
        "extract_flat": True,
        "quiet": True,
        "no_warnings": True,
        "playlistend": max_results,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(profile_url, download=False)

    entries = [e for e in (info.get("entries") or []) if e]
    records = []
    for entry in entries:
        records.append(
            TikTokVideoMetadata(
                video_id=entry["id"],
                channel_id=entry.get("uploader_id") or handle,
                channel_title=entry.get("uploader") or handle,
                title=(entry.get("title") or entry.get("description") or "").strip(),
                published_at=_to_iso8601(entry.get("timestamp")),
                view_count=int(entry.get("view_count") or 0),
                like_count=int(entry.get("like_count") or 0),
                comment_count=int(entry.get("comment_count") or 0),
                repost_count=int(entry.get("repost_count") or 0),
                duration_seconds=int(entry.get("duration") or 0),
                url=entry.get("url") or f"https://www.tiktok.com/@{handle}/video/{entry['id']}",
            )
        )
    logger.info("%s: %d videos found", handle, len(records))
    return records


def _load_existing(output_dir: Path) -> list[TikTokVideoMetadata]:
    json_path = output_dir / "tiktok_metadata.json"
    if not json_path.exists():
        return []
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return [TikTokVideoMetadata(**item) for item in data]


def _write_outputs(records: list[TikTokVideoMetadata], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "tiktok_metadata.json"
    json_path.write_text(json.dumps([asdict(r) for r in records], indent=2), encoding="utf-8")

    csv_path = output_dir / "tiktok_metadata.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in fields(TikTokVideoMetadata)])
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))

    logger.info("Wrote %d records to %s and %s", len(records), csv_path, json_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--handles", nargs="+", required=True, help="TikTok handles, without the @")
    parser.add_argument("--output-dir", default="data-collection/dataset")
    parser.add_argument("--max-per-channel", type=int, default=500)
    args = parser.parse_args()

    new_records: list[TikTokVideoMetadata] = []
    for handle in args.handles:
        handle = handle.lstrip("@")
        try:
            records = collect_handle(handle, args.max_per_channel)
        except Exception as exc:
            logger.error("Failed to collect handle %s: %s", handle, exc)
            continue
        new_records.extend(records)

    output_dir = Path(args.output_dir)
    merged = {r.video_id: r for r in _load_existing(output_dir)}
    merged.update({r.video_id: r for r in new_records})

    _write_outputs(list(merged.values()), output_dir)
    logger.info(
        "This run: %d records across %d handle(s). Dataset total: %d records.",
        len(new_records), len(args.handles), len(merged),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
