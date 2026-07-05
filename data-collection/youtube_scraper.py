"""
Pulls video metadata for a list of YouTube channels via the YouTube Data API v3.

For each channel, walks its "uploads" playlist (cheaper on API quota than
search.list), fetches view/like counts and publish dates for every video, and
tags each one as a Short based on duration. Results are written to
data-collection/dataset/metadata.csv and metadata.json.

Downloading video files with yt-dlp is deliberately not wired up yet — get the
metadata validated first.

Usage:
  python data-collection/youtube_scraper.py \\
      --channels UCxxxx UCyyyy \\
      --output-dir data-collection/dataset \\
      --max-per-channel 50
"""

import argparse
import csv
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from googleapiclient.discovery import build

import config

logger = logging.getLogger(__name__)

SHORTS_MAX_DURATION_SECONDS = 180  # generous cutoff; actual Shorts are <= 60s today


@dataclass
class VideoMetadata:
    video_id: str
    channel_id: str
    channel_title: str
    subscriber_count: int
    title: str
    published_at: str
    view_count: int
    like_count: int
    duration_seconds: int
    is_short: bool


def _iso8601_duration_to_seconds(duration: str) -> int:
    """Parse an ISO 8601 duration like 'PT1M30S' into seconds."""
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration)
    if not match:
        return 0
    hours, minutes, seconds = (int(g) if g else 0 for g in match.groups())
    return hours * 3600 + minutes * 60 + seconds


class YouTubeScraper:
    """Pulls video metadata for a list of channels via the YouTube Data API."""

    def __init__(self, api_key: str) -> None:
        self._youtube = build("youtube", "v3", developerKey=api_key)

    def _channel_info(self, channel_id: str) -> tuple[str, int, str]:
        """Returns (channel_title, subscriber_count, uploads_playlist_id)."""
        resp = (
            self._youtube.channels()
            .list(part="snippet,statistics,contentDetails", id=channel_id)
            .execute()
        )
        items = resp.get("items", [])
        if not items:
            raise ValueError(f"Channel not found: {channel_id}")
        item = items[0]
        title = item["snippet"]["title"]
        subs = int(item["statistics"].get("subscriberCount", 0))
        uploads_playlist_id = item["contentDetails"]["relatedPlaylists"]["uploads"]
        return title, subs, uploads_playlist_id

    def _playlist_video_ids(self, playlist_id: str, max_results: int) -> list[str]:
        video_ids: list[str] = []
        page_token = None
        while len(video_ids) < max_results:
            resp = (
                self._youtube.playlistItems()
                .list(
                    part="contentDetails",
                    playlistId=playlist_id,
                    maxResults=min(50, max_results - len(video_ids)),
                    pageToken=page_token,
                )
                .execute()
            )
            video_ids.extend(item["contentDetails"]["videoId"] for item in resp.get("items", []))
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        return video_ids

    def _video_stats(self, video_ids: list[str]) -> list[dict]:
        """Fetch stats/contentDetails, batching in groups of 50 (API limit)."""
        results = []
        for i in range(0, len(video_ids), 50):
            batch = video_ids[i : i + 50]
            resp = (
                self._youtube.videos()
                .list(part="snippet,statistics,contentDetails", id=",".join(batch))
                .execute()
            )
            results.extend(resp.get("items", []))
        return results

    def collect_channel(self, channel_id: str, max_results: int) -> list[VideoMetadata]:
        channel_title, subscriber_count, uploads_playlist_id = self._channel_info(channel_id)
        video_ids = self._playlist_video_ids(uploads_playlist_id, max_results)

        records = []
        for item in self._video_stats(video_ids):
            duration = _iso8601_duration_to_seconds(item["contentDetails"]["duration"])
            stats = item["statistics"]
            records.append(
                VideoMetadata(
                    video_id=item["id"],
                    channel_id=channel_id,
                    channel_title=channel_title,
                    subscriber_count=subscriber_count,
                    title=item["snippet"]["title"],
                    published_at=item["snippet"]["publishedAt"],
                    view_count=int(stats.get("viewCount", 0)),
                    like_count=int(stats.get("likeCount", 0)),
                    duration_seconds=duration,
                    is_short=duration <= SHORTS_MAX_DURATION_SECONDS,
                )
            )
        logger.info("%s: %d videos found (%d Shorts)", channel_id, len(records), sum(r.is_short for r in records))
        return records


def _write_outputs(records: list[VideoMetadata], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "metadata.json"
    json_path.write_text(json.dumps([asdict(r) for r in records], indent=2), encoding="utf-8")

    csv_path = output_dir / "metadata.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in fields(VideoMetadata)])
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))

    logger.info("Wrote %d records to %s and %s", len(records), csv_path, json_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", nargs="+", required=True, help="YouTube channel IDs")
    parser.add_argument("--output-dir", default="data-collection/dataset")
    parser.add_argument("--max-per-channel", type=int, default=50)
    parser.add_argument(
        "--shorts-only", action="store_true", help="Drop videos longer than %ds" % SHORTS_MAX_DURATION_SECONDS
    )
    args = parser.parse_args()

    if not config.YOUTUBE_API_KEY:
        logger.error("YOUTUBE_API_KEY is not set — see .env.example")
        sys.exit(1)

    scraper = YouTubeScraper(config.YOUTUBE_API_KEY)

    all_records: list[VideoMetadata] = []
    for channel_id in args.channels:
        try:
            records = scraper.collect_channel(channel_id, args.max_per_channel)
        except Exception as exc:
            logger.error("Failed to collect channel %s: %s", channel_id, exc)
            continue
        all_records.extend(records)

    if args.shorts_only:
        all_records = [r for r in all_records if r.is_short]

    _write_outputs(all_records, Path(args.output_dir))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
