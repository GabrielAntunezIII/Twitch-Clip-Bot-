# data-collection

Scrapes labeled training data from popular existing clip channels on YouTube.

- Pulls metadata (views, likes, publish date, duration, channel subscriber count) via
  the YouTube Data API for a given list of channel IDs, walking each channel's uploads
  playlist. Writes `dataset/metadata.csv` and `dataset/metadata.json`.
- Metadata + engagement numbers serve as a proxy label for "clip-worthy" when training
  the scoring model in `model-training/`.
- Downloading the underlying video files with `yt-dlp` is not wired up yet — metadata
  is being validated first (`--shorts-only` filters to Shorts-length videos once that
  looks right).

See `youtube_scraper.py` for the entry point. Requires `YOUTUBE_API_KEY` in `.env`
(loaded via `config.py`).
