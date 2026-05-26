import os
from dotenv import load_dotenv

load_dotenv()

TWITCH_CLIENT_ID = os.environ["TWITCH_CLIENT_ID"]
TWITCH_CLIENT_SECRET = os.environ["TWITCH_CLIENT_SECRET"]
TWITCH_ACCESS_TOKEN = os.environ["TWITCH_ACCESS_TOKEN"]
TWITCH_BROADCASTER_ID = os.environ["TWITCH_BROADCASTER_ID"]
TWITCH_CHANNEL = os.environ["TWITCH_CHANNEL"]

# Hype detection tuning
HYPE_WINDOW_SECONDS = 10        # rolling window for message rate + keyword scores
HYPE_RATE_THRESHOLD = 20        # raw message count in window to trigger
HYPE_SCORE_THRESHOLD = 15       # weighted keyword score in window to trigger
HYPE_COOLDOWN_SECONDS = 60      # minimum gap between clips
EVENTSUB_KEEPALIVE_BUFFER = 10  # extra seconds before declaring keepalive timeout

# Stream capture
STREAM_QUALITY = "best"
SEGMENT_DURATION = 8            # seconds per rolling buffer segment
BUFFER_DURATION = 120           # total seconds of buffer to keep on disk
CLIP_DURATION_BEFORE = 35       # seconds before spike to include in clip
CLIP_DURATION_AFTER = 10        # seconds after spike to wait before cutting
CLIPS_DIR = "clips"             # output directory for finished clips

# Video processing
WHISPER_MODEL = "base"          # tiny | base | small | medium | large
WORDS_PER_CAPTION = 3           # words per caption cue
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
BLUR_RADIUS = 30                # background blur strength
CAPTION_FONT = "Arial Black"
CAPTION_SIZE = 88               # pt at 1080x1920
CAPTION_MARGIN_BOTTOM = 180     # px from bottom of frame
