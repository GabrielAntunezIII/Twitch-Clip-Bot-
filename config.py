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
