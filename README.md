# Twitch Clip Bot

I built this to solve a problem I kept running into as someone who watches a lot of
streams: the best moments are scattered across hours of VOD and nobody has time to go
find them. This bot watches a live Twitch stream, figures out when something clip-worthy
is happening based on chat reacting, double-checks that with an LLM so it's not just
chasing spam, cuts the clip, reformats it for vertical video with captions burned in, and
posts it to TikTok no manual editing required.

I'm also working on a second track in parallel (`training/data-collection/`,
`training/alignment/`, `training/feature-extraction/`, `training/model-training/`) that
trains an XGBoost model on how clips
actually performed after posting, so I eventually have a data-driven second opinion
running alongside the LLM's judgment instead of relying on the LLM alone.

## How it works

- **Capture** - `bot/capture/stream_capture.py` grabs the live stream with streamlink and
  pipes it into ffmpeg, which keeps a rolling buffer of short segments on disk (old ones
  get pruned automatically so disk usage doesn't creep up).
- **Chat monitoring** - `bot/twitch/chat_monitor.py` sits on Twitch IRC over a WebSocket
  and watches two signals in a rolling window: how fast messages are coming in, and a
  weighted score for hype keywords ("pog," "clip it," "insane," etc). When either spikes
  past a threshold, it fires off a `HypeEvent` with the chat context around it.
- **AI validation** - `bot/ai/validator.py` sends that chat snapshot to Claude (Haiku 4.5)
  to sanity-check whether it's a real highlight or just noise — chat exploding over an
  actual play versus one person spamming or a raid bot flood. Only clips that pass get
  processed further.
- **Clip extraction** - the buffered segments covering the moment get stitched together
  with ffmpeg, no re-encoding, so this part is quick.
- **Transcription & vertical formatting** - `bot/process/video_processor.py` runs Whisper
  for word-level timestamps, builds captions from that, and uses ffmpeg to composite
  everything into a 9:16 vertical video (blurred background behind the original footage),
  burns in the captions, adds a title card (`bot/ai/title_generator.py`), and normalizes
  audio loudness (EBU R128).
- **Upload** - `bot/upload/tiktok_uploader.py` uses Playwright to drive a real Chromium
  browser, log into TikTok (session gets saved locally so it's not re-logging in every
  time), and post the finished clip with a caption and hashtags.

### The ML side (still in progress)

Instead of hand-labeling a bunch of clips myself, I'm deriving labels from how clips
actually did after they were posted:

- `training/data-collection/` pulls clip metadata (views, likes, publish date) from
  YouTube along with the full-length source VODs
- `training/alignment/` audio-fingerprints clips against their source VODs (FFT
  cross-correlation on onset-strength envelopes) to find exactly where in the original
  stream each clip came from
- `training/feature-extraction/` turns each matched window into features - audio energy
  spikes, scene-change counts, transcript keywords, sentiment (VADER)
- `training/model-training/` trains an XGBoost classifier on those features, using each clip's
  view-count percentile within its own channel as the label (normalized by publish date,
  so newer clips aren't unfairly penalized for having less time to rack up views)

Once this is solid, the plan is to plug it into `bot/ai/validator.py` so the bot has a
fast, learned signal running alongside the LLM call instead of the LLM being the only
judgment. It's not wired in yet, and the model isn't at the accuracy I want yet either.

## Tech Stack

- **Language:** Python 3.11+
- **Stream capture:** streamlink, ffmpeg
- **Chat ingestion:** Twitch IRC over WebSockets, Twitch Helix API
- **AI decisioning:** Claude (Haiku 4.5) for clip validation and title generation
- **Transcription:** OpenAI Whisper (word-level timestamps)
- **Video processing:** ffmpeg (filter graphs, subtitle burn-in, loudness normalization)
- **Upload automation:** Playwright (Chromium)
- **Auth:** Twitch OAuth2 with a local self-signed HTTPS callback
- **ML pipeline:** XGBoost, scikit-learn, pandas, librosa/scipy for audio fingerprinting,
  VADER for sentiment

## License

MIT
