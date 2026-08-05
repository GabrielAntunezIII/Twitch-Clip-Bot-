# feature-extraction

Turns raw downloaded clips into feature vectors for model training (and eventually
for live scoring alongside/inside `bot/ai/validator.py`).

Planned features:
- **Audio spike detection** — loudness/energy spikes via ffmpeg (`astats`/`silencedetect`),
  building on the ffmpeg subprocess conventions in `bot/process/video_processor.py`.
- **Scene-change / motion detection** — ffmpeg scene-change scoring (`select='gt(scene,X)'`)
  to capture fast cuts or high-motion moments.
- **Transcript sentiment analysis** — reuses the Whisper word-timestamp transcription
  already implemented in `bot/process/video_processor.py::_transcribe_sync`, run through
  a sentiment/emotion model.

None of this exists in the current pipeline yet — `bot/ai/validator.py` only looks at
chat text today, so these are net-new signals.
