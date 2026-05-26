# Twitch Clip Bot

## What this is
A bot that monitors live Twitch streams, detects highlight moments using chat activity and AI, clips them, formats them vertically with captions, and uploads to TikTok.

## Pipeline
1. Capture live stream with streamlink
2. Monitor chat via Twitch EventSub for hype spikes
3. AI validates whether the moment is clip-worthy
4. ffmpeg processes clip to 9:16 with captions
5. Playwright uploads to TikTok

## Stack
Python, streamlink, ffmpeg, Whisper, Playwright, Twitch API, TikTok