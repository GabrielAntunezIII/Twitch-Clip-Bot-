"""
Twitch OAuth helper — run directly to authorize:

    python -m bot.twitch.auth

Opens the browser, handles the localhost callback, exchanges the code for
an access + refresh token, and writes them into your .env file.
"""

import asyncio
import json
import logging
import os
import secrets
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Event, Thread

import httpx
from dotenv import set_key, load_dotenv

logger = logging.getLogger(__name__)

AUTHORIZE_URL = "https://id.twitch.tv/oauth2/authorize"
TOKEN_URL = "https://id.twitch.tv/oauth2/token"
VALIDATE_URL = "https://id.twitch.tv/oauth2/validate"
REDIRECT_URI = "https://localhost:3000/callback"
SCOPES = ["user:read:chat"]

ENV_PATH = Path(__file__).parents[3] / ".env"


# ── Local callback server ──────────────────────────────────────────────────────

class _CallbackHandler(BaseHTTPRequestHandler):
    code: str | None = None
    state_received: str | None = None
    done = Event()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        if "error" in params:
            error = params["error"][0]
            self.send_response(400)
            self.end_headers()
            self.wfile.write(f"Authorization denied: {error}".encode())
            _CallbackHandler.done.set()
            return

        _CallbackHandler.code = params.get("code", [None])[0]
        _CallbackHandler.state_received = params.get("state", [None])[0]

        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"<h2>Authorized! You can close this tab.</h2>")
        _CallbackHandler.done.set()

    def log_message(self, *_):
        pass


def _run_callback_server() -> HTTPServer:
    server = HTTPServer(("localhost", 3000), _CallbackHandler)
    Thread(target=server.serve_forever, daemon=True).start()
    return server


# ── Token exchange & persistence ───────────────────────────────────────────────

def _exchange_code(client_id: str, client_secret: str, code: str) -> dict:
    r = httpx.post(TOKEN_URL, data={
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": REDIRECT_URI,
    })
    r.raise_for_status()
    return r.json()


def _save_tokens(access_token: str, refresh_token: str) -> None:
    ENV_PATH.touch(exist_ok=True)
    set_key(str(ENV_PATH), "TWITCH_ACCESS_TOKEN", access_token)
    set_key(str(ENV_PATH), "TWITCH_REFRESH_TOKEN", refresh_token)
    logger.info("Tokens saved to %s", ENV_PATH)


def _fetch_broadcaster_id(access_token: str, client_id: str) -> str:
    r = httpx.get(
        "https://api.twitch.tv/helix/users",
        headers={"Authorization": f"Bearer {access_token}", "Client-Id": client_id},
    )
    r.raise_for_status()
    user = r.json()["data"][0]
    return user["id"]


# ── Token refresh ──────────────────────────────────────────────────────────────

def refresh_access_token(client_id: str, client_secret: str, refresh_token: str) -> str:
    """Exchange a refresh token for a new access token and persist it. Returns new access token."""
    r = httpx.post(TOKEN_URL, data={
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    })
    r.raise_for_status()
    data = r.json()
    _save_tokens(data["access_token"], data["refresh_token"])
    logger.info("Access token refreshed")
    return data["access_token"]


def validate_token(access_token: str) -> bool:
    """Returns True if the token is still valid."""
    r = httpx.get(VALIDATE_URL, headers={"Authorization": f"OAuth {access_token}"})
    return r.status_code == 200


# ── Main authorization flow ────────────────────────────────────────────────────

def authorize() -> None:
    load_dotenv(ENV_PATH)
    client_id = os.environ.get("TWITCH_CLIENT_ID")
    client_secret = os.environ.get("TWITCH_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise SystemExit(
            "TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET must be set in .env before running auth."
        )

    state = secrets.token_urlsafe(16)
    _CallbackHandler.done.clear()

    server = _run_callback_server()

    auth_url = (
        f"{AUTHORIZE_URL}?response_type=code"
        f"&client_id={client_id}"
        f"&redirect_uri={urllib.parse.quote(REDIRECT_URI, safe='')}"
        f"&scope={urllib.parse.quote(' '.join(SCOPES))}"
        f"&state={state}"
    )

    print(f"\nOpening browser for Twitch authorization...\n{auth_url}\n")
    webbrowser.open(auth_url)

    _CallbackHandler.done.wait(timeout=120)
    server.shutdown()

    if not _CallbackHandler.code:
        raise SystemExit("Authorization failed or timed out.")

    if _CallbackHandler.state_received != state:
        raise SystemExit("State mismatch — possible CSRF. Aborting.")

    tokens = _exchange_code(client_id, client_secret, _CallbackHandler.code)
    access_token = tokens["access_token"]
    refresh_token = tokens["refresh_token"]

    _save_tokens(access_token, refresh_token)

    broadcaster_id = _fetch_broadcaster_id(access_token, client_id)
    set_key(str(ENV_PATH), "TWITCH_BROADCASTER_ID", broadcaster_id)
    logger.info("Broadcaster ID saved: %s", broadcaster_id)

    print("\nAuthorization complete!")
    print(f"  Access token  : {access_token[:10]}...")
    print(f"  Broadcaster ID: {broadcaster_id}")
    print(f"  Saved to      : {ENV_PATH}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    authorize()
