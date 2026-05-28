"""
Twitch OAuth helper — run directly to authorize:

    python -m bot.twitch.auth

Opens the browser, handles the localhost HTTPS callback, exchanges the code
for an access + refresh token, and writes them into your .env file.
"""

import datetime
import logging
import os
import secrets
import ssl
import tempfile
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Event, Thread

import httpx
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from dotenv import load_dotenv, set_key

logger = logging.getLogger(__name__)

AUTHORIZE_URL = "https://id.twitch.tv/oauth2/authorize"
TOKEN_URL = "https://id.twitch.tv/oauth2/token"
VALIDATE_URL = "https://id.twitch.tv/oauth2/validate"
REDIRECT_URI = "https://localhost:3000/callback"
SCOPES = ["chat:read"]

ENV_PATH = Path(__file__).parents[2] / ".env"


# ── Self-signed certificate ────────────────────────────────────────────────────

def _generate_self_signed_cert() -> tuple[str, str]:
    """Generate a temporary self-signed cert for localhost. Returns (cert_path, key_path)."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.datetime.now(datetime.timezone.utc)

    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(hours=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost")]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    cert_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pem")
    key_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pem")

    cert_file.write(cert.public_bytes(serialization.Encoding.PEM))
    key_file.write(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    ))
    cert_file.close()
    key_file.close()

    return cert_file.name, key_file.name


# ── Local HTTPS callback server ────────────────────────────────────────────────

class _CallbackHandler(BaseHTTPRequestHandler):
    code: str | None = None
    state_received: str | None = None
    done = Event()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        if "error" in params:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(f"Authorization denied: {params['error'][0]}".encode())
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


def _run_callback_server() -> tuple[HTTPServer, str, str]:
    cert_path, key_path = _generate_self_signed_cert()

    server = HTTPServer(("localhost", 3000), _CallbackHandler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert_path, key_path)
    server.socket = ctx.wrap_socket(server.socket, server_side=True)

    Thread(target=server.serve_forever, daemon=True).start()
    return server, cert_path, key_path


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
    return r.json()["data"][0]["id"]


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

    server, cert_path, key_path = _run_callback_server()

    auth_url = (
        f"{AUTHORIZE_URL}?response_type=code"
        f"&client_id={client_id}"
        f"&redirect_uri={urllib.parse.quote(REDIRECT_URI, safe='')}"
        f"&scope={urllib.parse.quote(' '.join(SCOPES))}"
        f"&state={state}"
    )

    print(f"\nOpening browser for Twitch authorization...")
    print("NOTE: Your browser will warn about an untrusted certificate — this is expected.")
    print("      Click 'Advanced' -> 'Proceed to localhost' to continue.\n")
    webbrowser.open(auth_url)

    _CallbackHandler.done.wait(timeout=120)
    server.shutdown()

    # Clean up temp cert files
    os.unlink(cert_path)
    os.unlink(key_path)

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

    print("\nAuthorization complete!")
    print(f"  Access token  : {access_token[:10]}...")
    print(f"  Broadcaster ID: {broadcaster_id}")
    print(f"  Saved to      : {ENV_PATH}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    authorize()
