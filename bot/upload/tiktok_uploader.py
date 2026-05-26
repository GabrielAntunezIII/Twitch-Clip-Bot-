import asyncio
import logging
from pathlib import Path

from playwright.async_api import (
    Browser, BrowserContext, Page,
    async_playwright,
    TimeoutError as PWTimeout,
)

import config

logger = logging.getLogger(__name__)

LOGIN_URL  = "https://www.tiktok.com/login/phone-or-email/email"
UPLOAD_URL = "https://www.tiktok.com/upload"
UPLOAD_TIMEOUT = 120_000  # ms — TikTok's server-side processing is slow
CAPTCHA_PAUSE  = 90       # seconds to wait for manual CAPTCHA solve

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


class TikTokUploader:

    def __init__(self):
        self._playwright = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._playwright = await async_playwright().start()
        session = Path(config.TIKTOK_SESSION_FILE)

        if session.exists():
            await self._launch(headless=config.TIKTOK_HEADLESS, session=session)
            if not await self._session_valid():
                logger.info("Saved session expired — re-logging in")
                await self._browser.close()
                await self._login_flow()
        else:
            await self._login_flow()

        logger.info("TikTok uploader ready")

    async def stop(self) -> None:
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    # ── Browser helpers ───────────────────────────────────────────────────────

    async def _launch(
        self,
        headless: bool,
        session: Path | None = None,
    ) -> None:
        self._browser = await self._playwright.chromium.launch(
            headless=headless,
            args=["--disable-blink-features=AutomationControlled"],
        )
        kwargs = dict(viewport={"width": 1280, "height": 800}, user_agent=_UA)
        if session and session.exists():
            kwargs["storage_state"] = str(session)
        self._context = await self._browser.new_context(**kwargs)

    async def _session_valid(self) -> bool:
        page = await self._context.new_page()
        try:
            await page.goto("https://www.tiktok.com/", wait_until="domcontentloaded", timeout=15_000)
            return "login" not in page.url
        except Exception:
            return False
        finally:
            await page.close()

    # ── Login flow (always visible so user can solve CAPTCHA) ─────────────────

    async def _login_flow(self) -> None:
        logger.info("Launching visible browser for TikTok login...")
        await self._launch(headless=False)

        page = await self._context.new_page()
        try:
            await self._login(page)
        finally:
            await page.close()

        # Save session, then relaunch headless for uploads
        await self._context.storage_state(path=config.TIKTOK_SESSION_FILE)
        logger.info("Session saved to %s", config.TIKTOK_SESSION_FILE)
        await self._browser.close()
        await self._launch(headless=config.TIKTOK_HEADLESS, session=Path(config.TIKTOK_SESSION_FILE))

    async def _login(self, page: Page) -> None:
        await page.goto(LOGIN_URL, wait_until="domcontentloaded")

        # Fill email
        email_input = page.locator(
            'input[name="username"], input[autocomplete="username"], input[type="email"]'
        ).first
        await email_input.wait_for(timeout=10_000)
        await email_input.fill(config.TIKTOK_EMAIL)

        # Fill password
        await page.locator('input[type="password"]').first.fill(config.TIKTOK_PASSWORD)

        # Submit
        submit = page.locator(
            '[data-e2e="login-button"], button[type="submit"]'
        ).first
        await submit.click()

        # CAPTCHA may appear here — wait for manual solve if needed
        await self._dismiss_captcha_if_present(page)

        # Wait until we leave the login page (home feed or /foryou)
        try:
            await page.wait_for_url(
                lambda url: "login" not in url and "tiktok.com" in url,
                timeout=60_000,
            )
            logger.info("Login successful")
        except PWTimeout:
            raise RuntimeError(
                "Login timed out — check credentials or solve any CAPTCHA in the browser window"
            )

    # ── Public upload ─────────────────────────────────────────────────────────

    async def upload(self, clip_path: Path, caption: str | None = None) -> bool:
        if caption is None:
            caption = config.TIKTOK_CAPTION_TEMPLATE.format(
                channel=config.TWITCH_CHANNEL,
                hashtags=config.TIKTOK_HASHTAGS,
            )

        page = await self._context.new_page()
        try:
            return await self._do_upload(page, clip_path, caption)
        except Exception as exc:
            logger.error("Upload failed: %s", exc)
            return False
        finally:
            await page.close()

    # ── Upload flow ───────────────────────────────────────────────────────────

    async def _do_upload(self, page: Page, clip_path: Path, caption: str) -> bool:
        await page.goto(UPLOAD_URL, wait_until="domcontentloaded")

        # Session may have expired mid-run
        if "login" in page.url:
            logger.info("Session expired during upload — re-logging in")
            await self._login_flow()
            await page.goto(UPLOAD_URL, wait_until="domcontentloaded")

        await self._dismiss_captcha_if_present(page)

        # TikTok wraps the uploader in an iframe
        frame = None
        for _ in range(10):
            for f in page.frames:
                if "upload" in f.url:
                    frame = f
                    break
            if frame:
                break
            await asyncio.sleep(1)

        target = frame or page

        logger.info("Attaching: %s", clip_path.name)
        async with page.expect_file_chooser() as fc_info:
            await target.get_by_text("Select file").click()
        await (await fc_info.value).set_files(str(clip_path))

        logger.info("Waiting for TikTok to process the video...")
        try:
            await target.wait_for_selector(
                "[data-e2e='caption-input'], div[contenteditable='true']",
                timeout=UPLOAD_TIMEOUT,
            )
        except PWTimeout:
            logger.error("Timed out waiting for caption input")
            return False

        await self._dismiss_captcha_if_present(page)

        caption_el = target.locator(
            "[data-e2e='caption-input'], div[contenteditable='true']"
        ).first
        await caption_el.click()
        await caption_el.fill("")
        await caption_el.type(caption, delay=30)
        logger.info("Caption set")

        await target.get_by_role("button", name="Post").click()
        logger.info("Post clicked — waiting for confirmation...")

        try:
            await page.wait_for_url("**/profile**", timeout=30_000)
            logger.info("Upload confirmed (profile redirect)")
            return True
        except PWTimeout:
            pass

        try:
            await page.wait_for_selector(
                "text=Your video is being uploaded, text=uploaded successfully",
                timeout=15_000,
            )
            logger.info("Upload confirmed (toast)")
            return True
        except PWTimeout:
            logger.warning("Upload status unknown — check TikTok manually")
            return False

    # ── CAPTCHA handling ──────────────────────────────────────────────────────

    async def _dismiss_captcha_if_present(self, page: Page) -> None:
        try:
            await page.wait_for_selector(
                "[id*='captcha'], [class*='captcha'], iframe[src*='captcha']",
                timeout=2_000,
            )
        except PWTimeout:
            return

        logger.warning(
            "CAPTCHA detected — solve it in the browser window. Waiting %ds...",
            CAPTCHA_PAUSE,
        )
        await asyncio.sleep(CAPTCHA_PAUSE)
