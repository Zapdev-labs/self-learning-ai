"""Browser-based validation for Next.js and web projects using Playwright."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    from playwright.async_api import async_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

from ..core.config import Config
from ..core.types import EvaluationResult

logger = logging.getLogger(__name__)


class BrowserTester:
    """Opens generated web apps in a headless browser and validates them."""

    def __init__(self, config: Config):
        self.cfg = config.browser
        self.headless = self.cfg.get("headless", True)
        self.viewport_w = self.cfg.get("viewport_width", 1280)
        self.viewport_h = self.cfg.get("viewport_height", 720)
        self.timeout = self.cfg.get("default_timeout", 30000)
        self.screenshot_on_fail = self.cfg.get("screenshot_on_fail", True)

        if not HAS_PLAYWRIGHT:
            logger.warning("Playwright not installed. Browser testing disabled.")

    async def validate_url(
        self,
        url: str,
        checks: Optional[Dict[str, Any]] = None,
        screenshot_path: Optional[Path] = None,
    ) -> Tuple[bool, str, Optional[Path]]:
        """
        Open a URL, run checks, optionally take a screenshot.
        Returns (all_checks_passed, details, screenshot_path).
        """
        if not HAS_PLAYWRIGHT:
            return False, "Playwright not available", None

        checks = checks or {}
        results: Dict[str, Any] = {}

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless)
            context = await browser.new_context(
                viewport={"width": self.viewport_w, "height": self.viewport_h}
            )
            page = await context.new_page()

            try:
                await page.goto(url, timeout=self.timeout, wait_until="networkidle")

                # Check page title
                if "title_contains" in checks:
                    title = await page.title()
                    results["title"] = checks["title_contains"].lower() in title.lower()

                # Check for selector presence
                if "selector_exists" in checks:
                    sel = checks["selector_exists"]
                    el = await page.query_selector(sel)
                    results["selector"] = el is not None

                # Check console errors
                console_errors: list = []
                page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
                # Small delay to catch early errors
                await page.wait_for_timeout(500)
                results["console_errors"] = len(console_errors)

                # Screenshot
                ss_path = None
                if screenshot_path:
                    await page.screenshot(path=str(screenshot_path), full_page=True)
                    ss_path = screenshot_path

                passed = all(v for k, v in results.items() if k != "console_errors")
                passed = passed and results.get("console_errors", 0) == 0

                details = f"Checks: {results}"
                if console_errors:
                    details += f"\nConsole errors: {console_errors[:5]}"

                await browser.close()
                return passed, details, ss_path

            except Exception as e:
                if screenshot_path and self.screenshot_on_fail:
                    try:
                        await page.screenshot(path=str(screenshot_path))
                    except Exception:
                        pass
                await browser.close()
                return False, str(e), screenshot_path if screenshot_path else None

    async def validate_nextjs_dev(
        self,
        port: int = 3000,
        screenshot_dir: Optional[Path] = None,
    ) -> Tuple[bool, str, Optional[Path]]:
        """Convenience method for local Next.js dev server."""
        url = f"http://localhost:{port}"
        ss = (screenshot_dir / "screenshot.png") if screenshot_dir else None
        return await self.validate_url(
            url,
            checks={"selector_exists": "body"},
            screenshot_path=ss,
        )
