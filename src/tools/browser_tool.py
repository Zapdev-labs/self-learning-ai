"""General-purpose browser automation beyond testing."""

import logging
from typing import Any, Dict, List, Optional

try:
    from playwright.async_api import async_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

from ..core.config import Config

logger = logging.getLogger(__name__)


class BrowserTool:
    """
    Agent-accessible browser for research, interaction, and validation.
    Can search, navigate, click, fill forms, extract data.
    """

    def __init__(self, config: Config):
        self.cfg = config.browser
        self.headless = self.cfg.get("headless", True)
        self.viewport = {
            "width": self.cfg.get("viewport_width", 1280),
            "height": self.cfg.get("viewport_height", 720),
        }
        self.timeout = self.cfg.get("default_timeout", 30000)
        self._browser = None
        self._context = None

        if not HAS_PLAYWRIGHT:
            logger.warning("Playwright not installed. Browser tool disabled.")

    async def _ensure_browser(self):
        if not HAS_PLAYWRIGHT:
            raise RuntimeError("Playwright not available")
        if self._browser is None:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=self.headless)
            self._context = await self._browser.new_context(viewport=self.viewport)

    async def navigate(self, url: str) -> Dict[str, Any]:
        await self._ensure_browser()
        page = await self._context.new_page()
        await page.goto(url, timeout=self.timeout, wait_until="networkidle")
        title = await page.title()
        content = await page.content()
        text = await page.evaluate("() => document.body.innerText")
        await page.close()
        return {
            "url": url,
            "title": title,
            "text": text[:5000],
            "links": await self._extract_links(page) if False else [],  # simplified
        }

    async def search_google(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Simple Google search via browser (no API key needed)."""
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        result = await self.navigate(url)
        # Very basic extraction
        text = result.get("text", "")
        # Parse out result titles/links roughly
        lines = [l.strip() for l in text.split("\n") if l.strip() and len(l.strip()) > 20]
        return [{"title": l[:100], "snippet": l} for l in lines[:max_results]]

    async def screenshot(self, url: str, path: str) -> str:
        await self._ensure_browser()
        page = await self._context.new_page()
        await page.goto(url, timeout=self.timeout)
        await page.screenshot(path=path, full_page=True)
        await page.close()
        return path

    async def click(self, url: str, selector: str) -> Dict[str, Any]:
        await self._ensure_browser()
        page = await self._context.new_page()
        await page.goto(url, timeout=self.timeout)
        await page.click(selector, timeout=self.timeout)
        await page.wait_for_load_state("networkidle")
        text = await page.evaluate("() => document.body.innerText")
        await page.close()
        return {"url": page.url, "text": text[:3000]}

    async def close(self):
        if self._browser:
            await self._browser.close()
            self._browser = None
        if hasattr(self, "_playwright"):
            await self._playwright.stop()
