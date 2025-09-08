from typing import Optional
from playwright.async_api import async_playwright

from ..config.settings import settings


async def scrape_with_playwright(url: str) -> Optional[str]:
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            await page.goto(url, timeout=settings.scrape_timeout)
            await page.wait_for_timeout(settings.content_wait_time)

            # Remove unwanted elements
            selectors_to_remove = [
                'nav', 'header', 'footer', '.navigation', '.nav',
                '.sidebar', '.menu', '.cookie', '.popup',
                'script', 'style', '.advertisement', '.ad'
            ]

            for selector in selectors_to_remove:
                try:
                    await page.evaluate(f'document.querySelectorAll("{selector}").forEach(el => el.remove())')
                except:
                    pass

            # Extract text content
            content = await page.evaluate('document.body.innerText')
            await browser.close()

            return content.strip() if content else None

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None
