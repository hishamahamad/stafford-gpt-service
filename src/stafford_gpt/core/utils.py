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

            # Force accordion tab content to be visible by setting display: block
            # This ensures content hidden in Elementor accordion tabs is accessible
            await page.evaluate('''
                document.querySelectorAll('[id*="elementor-tab-content-"]').forEach(el => {
                    el.style.display = 'block !important';
                    el.style.visibility = 'visible';
                    el.style.opacity = '1';
                    el.style.height = 'auto';
                    el.style.overflow = 'visible';
                });
            ''')

            # Wait a moment for any dynamic content to load after making tabs visible
            await page.wait_for_timeout(1000)

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
