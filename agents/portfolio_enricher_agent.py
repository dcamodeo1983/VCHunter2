from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

class PortfolioEnricherAgent:
    def __init__(self):
        self.headers = {"User-Agent": "Mozilla/5.0"}

    def extract_portfolio_entries(self, html):
        soup = BeautifulSoup(html, "html.parser")
        entries = []

        blocks = soup.find_all(["div", "li", "article", "section"])
        for block in blocks:
            text = block.get_text(separator=" ", strip=True)
            if not text or len(text) < 20 or len(text.split()) > 60:
                continue

            name_candidate = block.find(["h2", "h3", "strong"])
            name = name_candidate.get_text(strip=True) if name_candidate else None

            if name and any(c.isalpha() for c in name):
                entries.append({
                    "name": name,
                    "description": text.replace(name, "").strip(),
                    "source_html": text
                })

        return entries

    def extract_portfolio_entries_from_pages(self, urls):
        all_entries = []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            for url in urls:
                try:
                    page.goto(url, timeout=20000)
                    page.wait_for_timeout(3000)  # Wait for JS content to load
                    html = page.content()
                    entries = self.extract_portfolio_entries(html)
                    all_entries.extend(entries)
                except Exception:
                    continue

            browser.close()

        return all_entries
