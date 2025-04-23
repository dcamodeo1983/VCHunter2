import re
from bs4 import BeautifulSoup
import requests

class PortfolioEnricherAgent:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {"User-Agent": "Mozilla/5.0"}

    def extract_portfolio_from_html(self, html):
        """
        Extract probable company names or short profiles from HTML content.
        Can be improved later with Crunchbase or API integrations.
        """
        try:
            soup = BeautifulSoup(html, "html.parser")
            text_blocks = soup.find_all(["h3", "h4", "p", "div", "span"])
            company_names = []

            for block in text_blocks:
                content = block.get_text(strip=True)
                if 3 < len(content.split()) < 30 and any(char.isalpha() for char in content):
                    company_names.append(content)

            clean = sorted(set(company_names))
            return "\n".join(clean[:50]) if clean else "[No portfolio text found]"
        except Exception as e:
            return f"[Error during enrichment: {e}]"

    def fetch_and_extract_from_url(self, url):
        try:
            response = self.session.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return self.extract_portfolio_from_html(response.text)
        except Exception as e:
            return f"[Error fetching from {url}: {e}]"
