import requests
from bs4 import BeautifulSoup

class PortfolioEnricherAgent:
    def __init__(self):
        self.session = requests.Session()
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

    def fetch_and_enrich(self, url):
        try:
            response = self.session.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return self.extract_portfolio_entries(response.text)
        except Exception as e:
            return f"[Error enriching portfolio: {e}]"
