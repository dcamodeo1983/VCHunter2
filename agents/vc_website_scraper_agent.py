import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

class VCWebsiteScraperAgent:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {"User-Agent": "Mozilla/5.0"}

    def scrape_text(self, url):
        try:
            response = self.session.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            tags = soup.find_all(["h1", "h2", "h3", "p", "li"])
            content = "\n".join(tag.get_text(strip=True) for tag in tags if tag.get_text(strip=True))
            return content.strip()
        except Exception as e:
            return f"[Error scraping {url}: {e}]"

    def find_portfolio_links(self, base_url):
        try:
            response = self.session.get(base_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            links = soup.find_all("a", href=True)
            portfolio_links = [
                urljoin(base_url, link["href"])
                for link in links
                if any(kw in link.get_text(strip=True).lower() for kw in ["portfolio", "companies", "investments"])
            ]
            return list(set(portfolio_links))
        except Exception:
            return []
