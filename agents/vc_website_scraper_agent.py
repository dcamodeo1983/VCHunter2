import streamlit as st
import cloudscraper
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin

class VCWebsiteScraperAgent:
    def __init__(self):
        self.scraper = cloudscraper.create_scraper()

    def scrape_text(self, url):
        try:
            st.write(f"🌐 Fetching {url}...")
            response = self.scraper.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()

            text_elements = soup.find_all(["p", "h1", "h2", "h3", "div", "section"])
            raw_text = " ".join([elem.get_text(strip=True) for elem in text_elements if elem.get_text(strip=True)])

            cleaned_text = re.sub(r"\s+", " ", raw_text).strip()
            if not cleaned_text:
                st.warning(f"⚠️ No meaningful text scraped from {url}.")
                return ""

            sentences = [s.strip() for s in cleaned_text.split(".") if s.strip()]
            filtered_sentences = [
                s for s in sentences
                if len(s.split()) > 5 and not any(keyword in s.lower() for keyword in ["cookie policy", "privacy policy", "terms of use", "contact us", "sign up", "log in"])
            ]
            final_text = ". ".join(filtered_sentences).strip()

            st.write(f"📝 Scraped {len(final_text)} chars from {url}")
            return final_text if final_text else ""

        except Exception as e:
            st.error(f"❌ Error scraping {url}: {str(e)}")
            return ""

    def find_portfolio_links(self, base_url):
        try:
            response = self.scraper.get(base_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            links = []
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if any(keyword in href.lower() or keyword in a.get_text(strip=True).lower() for keyword in ["portfolio", "companies", "investments"]):
                    href = urljoin(base_url, href)
                    links.append(href)
            return list(set(links))[:5]
        except Exception as e:
            st.error(f"❌ Error finding portfolio links for {base_url}: {str(e)}")
            return []
