
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
            response = self.scraper.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()

            text_elements = soup.find_all(["p", "h1", "h2", "h3", "h4", "div", "section", "article", "span"])
            raw_text = " ".join([elem.get_text(strip=True) for elem in text_elements if elem.get_text(strip=True)])

            cleaned_text = re.sub(r"\s+", " ", raw_text).strip()
            if not cleaned_text:
                st.warning(f"⚠️ No meaningful text scraped from {url}.")
                return ""

            sentences = [s.strip() for s in cleaned_text.split(".") if s.strip()]
            filtered_sentences = [
                s for s in sentences
                if len(s.split()) > 3 and not any(keyword in s.lower() for keyword in [
                    "cookie policy", "privacy policy", "terms of use", "contact us", "sign up", "log in", "newsletter"
                ])
            ]
            final_text = ". ".join(filtered_sentences).strip()

            if len(final_text.split()) < 50:
                st.warning(f"⚠️ Scraped content from {url} is too short for analysis.")
                return ""

            st.write(f"📝 Scraped {len(final_text)} characters from {url}")
            return final_text

        except Exception as e:
            st.error(f"❌ Error scraping {url}: {str(e)}")
            return ""
