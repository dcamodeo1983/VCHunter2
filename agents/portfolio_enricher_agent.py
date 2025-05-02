import streamlit as st
import cloudscraper
from bs4 import BeautifulSoup
import re

class PortfolioEnricherAgent:
    def __init__(self):
        self.scraper = cloudscraper.create_scraper()

    def extract_portfolio_entries(self, html):
        try:
            soup = BeautifulSoup(html, "html.parser")
            entries = []

            blocks = soup.find_all(["div", "li", "article", "section", "tr", "td"])
            for block in blocks:
                text = block.get_text(separator=" ", strip=True)
                if not text or len(text) < 10:
                    continue

                name_candidates = block.find_all(
                    ["h2", "h3", "h4", "strong", "span", "div", "a"],
                    class_=[re.compile("company.*", re.I), re.compile("name.*", re.I), re.compile("title.*", re.I)]
                )
                name = None
                for candidate in name_candidates:
                    candidate_text = candidate.get_text(strip=True)
                    if candidate_text and any(c.isalpha() for c in candidate_text) and len(candidate_text.split()) <= 6:
                        name = candidate_text
                        break

                if not name:
                    name_tags = block.find_all(["h2", "h3", "h4", "strong", "a"])
                    for tag in name_tags:
                        candidate_text = tag.get_text(strip=True)
                        if candidate_text and any(c.isalpha() for c in candidate_text) and len(candidate_text.split()) <= 6:
                            name = candidate_text
                            break

                if not name:
                    words = text.split()
                    if len(words) <= 6 and any(c.isalpha() for c in text):
                        name = text.strip()

                if name:
                    description = re.sub(r"\s+", " ", text.replace(name, "").strip())
                    if not description or len(description) < 10:
                        description = text.strip()
                    entries.append({
                        "name": name,
                        "description": description,
                        "source_html": text
                    })

            return entries
        except Exception as e:
            st.error(f"❌ Error extracting portfolio entries: {str(e)}")
            return []

    def extract_portfolio_entries_from_pages(self, urls):
        all_entries = []
        for url in urls:
            try:
                st.write(f"🌐 Fetching portfolio page: {url}")
                response = self.scraper.get(url, timeout=15)
                response.raise_for_status()
                html = response.text
                entries = self.extract_portfolio_entries(html)
                all_entries.extend(entries)
                st.write(f"📈 Extracted {len(entries)} portfolio entries from {url}")
            except Exception as e:
                st.error(f"❌ Error processing portfolio page {url}: {str(e)}")
                continue
        return all_entries
