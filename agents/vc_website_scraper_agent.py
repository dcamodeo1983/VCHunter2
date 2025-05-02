import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
import re
from urllib.parse import urljoin

class VCWebsiteScraperAgent:
    def __init__(self):
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        try:
            self.driver = webdriver.Chrome(options=options)
        except Exception as e:
            st.error(f"❌ Failed to initialize Selenium driver: {str(e)}. Ensure ChromeDriver is installed.")
            raise
        self.session = requests.Session()
        self.headers = {"User-Agent": "Mozilla/5.0"}

    def scrape_text(self, url):
        try:
            st.write(f"🌐 Navigating to {url}...")
            self.driver.get(url)
            time.sleep(3)
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, "html.parser")

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
                if len(s.split()) > 5 and not any(keyword in s.lower() for keyword in ["cookie policy", "privacy policy", "terms of use", "contact us", "sign up"])
            ]
            final_text = ". ".join(filtered_sentences).strip()

            st.write(f"📝 Scraped {len(final_text)} chars from {url}")
            return final_text if final_text else ""

        except Exception as e:
            st.error(f"❌ Error scraping {url}: {str(e)}")
            return ""

    def find_portfolio_links(self, base_url):
        try:
            self.driver.get(base_url)
            time.sleep(2)
            soup = BeautifulSoup(self.driver.page_source, "html.parser")
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

    def __del__(self):
        try:
            self.driver.quit()
        except:
            pass
