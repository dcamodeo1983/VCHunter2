# VC Hunter Streamlit UI Upgrade (Narrative-Driven)

import streamlit as st
import os
from dotenv import load_dotenv
from agents.founder_doc_reader_agent import FounderDocReaderAgent
from agents.llm_summarizer_agent import LLMSummarizerAgent
from agents.embedder_agent import EmbedderAgent
from agents.vc_website_scraper_agent import VCWebsiteScraperAgent
from agents.portfolio_enricher_agent import PortfolioEnricherAgent
from utils.utils import clean_text, count_tokens, embed_vc_profile

st.set_page_config(page_title="VC Hunter", layout="wide")

st.title("🧠 VC Hunter: Founder Intelligence Report")
st.markdown("""
Upload your startup concept to receive curated insights and a clear summary of your business, powered by LLMs.
""")

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

# === Upload & Run ===
uploaded_file = st.file_uploader("📄 Upload Your White Paper", type=["pdf", "txt", "docx"])
if uploaded_file:
    reader = FounderDocReaderAgent()
    summarizer = LLMSummarizerAgent(api_key=openai_api_key)
    embedder = EmbedderAgent(api_key=openai_api_key)

    st.info("⏳ Extracting text from your file...")
    text = reader.extract_text(uploaded_file)
    if not text.strip():
        st.error("❌ No readable text found in the document.")
    else:
        cleaned_text = clean_text(text)
        token_count = count_tokens(cleaned_text)
        st.success(f"✅ Document processed. ({token_count} tokens)")

        st.info("🧠 Summarizing your concept using GPT...")
        summary = summarizer.summarize(cleaned_text)

        st.header("📄 Startup Summary")
        st.markdown(f"> {summary}")

        st.info("🔗 Creating embedding for downstream analysis...")
        embedding = embedder.embed_text(summary)
        if isinstance(embedding, list):
            st.success(f"✅ Embedding created. Vector length: {len(embedding)}")
        else:
            st.error(embedding)

# === VC Analysis Section ===
st.divider()
st.header("🏢 Analyze a VC Firm")
vc_url = st.text_input("Enter a VC website URL (e.g., https://luxcapital.com)")

if vc_url:
    scraper = VCWebsiteScraperAgent()
    enricher = PortfolioEnricherAgent()

    st.info("🔎 Scraping VC website...")
    vc_site_text = scraper.scrape_text(vc_url)
    st.success("✅ VC site text retrieved.")

    st.info("📊 Extracting portfolio overview...")
    portfolio_text = enricher.extract_portfolio_from_html(vc_site_text)
    st.success("✅ Portfolio data extracted.")

    st.info("🧠 Generating VC profile embedding...")
    vc_embedding = embed_vc_profile(vc_site_text, portfolio_text, embedder)
    if isinstance(vc_embedding, list):
        st.success(f"✅ VC embedding created. Vector length: {len(vc_embedding)}")
    else:
        st.error(vc_embedding)
