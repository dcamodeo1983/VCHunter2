
import streamlit as st
import os
import pandas as pd
import json
from dotenv import load_dotenv
from agents.founder_doc_reader_agent import FounderDocReaderAgent
from agents.strategic_tagger_agent import StrategicTaggerAgent
from agents.llm_summarizer_agent import LLMSummarizerAgent
from agents.embedder_agent import EmbedderAgent
from agents.vc_website_scraper_agent import VCWebsiteScraperAgent
from agents.portfolio_enricher_agent import PortfolioEnricherAgent
from agents.vc_strategic_interpreter_agent import VCStrategicInterpreterAgent
from agents.clustering_agent import ClusteringAgent
from agents.categorizer_agent import CategorizerAgent
from agents.visualization_agent import VisualizationAgent
from agents.dimension_explainer_agent import DimensionExplainerAgent
from agents.founder_survey_agent import FounderSurveyAgent
from agents.founder_matcher_agent import FounderMatcherAgent
from utils.utils import clean_text, count_tokens, embed_vc_profile

VC_PROFILE_PATH = "outputs/vc_profiles.json"

def load_vc_profiles():
    try:
        if os.path.exists(VC_PROFILE_PATH):
            with open(VC_PROFILE_PATH, "r") as f:
                return json.load(f)
    except json.JSONDecodeError:
        return []
    return []

def save_vc_profiles(profiles):
    if not profiles:
        st.warning("⚠️ Attempted to save an empty list of profiles — skipping save.")
        return
    with open(VC_PROFILE_PATH, "w") as f:
        json.dump(profiles, f, indent=2)
    st.write(f"📁 Saved {len(profiles)} VC profiles to {VC_PROFILE_PATH}")

st.set_page_config(page_title="VC Hunter", layout="wide")
st.title("🧠 VC Hunter: Founder Intelligence Report")
st.markdown("Upload your startup concept to receive curated insights and a clear summary of your business, powered by LLMs.")

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
embedder = EmbedderAgent(api_key=openai_api_key)

# === VC URL Upload ===
st.divider()
st.header("📥 Upload CSV of VC URLs")

vc_csv = st.file_uploader("Upload a CSV with a column named 'url'", type=["csv"])
if vc_csv:
    df = pd.read_csv(vc_csv)
    urls = df["url"].dropna().unique().tolist()
    st.success(f"✅ Loaded {len(urls)} VC URLs")

    for url in urls:
        with st.expander(f"🔍 {url}"):
            scraper = VCWebsiteScraperAgent()
            enricher = PortfolioEnricherAgent()
            interpreter = VCStrategicInterpreterAgent(api_key=openai_api_key)

            st.info("Scraping site text...")
            vc_site_text = scraper.scrape_text(url)

            st.info("Extracting portfolio entries...")
            portfolio_links = scraper.find_portfolio_links(url)
            if portfolio_links:
                st.info(f"🔗 Found {len(portfolio_links)} portfolio link(s). Scraping...")
                structured_portfolio = enricher.extract_portfolio_entries_from_pages(portfolio_links)
            else:
                st.warning("⚠️ No portfolio page links found. Using homepage instead.")
                structured_portfolio = enricher.extract_portfolio_entries(vc_site_text)

            st.markdown(f"✅ {len(structured_portfolio)} portfolio entries found.")

            st.info("Embedding profile...")
            portfolio_text = "\n".join([
                f"{entry['name']}: {entry['description']}" for entry in structured_portfolio
            ])

            st.info("Interpreting strategy...")
            strategy_summary = interpreter.interpret_strategy(url, vc_site_text, structured_portfolio)
            st.markdown(f"🧠 Strategy Summary: {strategy_summary[:300]}...")

            tagger = StrategicTaggerAgent(api_key=openai_api_key)
            vc_tag_data = tagger.generate_tags_and_signals(strategy_summary)
            vc_tags = vc_tag_data.get("tags", [])
            vc_motivations = vc_tag_data.get("motivational_signals", [])

            cached_profiles = load_vc_profiles()

            vc_profile = {
                "name": url.split("//")[-1].replace("www.", ""),
                "url": url,
                "embedding": embed_vc_profile(vc_site_text, portfolio_text, strategy_summary, embedder),
                "portfolio_size": len(structured_portfolio),
                "strategy_summary": strategy_summary,
                "strategic_tags": vc_tags,
                "motivational_signals": vc_motivations,
                "category": next((p.get("category") for p in cached_profiles if p["url"] == url), "Uncategorized"),
                "cluster_id": None,
                "coordinates": [None, None],
            }

            cached_profiles = [p for p in cached_profiles if p["url"] != url]
            cached_profiles.append(vc_profile)
            save_vc_profiles(cached_profiles)
