
# Complete VC Hunter App with Founder Upload, Survey, VC URL Upload, Clustering, and Visualization

import streamlit as st
import os
import pandas as pd
import json
from dotenv import load_dotenv
from agents.founder_doc_reader_agent import FounderDocReaderAgent
from agents.llm_summarizer_agent import LLMSummarizerAgent
from agents.embedder_agent import EmbedderAgent
from agents.vc_website_scraper_agent import VCWebsiteScraperAgent
from agents.portfolio_enricher_agent import PortfolioEnricherAgent
from agents.vc_strategic_interpreter_agent import VCStrategicInterpreterAgent
from agents.clustering_agent import ClusteringAgent
from agents.categorizer_agent import CategorizerAgent
from agents.visualization_agent import VisualizationAgent
from agents.founder_survey_agent import FounderSurveyAgent
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

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
embedder = EmbedderAgent(api_key=openai_api_key)

# === Founder Upload Section ===
uploaded_file = st.file_uploader("📄 Upload Your White Paper", type=["pdf", "txt", "docx"])
survey_summary = ""
if uploaded_file:
    reader = FounderDocReaderAgent()
    summarizer = LLMSummarizerAgent(api_key=openai_api_key)
    survey_agent = FounderSurveyAgent()

    text = reader.extract_text(uploaded_file)
    cleaned_text = clean_text(text)
    summary = summarizer.summarize(cleaned_text)
    st.subheader("📄 Startup Summary")
    st.text_area("Extracted Text", cleaned_text[:1000])
    st.markdown(f"> {summary}")

    with st.form("founder_survey"):
        stage = st.selectbox("Stage", ["Idea", "MVP", "Growth"])
        revenue = st.selectbox("Revenue", ["$0", "<$10k", ">$10k"])
        gtm = st.selectbox("Go to Market", ["PLG", "Sales-led"])
        submitted = st.form_submit_button("Submit Survey")
        if submitted:
            survey_summary = survey_agent.format_survey_summary({
                "stage": stage, "revenue": revenue, "gtm": gtm
            })
    final_input = f"{summary}\n\n{survey_summary.strip()}" if survey_summary else summary
    embedding = embedder.embed_text(final_input)

# === VC CSV Upload + Embedding ===
st.divider()
st.subheader("📥 Upload CSV of VC URLs")

vc_csv = st.file_uploader("Upload CSV with `url` column", type=["csv"])
if vc_csv:
    df = pd.read_csv(vc_csv)
    urls = df['url'].dropna().unique().tolist()
    for url in urls:
        scraper = VCWebsiteScraperAgent()
        enricher = PortfolioEnricherAgent()
        interpreter = VCStrategicInterpreterAgent(api_key=openai_api_key)
        site_text = scraper.scrape_text(url)
        if len(site_text.strip()) < 100:
            continue
        portfolio_links = scraper.find_portfolio_links(url)
        portfolio = (
            enricher.extract_portfolio_entries_from_pages(portfolio_links)
            if portfolio_links else enricher.extract_portfolio_entries(site_text)
        )
        summary = interpreter.interpret_strategy(url, site_text, portfolio)
        portfolio_text = "\n".join([f"{x['name']}: {x['description']}" for x in portfolio])
        vc_embedding = embed_vc_profile(site_text, portfolio_text, summary, embedder)
        profile = {
            "name": url.split("//")[-1].replace("www.", ""),
            "url": url,
            "embedding": vc_embedding,
            "portfolio_size": len(portfolio),
            "strategy_summary": summary,
            "category": None,
            "motivational_signals": [],
            "cluster_id": None,
            "coordinates": [None, None]
        }
        profiles = load_vc_profiles()
        profiles = [p for p in profiles if p['url'] != url]
        profiles.append(profile)
        save_vc_profiles(profiles)

# === Clustering & Categorization ===
st.divider()
if st.button("🧭 Cluster & Categorize VCs"):
    ClusteringAgent(n_clusters=5).cluster()
    CategorizerAgent(api_key=openai_api_key).categorize_clusters()
    st.success("✅ Clustering and categorization complete")

# === Visualization ===
st.divider()
viz_agent = VisualizationAgent(api_key=openai_api_key)
if st.button("🔁 Regenerate Axis Labels"):
    viz_agent.regenerate_axis_labels()

fig = viz_agent.generate_cluster_map()
if fig:
    labels = viz_agent.load_axis_labels()
    st.markdown(f"**🧭 X-Axis ({labels['x_label']}):** {labels.get('x_description', '')}")
    st.markdown(f"**🧭 Y-Axis ({labels['y_label']}):** {labels.get('y_description', '')}")
    st.plotly_chart(fig)
