# VC Hunter Streamlit UI Upgrade (Full Integration)

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

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
embedder = EmbedderAgent(api_key=openai_api_key)

# === Founder Upload + Survey + Embedding ===
uploaded_file = st.file_uploader("📄 Upload Your White Paper", type=["pdf", "txt", "docx"])
founder_embedding = None
summary = ""
survey_summary = ""

if uploaded_file:
    reader = FounderDocReaderAgent()
    summarizer = LLMSummarizerAgent(api_key=openai_api_key)
    survey_agent = FounderSurveyAgent()

    text = reader.extract_text(uploaded_file)
    if not text.strip():
        st.error("❌ No readable text found.")
    else:
        cleaned_text = clean_text(text)
        summary = summarizer.summarize(cleaned_text)
        st.subheader("📄 Summary")
        st.markdown(f"> {summary}")

        st.subheader("📋 Founder Survey")
        with st.form("founder_survey"):
            product_stage = st.selectbox("Stage?", ["Idea", "Prototype", "MVP", "Scaling"])
            revenue = st.selectbox("Revenue?", ["$0", "< $10K", "$10K–$100K", "$100K+"])
            team_size = st.number_input("Founders?", 1, 10, 2)
            product_type = st.selectbox("Product?", ["SaaS", "Consumer App", "Deep Tech", "Hardware", "Marketplace", "Other"])
            location = st.text_input("HQ Location")
            gtm = st.selectbox("Go-to-Market?", ["Sales-led", "Product-led", "Bottom-up", "Enterprise"])
            customer = st.selectbox("Customer Type?", ["Enterprise", "SMB", "Consumer", "Government"])
            moat = st.selectbox("Moat?", ["Yes – IP", "Yes – Data", "Yes – Brand", "No Moat Yet"])
            submitted = st.form_submit_button("Submit")

            if submitted:
                survey_summary = survey_agent.format_survey_summary({
                    "product_stage": product_stage,
                    "revenue": revenue,
                    "team_size": team_size,
                    "product_type": product_type,
                    "location": location,
                    "gtm": gtm,
                    "customer": customer,
                    "moat": moat
                })
                st.success("✅ Survey saved.")
                st.text(survey_summary)

        combined_input = f"{summary.strip()}\n\n{survey_summary.strip()}" if survey_summary else summary.strip()
        founder_embedding = embedder.embed_text(combined_input)
        if not isinstance(founder_embedding, list):
            st.error("Embedding failed.")
            founder_embedding = None

# === VC CSV Upload and Scraping ===
st.divider()
st.subheader("📥 Upload VC CSV")
vc_csv = st.file_uploader("Upload a CSV with a column named 'url'", type=["csv"])
if vc_csv:
    df = pd.read_csv(vc_csv)
    urls = df['url'].dropna().unique().tolist()
    for url in urls:
        scraper = VCWebsiteScraperAgent()
        enricher = PortfolioEnricherAgent()
        interpreter = VCStrategicInterpreterAgent(api_key=openai_api_key)
        vc_site_text = scraper.scrape_text(url)
        if len(vc_site_text.strip()) < 100:
            continue
        links = scraper.find_portfolio_links(url)
        portfolio = enricher.extract_portfolio_entries_from_pages(links) if links else enricher.extract_portfolio_entries(vc_site_text)
        strategy = interpreter.interpret_strategy(url, vc_site_text, portfolio)
        portfolio_text = "\n".join([f"{e['name']}: {e['description']}" for e in portfolio])
        embedding = embed_vc_profile(vc_site_text, portfolio_text, strategy, embedder)
        if not isinstance(embedding, list):
            continue
        profile = {
            "name": url.split("//")[-1].replace("www.", ""),
            "url": url,
            "embedding": embedding,
            "portfolio_size": len(portfolio),
            "strategy_summary": strategy,
            "category": None,
            "motivational_signals": [],
            "cluster_id": None,
            "coordinates": [None, None]
        }
        cache = load_vc_profiles()
        cache = [p for p in cache if p['url'] != url]
        cache.append(profile)
        save_vc_profiles(cache)

# === Clustering & Categorization ===
st.divider()
st.subheader("🧭 Categorize VC Landscape")
if st.button("Run Clustering + Categorization"):
    clustered = ClusteringAgent(n_clusters=5).cluster()
    categorized = CategorizerAgent(api_key=openai_api_key).categorize_clusters()
    st.success(f"🗂 {len(categorized)} VC profiles categorized.")

# === Matching Founders to VCs
if founder_embedding:
    st.divider()
    st.subheader("🤝 VC Matching Results")
    st.markdown("The following firms are the most aligned with your concept:")
    matcher = FounderMatcherAgent()
    vcs = load_vc_profiles()
    top = matcher.match(founder_embedding, vcs, top_n=5)
    for m in top:
        st.markdown(f"### ⭐ {m['name']} (Score: {m['score']:.3f})")
        st.markdown(f"**Why:** {m['why']}")
        st.markdown(f"**Suggested Messaging:** {m['message']}")

# === Visualization
st.divider()
st.subheader("📊 VC Landscape Map")
viz = VisualizationAgent(api_key=openai_api_key)
if st.button("🔁 Refresh Axis Labels"):
    viz.regenerate_axis_labels()
fig = viz.generate_cluster_map(founder_embedding_2d=matcher.founder_coords if founder_embedding else None)
if fig:
    lbl = viz.load_axis_labels()
    st.markdown(f"**X ({lbl['x_label']}):** {lbl.get('x_description', '')}")
    st.markdown(f"**Y ({lbl['y_label']}):** {lbl.get('y_description', '')}")
    st.plotly_chart(fig)
