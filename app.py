
import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

from agents.founder_doc_reader_agent import FounderDocReaderAgent
from agents.founder_survey_agent import FounderSurveyAgent
from agents.vc_website_scraper_agent import VCWebsiteScraperAgent
from agents.portfolio_enricher_agent import PortfolioEnricherAgent
from agents.embedder_agent import EmbedderAgent
from agents.clustering_agent import ClusteringAgent
from agents.dimension_explainer_agent import DimensionExplainerAgent
from agents.visualization_agent import VisualizationAgent
from agents.founder_matcher_agent import FounderMatcherAgent
from utils.utils import load_vc_profiles, save_vc_profiles, count_tokens

load_dotenv()
st.set_page_config(layout="wide", page_title="🚀 VC Hunter")

st.title("🚀 VC Hunter: Founder Intelligence")
openai_api_key = st.text_input("🔑 Enter your OpenAI API key", type="password")

model_choice = st.selectbox("🧠 Choose LLM Model", ["gpt-4", "gpt-3.5-turbo"])

uploaded_file = st.file_uploader("📄 Upload your startup's white paper", type=["pdf", "txt"])
survey_data = {}
vc_csv = st.file_uploader("📁 Upload VC Firm List (CSV with `url` column)", type=["csv"])

if st.button("Run Analysis"):
    if not openai_api_key or not uploaded_file or not vc_csv:
        st.error("❌ Please provide all required inputs.")
        st.stop()

    # Founder Summary
    with st.spinner("📖 Reading founder document..."):
        doc_agent = FounderDocReaderAgent(api_key=openai_api_key, model=model_choice)
        founder_summary = doc_agent.process(uploaded_file)

    # Founder Survey
    with st.spinner("🧩 Processing founder survey..."):
        survey_agent = FounderSurveyAgent()
        with st.form("founder_survey_form"):
        st.markdown("### 🧩 Founder Survey")
        product_stage = st.selectbox("Product stage?", ["Idea", "Prototype", "Launched", "Scaling"])
        revenue = st.selectbox("Revenue?", ["Pre-revenue", "$0–10K", "$10K–100K", "$100K+"])
        team_size = st.selectbox("Team size?", ["1", "2-5", "6-10", "10+"])
        customer_type = st.selectbox("Target customers?", ["Consumers", "SMBs", "Enterprises", "Governments"])
        moat = st.selectbox("Moat type?", ["Tech/IP", "Brand", "Network Effects", "Execution", "None"])
        channel = st.selectbox("Go-to-market?", ["Direct", "Partnerships", "Bottom-up SaaS", "Marketplace"])
        traction = st.selectbox("Traction signal?", ["Beta users", "Revenue", "Media coverage", "Partnerships"])
        capital_needs = st.selectbox("Funding needs?", ["<$250K", "$250K–1M", "$1M–5M", "$5M+"])
        submitted = st.form_submit_button("Submit Survey")

        if not submitted:
            st.stop()

        survey_responses = {
            "product_stage": product_stage,
            "revenue": revenue,
            "team_size": team_size,
            "customer_type": customer_type,
            "moat": moat,
            "channel": channel,
            "traction": traction,
            "capital_needs": capital_needs
        }
        survey_narrative = survey_agent.format_survey_summary(survey_responses)
        full_founder_narrative = founder_summary + "\n" + survey_narrative

    # VC Scraping
    with st.spinner("🌐 Scraping VC websites..."):
        vc_df = pd.read_csv(vc_csv)
        vc_urls = vc_df["url"].dropna().unique().tolist()
        scraper = VCWebsiteScraperAgent(api_key=openai_api_key, model=model_choice)
        raw_profiles = scraper.scrape(vc_urls)

    # Enrich Portfolios
    with st.spinner("🔍 Enriching portfolio data..."):
        enricher = PortfolioEnricherAgent(api_key=openai_api_key, model=model_choice)
        enriched_profiles = enricher.enrich(raw_profiles)

    # Embed
    with st.spinner("🧠 Embedding VC and founder profiles..."):
        embedder = EmbedderAgent(api_key=openai_api_key, model=model_choice)
        founder_embedding = embedder.embed_text(full_founder_narrative)
        enriched_profiles = embedder.embed_profiles(enriched_profiles)

        if not isinstance(founder_embedding, list):
            st.error("❌ Embedding failed for founder profile.")
            st.stop()

    # Save for reuse
    save_vc_profiles(enriched_profiles)

    # Clustering
    with st.spinner("📊 Clustering VC profiles..."):
        cluster_agent = ClusteringAgent()
        if len(enriched_profiles) < cluster_agent.n_clusters:
            cluster_agent.n_clusters = len(enriched_profiles)
        vc_profiles = cluster_agent.cluster(enriched_profiles)
        pca = cluster_agent.pca

    # Axis Labeling
    with st.spinner("🧭 Interpreting PCA axes..."):
        dim_agent = DimensionExplainerAgent(api_key=openai_api_key, model=model_choice)
        dim_agent.generate_axis_labels()
        dimension_labels = dim_agent.load_dimension_labels()

    # Visualization
    with st.spinner("🖼️ Generating landscape map..."):
        viz_agent = VisualizationAgent()
        fig, _ = viz_agent.generate_cluster_map(vc_profiles, founder_embedding, dimension_labels)
        st.plotly_chart(fig, use_container_width=True)

    # Matching
    with st.spinner("💡 Matching top VC firms..."):
        matcher = FounderMatcherAgent(founder_embedding)
        matches = matcher.match(top_k=5)

        st.subheader("🔎 Top VC Matches")
        for match in matches[:5]:
            st.markdown(f"**{match['name']}**  
Score: {match['score']:.2f}  
URL: {match['url']}")

    st.success("✅ Analysis Complete.")
