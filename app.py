import streamlit as st
import os
import pandas as pd
import json
import numpy as np
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from agents.founder_doc_reader_agent import FounderDocReaderAgent
from agents.strategic_tagger_agent import StrategicTaggerAgent
from agents.llm_summarizer_agent import LLMSummarizerAgent
from agents.embedder_agent import EmbedderAgent
from agents.vc_website_scraper_agent import VCWebsiteScraperAgent
from agents.portfolio_enricher_agent import PortfolioEnricherAgent
from agents.vc_strategic_interpreter_agent import VCStrategicInterpreterAgent
from agents.cluster_interpreter_agent import ClusterInterpreterAgent
from agents.categorizer_agent import CategorizerAgent
from agents.visualization_agent import VisualizationAgent
from agents.dimension_explainer_agent import DimensionExplainerAgent
from agents.founder_survey_agent import FounderSurveyAgent
from agents.founder_matcher_agent import FounderMatcherAgent
from utils.utils import clean_text, count_tokens, embed_vc_profile
import openai

VC_PROFILE_PATH = "outputs/vc_profiles.json"

def load_vc_profiles(expected_dim=1536):
    """Load VC profiles with valid embeddings matching expected dimensionality."""
    try:
        if os.path.exists(VC_PROFILE_PATH):
            with open(VC_PROFILE_PATH, "r") as f:
                profiles = json.load(f)
            valid_profiles = [
                p for p in profiles
                if isinstance(p.get("embedding"), list) and len(p["embedding"]) == expected_dim
            ]
            if len(valid_profiles) < len(profiles):
                st.warning(f"⚠️ Skipped {len(profiles) - len(valid_profiles)} profiles with invalid embedding dimensions.")
            return valid_profiles
        return []
    except (json.JSONDecodeError, FileNotFoundError) as e:
        st.warning(f"⚠️ Error loading VC profiles: {str(e)}. Starting with empty list.")
        return []

def save_vc_profiles(profiles):
    """Save VC profiles to file, ensuring valid data."""
    if not profiles:
        st.warning("⚠️ Attempted to save an empty list of profiles — skipping save.")
        return
    try:
        with open(VC_PROFILE_PATH, "w") as f:
            json.dump(profiles, f, indent=2)
        st.write(f"📁 Saved {len(profiles)} VC profiles to {VC_PROFILE_PATH}")
    except Exception as e:
        st.error(f"❌ Error saving profiles: {str(e)}")

st.set_page_config(page_title="VC Hunter", layout="wide")
st.title("🧠 VC Hunter: Founder Intelligence Report")
st.markdown("Upload your startup concept to receive curated VC insights and a competitive landscape map.")

# Initialize environment and agents
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not openai_api_key:
    st.error("❌ OpenAI API key not found. Please set it in .env or Streamlit secrets.")
    st.stop()
openai.api_key = openai_api_key
embedder = EmbedderAgent(api_key=openai_api_key)
client = openai.OpenAI(api_key=openai_api_key)

# Document upload and processing
uploaded_file = st.file_uploader("📄 Upload Your White Paper", type=["pdf", "txt", "docx"])
founder_embedding = None
top_matches = []
combined_input = ""
if uploaded_file:
    try:
        reader = FounderDocReaderAgent()
        summarizer = LLMSummarizerAgent(api_key=openai_api_key)
        text = reader.extract_text(uploaded_file)

        if not text.strip():
            st.error("❌ No readable text found in the document.")
            st.stop()

        cleaned_text = clean_text(text)
        summary = summarizer.summarize(cleaned_text)
        st.header("📄 Startup Summary")
        st.markdown(f"> {summary}")

        # Founder survey
        st.header("🧾 Founder Survey (Optional)")
        survey_agent = FounderSurveyAgent()
        with st.form("founder_survey"):
            product_stage = st.selectbox("Product Stage", ["Idea", "Prototype", "MVP", "Scaling"])
            revenue = st.selectbox("Revenue Range", ["$0", "< $10K", "$10K–$100K", "$100K+"])
            team_size = st.number_input("Team Size", min_value=1, max_value=100, value=1)
            product_type = st.selectbox("Product Type", ["SaaS", "Consumer App", "Deep Tech", "Hardware", "Marketplace", "Other"])
            location = st.text_input("HQ Location")
            gtm = st.selectbox("Go-To-Market", ["Sales-led", "Product-led", "Bottom-up", "Enterprise"])
            customer = st.selectbox("Customer Type", ["Enterprise", "SMB", "Consumer", "Government"])
            moat = st.selectbox("Moat", ["Yes – IP", "Yes – Data", "Yes – Brand", "No Moat Yet"])
            submitted = st.form_submit_button("Save")

        survey_summary = ""
        if submitted:
            responses = {
                "product_stage": product_stage,
                "revenue": revenue,
                "team_size": team_size,
                "product_type": product_type,
                "location": location,
                "gtm": gtm,
                "customer": customer,
                "moat": moat,
            }
            survey_summary = survey_agent.format_survey_summary(responses)
            st.success("✅ Survey submitted!")

        # Generate founder embedding
        combined_input = f"{summary.strip()}\n\n{survey_summary.strip()}" if survey_summary else summary.strip()
        founder_embedding = embedder.embed_text(combined_input)

        if isinstance(founder_embedding, list):
            st.success("✅ Founder embedding created.")
            matcher = FounderMatcherAgent(founder_embedding)
            try:
                top_matches = matcher.match(top_k=5)
                top_vc_urls = [m["url"].strip().lower() for m in top_matches]
                st.write(f"✅ Found {len(top_matches)} top VC matches.")
            except Exception as e:
                st.error(f"❌ Error matching
