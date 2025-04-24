# VC Hunter Streamlit UI Upgrade (Semantic-First + Survey + Match Explained)

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
st.markdown("Upload your startup concept to receive curated insights and a strategic match to venture capital firms.")

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
embedder = EmbedderAgent(api_key=openai_api_key)

# === Upload Founder Document ===
uploaded_file = st.file_uploader("📄 Upload Your White Paper", type=["pdf", "txt", "docx"])
founder_embedding = None
summary = ""
survey_summary = ""

if uploaded_file:
    reader = FounderDocReaderAgent()
    summarizer = LLMSummarizerAgent(api_key=openai_api_key)
    survey_agent = FounderSurveyAgent()

    st.info("⏳ Extracting text from your file...")
    text = reader.extract_text(uploaded_file)
    if not text.strip():
        st.error("❌ No readable text found in the document.")
    else:
        cleaned_text = clean_text(text)
        token_count = count_tokens(cleaned_text)
        st.success(f"✅ Document processed. ({token_count} tokens)")

        st.info("🧠 Summarizing your concept...")
        summary = summarizer.summarize(cleaned_text)
        st.subheader("🧾 Preview of Extracted Text")
        st.text(cleaned_text[:1000])

        st.header("📄 Startup Summary")
        st.markdown(f"> {summary}")

        st.header("🧾 Founder Survey (Optional but Recommended)")
        with st.form("founder_survey"):
            product_stage = st.selectbox("What stage is your product in?", ["Idea", "Prototype", "MVP", "Scaling"])
            revenue = st.selectbox("What is your current revenue range?", ["$0", "< $10K", "$10K–$100K", "$100K+"])
            team_size = st.number_input("How many full-time founders are on your team?", min_value=1, max_value=10, step=1)
            product_type = st.selectbox("What type of product are you building?", ["SaaS", "Consumer App", "Deep Tech", "Hardware", "Marketplace", "Other"])
            location = st.text_input("Where is your company headquartered?")
            gtm = st.selectbox("Primary go-to-market motion?", ["Sales-led", "Product-led", "Bottom-up", "Enterprise"])
            customer = st.selectbox("Primary customer type?", ["Enterprise", "SMB", "Consumer", "Government"])
            moat = st.selectbox("Do you believe you have a moat?", ["Yes – IP", "Yes – Data", "Yes – Brand", "No Moat Yet"])
            submitted = st.form_submit_button("Save Survey")

            if submitted:
                responses = {
                    "product_stage": product_stage,
                    "revenue": revenue,
                    "team_size": team_size,
                    "product_type": product_type,
                    "location": location,
                    "gtm": gtm,
                    "customer": customer,
                    "moat": moat
                }
                survey_summary = survey_agent.format_survey_summary(responses)
                st.success("✅ Survey captured successfully!")
                st.text(survey_summary)

        if survey_summary:
            combined_input = f"{summary.strip()}\n\n{survey_summary.strip()}"

        else:
            combined_input = summary.strip()

        st.info("🔗 Creating embedding...")
        founder_embedding = embedder.embed_text(combined_input)
        if isinstance(founder_embedding, list):
            st.success(f"✅ Embedding created. Vector length: {len(founder_embedding)}")
        else:
            st.error(founder_embedding)
            founder_embedding = None

# === Match to VCs ===
if founder_embedding:
    st.divider()
    st.subheader("🔍 Find Your Top VC Matches")

    st.markdown("🧠 VC match scores are based on semantic similarity between your concept and each firm's strategy.")
    st.markdown("The following list shows your top matches along with reasons why they may be aligned with your business.")

    matcher = FounderMatcherAgent(founder_embedding)
    vc_profiles = load_vc_profiles()
    top_matches = matcher.match(founder_embedding, vc_profiles, top_n=5)

    for match in top_matches:
        st.markdown(f"### ⭐ {match['name']} (Score: {match['score']:.3f})")
        st.markdown(f"**Why This Firm Might Be a Good Match:**
{match['why']}")
        st.markdown(f"**Suggested Messaging Themes:**
{match['message']}")

# === VC Visualization
st.divider()
st.subheader("📊 VC Landscape Map")

viz_agent = VisualizationAgent(api_key=openai_api_key)

if st.button("🔁 Regenerate Axis Labels (Optional)"):
    viz_agent.regenerate_axis_labels()
    st.success("🧠 PCA axis labels refreshed via LLM.")

fig = viz_agent.generate_cluster_map(founder_embedding_2d=matcher.founder_coords if founder_embedding else None)
if fig:
    labels = viz_agent.load_axis_labels()
    st.markdown(f"**🧭 X-Axis ({labels['x_label']}):** {labels.get('x_description', '')}")
    st.markdown(f"**🧭 Y-Axis ({labels['y_label']}):** {labels.get('y_description', '')}")
    st.plotly_chart(fig)
else:
    st.warning("No VC profiles found with valid cluster coordinates.")
