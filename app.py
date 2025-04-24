
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

st.set_page_config(page_title="VC Hunter", layout="wide")
st.title("🧠 VC Hunter: Founder Intelligence Report")

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
embedder = EmbedderAgent(api_key=openai_api_key)
founder_embedding = None

# === Upload Founder Doc and Survey ===
uploaded_file = st.file_uploader("📄 Upload Your White Paper", type=["pdf", "txt", "docx"])
survey_summary = ""
if uploaded_file:
    reader = FounderDocReaderAgent()
    summarizer = LLMSummarizerAgent(api_key=openai_api_key)
    survey_agent = FounderSurveyAgent()

    text = reader.extract_text(uploaded_file)
    cleaned_text = clean_text(text)
    summary = summarizer.summarize(cleaned_text)

    with st.form("founder_survey"):
        stage = st.selectbox("Stage?", ["Idea", "Prototype", "MVP", "Scaling"])
        revenue = st.selectbox("Revenue?", ["$0", "<$10K", "$10K–$100K", "$100K+"])
        team = st.number_input("Founders?", min_value=1, max_value=10)
        gtm = st.selectbox("GTM?", ["Sales-led", "Product-led", "Bottom-up", "Enterprise"])
        submit = st.form_submit_button("Save")
        if submit:
            survey_summary = survey_agent.format_survey_summary({
                "product_stage": stage, "revenue": revenue, "team_size": team, "gtm": gtm
            })
            st.success("✅ Survey saved")
            st.text(survey_summary)

    full_input = summary + "\n\n" + survey_summary if survey_summary else summary
    founder_embedding = embedder.embed_text(full_input)
    st.success("✅ Founder embedding created")

# === Clustering + Categorization ===
if st.button("Run Clustering"):
    cluster_agent = ClusteringAgent(n_clusters=5)
    cluster_agent.cluster()
    categorize_agent = CategorizerAgent(api_key=openai_api_key)
    categorize_agent.categorize_clusters()
    st.success("🗂 VC profiles clustered and categorized")

# === Visualization ===
viz_agent = VisualizationAgent(api_key=openai_api_key)
fig = viz_agent.generate_cluster_map(founder_embedding_2d=founder_embedding[:2] if founder_embedding else None)
if fig:
    labels = viz_agent.load_axis_labels()
    st.markdown(f"**🧭 X-Axis ({labels['x_label']}):** {labels.get('x_description', '')}")
    st.markdown(f"**🧭 Y-Axis ({labels['y_label']}):** {labels.get('y_description', '')}")
    st.plotly_chart(fig)

# === Founder-VC Match Results
if founder_embedding and st.button("Find Top VC Matches"):
    matcher = FounderMatcherAgent(api_key=openai_api_key)
    matches = matcher.find_matches(founder_embedding)

    for match in matches:
        st.markdown(f"### 🔗 [{match['name']}]({match['url']})")
        st.markdown(f"**Score:** {match['score']} | **Category:** {match['category']}")
        st.markdown(match["explanation"])
