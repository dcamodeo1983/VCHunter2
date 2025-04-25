# VC Hunter Streamlit UI Upgrade (Narrative-Driven)

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
st.markdown('''
Upload your startup concept to receive curated insights and a clear summary of your business, powered by LLMs.
''')

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
embedder = EmbedderAgent(api_key=openai_api_key)

# === Upload & Run ===
founder_2d = None
founder_cluster_id = None
uploaded_file = st.file_uploader("📄 Upload Your White Paper", type=["pdf", "txt", "docx"])
if uploaded_file:
    reader = FounderDocReaderAgent()
    summarizer = LLMSummarizerAgent(api_key=openai_api_key)

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

        survey_agent = FounderSurveyAgent()
        survey_summary = ""

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
        embedding = embedder.embed_text(combined_input)
        if isinstance(embedding, list):
            st.success(f"✅ Embedding created. Vector length: {len(embedding)}")

            matcher = FounderMatcherAgent(embedding)
            top_matches = matcher.match(top_k=5)
            if top_matches:
                top_match_url = top_matches[0]["url"]
                top_cluster = next((p.get("cluster_id") for p in load_vc_profiles() if p["url"] == top_match_url), None)
                founder_cluster_id = top_cluster

                st.subheader("🎯 Top VC Matches")
                for match in top_matches:
                    st.markdown(f"**{match['name']}** — [{match['url']}]({match['url']})")
                    st.markdown(f"• Category: {match['category']}  |  Similarity Score: {match['score']}")
                    st.markdown(f"• Strategy: {match['rationale']}")
                    st.markdown("---")
        else:
            st.error("❌ No valid embedding returned.")

# === VC URL Upload ===
st.divider()
st.header("📥 Upload CSV of VC URLs")

vc_csv = st.file_uploader("Upload a CSV with a column named 'url'", type=["csv"])

if vc_csv:
    df = pd.read_csv(vc_csv)
    urls = df['url'].dropna().unique().tolist()
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
            portfolio_text = "
".join([entry['name'] + ": " + entry['description'] for entry in structured_portfolio])
            vc_embedding = embed_vc_profile(vc_site_text, portfolio_text, strategy_summary, embedder)
            st.write("🔍 Embedding type and preview:", type(vc_embedding), vc_embedding[:5] if isinstance(vc_embedding, list) else vc_embedding)

            if strategy_summary:
                lines = strategy_summary.split("
")
                for line in lines:
                    if line.lower().startswith("category"):
                        st.markdown(f"### 🧠 Strategic Identity")
                    elif line.lower().startswith("rationale"):
                        st.markdown(f"**Rationale:** {line.replace('Rationale:', '').strip()}")
                    elif line.lower().startswith("motivational signals"):
                        st.markdown(f"**Motivational Signals:** {line.replace('Motivational Signals:', '').strip()}")
                    else:
                        st.markdown(line)

                vc_profile = {
                    "name": url.split("//")[-1].replace("www.", ""),
                    "url": url,
                    "embedding": vc_embedding,
                    "portfolio_size": len(structured_portfolio),
                    "strategy_summary": strategy_summary,
                    "category": None,
                    "motivational_signals": [],
                    "cluster_id": None,
                    "coordinates": [None, None]
                }

                cached_profiles = load_vc_profiles()
                cached_profiles = [p for p in cached_profiles if p['url'] != url]
                cached_profiles.append(vc_profile)
                save_vc_profiles(cached_profiles)

# === Clustering + Categorization ===
st.divider()
st.subheader("🧭 VC Landscape Categorization")

if st.button("Run Clustering + Categorization"):
    st.info("Clustering VC embeddings...")
    cluster_agent = ClusteringAgent(n_clusters=5)
    clustered_profiles = cluster_agent.cluster()
    st.success("Clustering complete.")

    if uploaded_file and isinstance(embedding, list):
        founder_2d = cluster_agent.transform(embedding)

    st.info("Categorizing each cluster...")
    categorize_agent = CategorizerAgent(api_key=openai_api_key)
    categorized_profiles = categorize_agent.categorize_clusters()
    st.success("Categorization complete.")

    st.balloons()
    st.success(f"🗂 Updated {len(categorized_profiles)} VC profiles with clusters and categories.")

    dim_agent = DimensionExplainerAgent(api_key=openai_api_key)
    dim_agent.generate_axis_labels()

# === Semantic Visualization with Axis Labels ===
st.divider()
st.subheader("📊 VC Landscape Map")

viz_agent = VisualizationAgent(api_key=openai_api_key)

if st.button("🔁 Regenerate Axis Labels (Optional)"):
    viz_agent.regenerate_axis_labels()
    st.success("🧠 PCA axis labels refreshed via LLM.")

fig, labels = viz_agent.generate_cluster_map(founder_embedding_2d=founder_2d, founder_cluster_id=founder_cluster_id)
if fig:
    st.markdown(f"**🧭 X-Axis ({labels['x_label']}, {labels.get('x_variance', 0.0) * 100:.1f}% variance):** {labels.get('x_description', '')}")
    st.markdown(f"**🧭 Y-Axis ({labels['y_label']}, {labels.get('y_variance', 0.0) * 100:.1f}% variance):** {labels.get('y_description', '')}")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No VC profiles found with valid cluster coordinates.")
