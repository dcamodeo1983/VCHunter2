
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
st.markdown("Upload your startup concept to receive curated VC insights and a competitive landscape map.")

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
embedder = EmbedderAgent(api_key=openai_api_key)

uploaded_file = st.file_uploader("📄 Upload Your White Paper", type=["pdf", "txt", "docx"])
if uploaded_file:
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

    st.header("🧾 Founder Survey (Optional)")
    survey_agent = FounderSurveyAgent()
    with st.form("founder_survey"):
        product_stage = st.selectbox("Product Stage", ["Idea", "Prototype", "MVP", "Scaling"])
        revenue = st.selectbox("Revenue Range", ["$0", "< $10K", "$10K–$100K", "$100K+"])
        team_size = st.number_input("Team Size", min_value=1, max_value=10)
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

    combined_input = f"{summary.strip()}

{survey_summary.strip()}" if survey_summary else summary.strip()
    embedding = embedder.embed_text(combined_input)

    if isinstance(embedding, list):
        st.success("✅ Founder embedding created.")
        matcher = FounderMatcherAgent(embedding)
        top_matches = matcher.match(top_k=5)
        top_vc_urls = [m["url"].strip().lower() for m in top_matches]

        import openai
        openai.api_key = openai_api_key

        for match in top_matches:
            prompt = f"""
{combined_input}

A venture capital firm has this strategy summary:

{match['rationale']}

Explain why this VC is a strong match. Respond in this format:
"This match specializes in [area]. It is a match for your business because [justification]."
"""
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a helpful VC advisor."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7
                )
                match['rationale'] = response.choices[0].message.content.strip()
            except Exception as e:
                match['rationale'] = "(Rationale generation failed)"
                st.warning(f"Rationale generation error: {str(e)}")

        if top_matches:
            st.subheader("🎯 Top VC Matches")
            st.dataframe(pd.DataFrame(top_matches), use_container_width=True)
            with st.expander("📝 Match Rationale"):
                for match in top_matches:
                    st.markdown(f"**{match['name']}** — [{match['url']}]({match['url']})")
                    st.markdown(f"• Category: {match['category']}  |  Score: {match['score']}")
                    st.markdown(f"> {match['rationale']}")
                    st.markdown("---")

# Scraping and embedding VC sites
st.divider()
st.header("📥 Upload VC CSV")
vc_csv = st.file_uploader("Upload a CSV with a column named 'url'", type=["csv"])
if vc_csv:
    df = pd.read_csv(vc_csv)
    if "url" not in df.columns:
        st.error("CSV must have a 'url' column.")
        st.stop()

    urls = df["url"].dropna().unique().tolist()
    st.success(f"Loaded {len(urls)} VC URLs.")

    for url in urls:
        with st.expander(url):
            scraper = VCWebsiteScraperAgent()
            enricher = PortfolioEnricherAgent()
            interpreter = VCStrategicInterpreterAgent(api_key=openai_api_key)

            vc_text = scraper.scrape_text(url)
            links = scraper.find_portfolio_links(url)
            portfolio = (
                enricher.extract_portfolio_entries_from_pages(links)
                if links else enricher.extract_portfolio_entries(vc_text)
            )

            summary = interpreter.interpret_strategy(url, vc_text, portfolio)
            st.markdown(f"🧠 Strategy: {summary[:300]}...")

            tagger = StrategicTaggerAgent(api_key=openai_api_key)
            tag_data = tagger.generate_tags_and_signals(summary)

            vc_embedding = embed_vc_profile(vc_text, "
".join([f"{e.get('name')}: {e.get('description')}" for e in portfolio]), summary, embedder)

            profile = {
                "name": url.split("//")[-1].replace("www.", ""),
                "url": url,
                "embedding": vc_embedding,
                "portfolio_size": len(portfolio),
                "strategy_summary": summary,
                "strategic_tags": tag_data.get("tags", []),
                "motivational_signals": tag_data.get("motivational_signals", []),
                "category": None,
                "cluster_id": None,
                "coordinates": [None, None],
            }

            all_profiles = [p for p in load_vc_profiles() if p["url"] != url]
            all_profiles.append(profile)
            save_vc_profiles(all_profiles)
            st.success("Profile saved.")

# Clustering and visualization
if os.path.exists(VC_PROFILE_PATH):
    profiles = load_vc_profiles()
    coords = PCA(n_components=2).fit_transform([p["embedding"] for p in profiles if isinstance(p.get("embedding"), list)])
    for i, p in enumerate(profiles):
        p["pca_x"], p["pca_y"] = float(coords[i][0]), float(coords[i][1])
    save_vc_profiles(profiles)

    if uploaded_file and isinstance(embedding, list):
        founder_2d = PCA(n_components=2).fit([p["embedding"] for p in profiles]).transform([embedding])[0]
        dim_agent = DimensionExplainerAgent(api_key=openai_api_key)
        dim_agent.generate_axis_labels()
        labels = dim_agent.load_dimension_labels()

        viz_agent = VisualizationAgent(api_key=openai_api_key)
        fig, labels = viz_agent.generate_cluster_map(
            profiles=profiles,
            coords_2d=coords,
            pca=None,
            founder_embedding_2d=founder_2d,
            founder_cluster_id=None,
            top_match_names=top_vc_urls,
            dimension_labels=labels
        )

        if fig:
            st.markdown(f"**🧭 X-Axis ({labels['x_label']}, {labels.get('x_variance', 0.0) * 100:.1f}%):** {labels.get('x_description', '')}")
            st.markdown(f"**🧭 Y-Axis ({labels['y_label']}, {labels.get('y_variance', 0.0) * 100:.1f}%):** {labels.get('y_description', '')}")
            st.plotly_chart(fig, use_container_width=True)

            if "descriptions_markdown" in labels:
                st.subheader("📚 VC Category Descriptions")
                for block in labels["descriptions_markdown"].split("\n"):
                    if block.strip():
                        st.markdown(f"🔹 {block}")

st.divider()
