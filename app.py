import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import streamlit as st
import os
import pandas as pd
import json
from dotenv import load_dotenv
import numpy as np
from sklearn.decomposition import PCA


# === Agent Imports ===
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

# === Utility Functions ===
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

# === Streamlit UI Setup ===
st.set_page_config(page_title="VC Hunter", layout="wide")
st.title("🧠 VC Hunter: Founder Intelligence Report")
st.markdown("Upload your startup concept to receive curated insights and a clear summary of your business, powered by LLMs.")

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
embedder = EmbedderAgent(api_key=openai_api_key)

founder_2d = None
founder_cluster_id = None
top_vc_names = []

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
            product_stage = st.selectbox("Product Stage", ["Idea", "Prototype", "MVP", "Scaling"])
            revenue = st.selectbox("Revenue Range", ["$0", "< $10K", "$10K–$100K", "$100K+"])
            team_size = st.number_input("Team Size", min_value=1, max_value=10, step=1)
            product_type = st.selectbox("Product Type", ["SaaS", "Consumer App", "Deep Tech", "Hardware", "Marketplace", "Other"])
            location = st.text_input("Headquarters Location")
            gtm = st.selectbox("Go-To-Market Strategy", ["Sales-led", "Product-led", "Bottom-up", "Enterprise"])
            customer = st.selectbox("Customer Type", ["Enterprise", "SMB", "Consumer", "Government"])
            moat = st.selectbox("Moat", ["Yes – IP", "Yes – Data", "Yes – Brand", "No Moat Yet"])
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
                "moat": moat,
            }
            survey_summary = survey_agent.format_survey_summary(responses)
            st.success("✅ Survey captured successfully!")
            st.text(survey_summary)

        combined_input = f"{summary.strip()}\n\n{survey_summary.strip()}" if survey_summary else summary.strip()
        if not combined_input.strip():
            st.error("Combined input is empty. Cannot create embedding.")
            st.stop()

        st.info("🔗 Creating embedding...")
        embedding = embedder.embed_text(combined_input)

        if isinstance(embedding, list):
            st.success(f"✅ Embedding created. Vector length: {len(embedding)}")
            matcher = FounderMatcherAgent(embedding)
            top_matches = matcher.match(top_k=5)

            # Generate rationale using OpenAI
            import openai
            openai.api_key = openai_api_key

            for match in top_matches:
                prompt = f"""
You are a startup advisor. A founder has submitted this business summary:

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
                top_vc_names = [match['url'].strip().lower() for match in top_matches]
                # Display summary table
                match_df = pd.DataFrame([{
                    "VC Name": m["name"],
                    "Category": m["category"],
                    "Score": m["score"],
                    "Justification": m.get("rationale", "")
                } for m in top_matches])
                st.dataframe(match_df, use_container_width=True)

                # Full rationale view in expander
                
with st.expander("📝 Full Justifications for Top Matches"):
    st.markdown("Each firm below is matched to your concept with a tailored rationale:")
    for match in top_matches:
        st.markdown(f"**{match['name']}** — [{match['url']}]({match['url']})")
        st.markdown(f"• Category: {match['category']}  |  Similarity Score: {match['score']}")
        rationale = match.get('rationale') or 'No strategy available.'
        st.markdown(f"**Why this VC is a match:**\n\n> {rationale}")
        st.markdown("---")


                    st.markdown("Each firm below is matched to your concept with a tailored rationale:")

                    # (Removed duplicated detailed top match display block)

                for match in top_matches:
                    st.markdown(f"**{match['name']}** — [{match['url']}]({match['url']})")
                    st.markdown(f"• Category: {match['category']}  |  Similarity Score: {match['score']}")
                    rationale = match.get('rationale') or 'No strategy available.'
                    st.markdown(f"**Why this VC is a match:**\n\n> {rationale}")
                    st.markdown("---")
else:
    st.info("Please upload a white paper to begin analysis.")

# === VC URL Upload ===
st.divider()
st.header("📥 Upload CSV of VC URLs")
vc_csv = st.file_uploader("Upload a CSV with a column named 'url'", type=["csv"])
if vc_csv:
    df = pd.read_csv(vc_csv)
    if "url" not in df.columns:
        st.error("CSV must contain a column named 'url'.")
        st.stop()
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
            structured_portfolio = (
                enricher.extract_portfolio_entries_from_pages(portfolio_links)
                if portfolio_links
                else enricher.extract_portfolio_entries(vc_site_text)
            )
            st.markdown(f"✅ {len(structured_portfolio)} portfolio entries found.")

            st.info("Embedding profile...")
            portfolio_text = "\n".join([
                f"{entry.get('name', 'Unnamed')}: {entry.get('description', 'No description')}"
                for entry in structured_portfolio
            ])

            st.info("Interpreting strategy...")
            strategy_summary = interpreter.interpret_strategy(url, vc_site_text, structured_portfolio)
            st.markdown(f"🧠 Strategy Summary: {strategy_summary[:300]}...")

            tagger = StrategicTaggerAgent(api_key=openai_api_key)
            vc_tag_data = tagger.generate_tags_and_signals(strategy_summary)

            vc_embedding = embed_vc_profile(vc_site_text, portfolio_text, strategy_summary, embedder)

            vc_profile = {
                "name": url.split("//")[-1].replace("www.", ""),
                "url": url,
                "embedding": vc_embedding,
                "portfolio_size": len(structured_portfolio),
                "strategy_summary": strategy_summary,
                "strategic_tags": vc_tag_data.get("tags", []),
                "motivational_signals": vc_tag_data.get("motivational_signals", []),
                "category": None,
                "cluster_id": None,
                "coordinates": [None, None],
            }

            cached_profiles = [p for p in load_vc_profiles() if p["url"] != url]
            cached_profiles.append(vc_profile)
            save_vc_profiles(cached_profiles)
            st.success(f"✅ Saved {len(cached_profiles)} profiles. Beginning clustering and categorization...")

    cluster_agent = ClusteringAgent(n_clusters=5)
    clustered_profiles = cluster_agent.cluster()
    categorize_agent = CategorizerAgent(api_key=openai_api_key)
    categorized_profiles = categorize_agent.categorize_clusters()
    save_vc_profiles(categorized_profiles)

    vc_profiles = load_vc_profiles()
    vc_embeddings = [p['embedding'] for p in vc_profiles if isinstance(p.get('embedding'), list)]

    try:
        pca = PCA(n_components=2, random_state=42)
        coords_2d = pca.fit_transform(np.array(vc_embeddings))
    except Exception as e:
        st.error(f"Error during PCA transformation: {e}")
        st.stop()

    for profile, (x, y) in zip(vc_profiles, coords_2d):
        profile['pca_x'] = float(x)
        profile['pca_y'] = float(y)
    save_vc_profiles(vc_profiles)

    if isinstance(embedding, list):
        founder_2d = pca.transform([embedding])[0]
        viz_agent = VisualizationAgent(api_key=openai_api_key)
        dim_agent = DimensionExplainerAgent(api_key=openai_api_key)
        dim_agent.generate_axis_labels()
        labels = dim_agent.load_dimension_labels()

        fig, labels = viz_agent.generate_cluster_map(
            profiles=vc_profiles,
            coords_2d=coords_2d,
            pca=pca,
            founder_embedding_2d=founder_2d,
            founder_cluster_id=founder_cluster_id,
            top_match_names=top_vc_names,
            dimension_labels=labels
        )

        if fig:
            st.markdown(f"**🧭 X-Axis ({labels['x_label']}, {labels.get('x_variance', 0.0) * 100:.1f}% variance):** {labels.get('x_description', '')}")
            st.markdown(f"**🧭 Y-Axis ({labels['y_label']}, {labels.get('y_variance', 0.0) * 100:.1f}% variance):** {labels.get('y_description', '')}")
            st.plotly_chart(fig, use_container_width=True)

            if 'descriptions_markdown' in labels:
                st.subheader("🔎 VC Category Descriptions")
                for block in labels['descriptions_markdown'].split("\n"):
                    if block.strip():
                        st.markdown(f"🔹 {block}")
        else:
            st.warning("No VC profiles found with valid cluster coordinates.")

st.divider()
