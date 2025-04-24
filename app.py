# VC Hunter Streamlit UI Upgrade (Semantic-First)

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

        st.info("🧠 Summarizing your concept...")
        summary = summarizer.summarize(cleaned_text)
        st.subheader("🧾 Preview of Extracted Text")
        st.text(cleaned_text[:1000])

        st.header("📄 Startup Summary")
        st.markdown(f"> {summary}")

        st.info("🔗 Creating embedding...")
        embedding = embedder.embed_text(summary)
        if isinstance(embedding, list):
            st.success(f"✅ Embedding created. Vector length: {len(embedding)}")
        else:
            st.error(embedding)

# === VC Landscape Analysis ===
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
            if len(vc_site_text.strip()) < 100:
                st.warning(f"⚠️ Skipping {url} due to short or empty site text.")
                continue
            st.text(vc_site_text[:500])

            st.info("Extracting portfolio entries...")
            portfolio_links = scraper.find_portfolio_links(url)
            structured_portfolio = (
                enricher.extract_portfolio_entries_from_pages(portfolio_links)
                if portfolio_links else enricher.extract_portfolio_entries(vc_site_text)
            )
            st.markdown(f"✅ {len(structured_portfolio)} portfolio entries found.")

            # 🧠 NEW: Run LLM interpreter FIRST
            st.info("🧠 Interpreting strategy (LLM)...")
            interpreter_summary = interpreter.interpret_strategy(url, vc_site_text, structured_portfolio)

            if not interpreter_summary or "Error" in interpreter_summary:
                st.error(f"❌ Interpretation failed for {url}. Skipping.")
                continue

            st.info("📐 Embedding profile (enriched)...")
            portfolio_text = "\n".join([entry['name'] + ": " + entry['description'] for entry in structured_portfolio])
            vc_embedding = embed_vc_profile(vc_site_text, portfolio_text, interpreter_summary, embedder)

            if not isinstance(vc_embedding, list) or not vc_embedding or not all(isinstance(x, (float, int)) for x in vc_embedding):
                st.error(f"❌ Invalid embedding for {url}. Skipping.")
                continue

            st.markdown("**Strategic Summary:**")
            st.text(interpreter_summary)

            vc_profile = {
                "name": url.split("//")[-1].replace("www.", ""),
                "url": url,
                "embedding": vc_embedding,
                "portfolio_size": len(structured_portfolio),
                "strategy_summary": interpreter_summary,
                "category": None,
                "motivational_signals": [],
                "cluster_id": None,
                "coordinates": [None, None]
            }

            cached = load_vc_profiles()
            cached = [p for p in cached if p['url'] != url]
            cached.append(vc_profile)
            save_vc_profiles(cached)

# === Clustering and Categorization ===
st.divider()
st.subheader("🧭 VC Landscape Categorization")

if st.button("Run Clustering + Categorization"):
    st.info("🔄 Running clustering...")
    cluster_agent = ClusteringAgent(n_clusters=5)
    clustered = cluster_agent.cluster()
    st.success("✅ Clustering complete.")

    st.info("🏷 Assigning semantic categories...")
    categorize_agent = CategorizerAgent(api_key=openai_api_key)
    categorized = categorize_agent.categorize_clusters()
    st.success("✅ Categorization complete.")

    st.balloons()
    st.success(f"🗂 {len(categorized)} VC profiles categorized and positioned.")

# === Visualization ===
st.divider()
st.subheader("📊 VC Landscape Map")

viz_agent = VisualizationAgent(api_key=openai_api_key)

if st.button("🔁 Regenerate Axis Labels (Optional)"):
    viz_agent.regenerate_axis_labels()
    st.success("🧠 PCA axis labels refreshed via LLM.")

fig = viz_agent.generate_cluster_map()
if fig:
    labels = viz_agent.load_axis_labels()
    st.markdown(f"**🧭 X-Axis ({labels['x_label']}):** {labels.get('x_description', '')}")
    st.markdown(f"**🧭 Y-Axis ({labels['y_label']}):** {labels.get('y_description', '')}")
    st.plotly_chart(fig)
else:
    st.warning("No VC profiles found with valid cluster coordinates.")

# === Category Explorer
st.divider()
st.subheader("📚 Strategic VC Category Browser")

profiles = load_vc_profiles()
by_category = {}
for p in profiles:
    cat = (p.get("category") or "").split("\n")[0].replace("Category:", "").strip()
    rationale = next((line for line in p.get("category", "").splitlines() if line.lower().startswith("rationale")), "")
    example = p.get("name", "")
    if cat not in by_category:
        by_category[cat] = {"rationale": rationale, "examples": set()}
    by_category[cat]["examples"].add(example)

for cat, details in by_category.items():
    st.markdown(f"### {cat}")
    if details["rationale"]:
        st.markdown(f"**Rationale:** {details['rationale']}")
    st.markdown(f"**Example Firms:** {', '.join(sorted(details['examples']))}")
