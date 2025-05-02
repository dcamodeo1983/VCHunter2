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
    """Load VC profiles with valid embeddings and strategy_summary."""
    try:
        if os.path.exists(VC_PROFILE_PATH):
            with open(VC_PROFILE_PATH, "r") as f:
                profiles = json.load(f)
            valid_profiles = []
            invalid_reasons = []
            for p in profiles:
                if not isinstance(p.get("embedding"), list) or len(p["embedding"]) != expected_dim:
                    invalid_reasons.append(f"Profile {p.get('name', 'unknown')}: invalid embedding (got {len(p.get('embedding', []))} dimensions)")
                elif not isinstance(p.get("strategy_summary"), str) or not p["strategy_summary"].strip() or p.get("strategy_summary") is None:
                    invalid_reasons.append(f"Profile {p.get('name', 'unknown')}: missing, empty, or None strategy_summary")
                elif not isinstance(p.get("url"), str) or not p["url"].strip():
                    invalid_reasons.append(f"Profile {p.get('name', 'unknown')}: missing or empty url")
                else:
                    valid_profiles.append(p)
            if invalid_reasons:
                st.error(f"❌ Found {len(invalid_reasons)} invalid profiles:\n" + "\n".join(invalid_reasons) + "\nClearing vc_profiles.json.")
                with open(VC_PROFILE_PATH, "w") as f:
                    json.dump([], f)
                return []
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
    for p in profiles:
        if not all(key in p and p[key] for key in ["name", "url", "embedding", "strategy_summary"]):
            st.error(f"❌ Cannot save profiles: Profile {p.get('name', 'unknown')} missing required fields: {[k for k in ['name', 'url', 'embedding', 'strategy_summary'] if k not in p or not p[k]]}")
            return
    try:
        with open(VC_PROFILE_PATH, "w") as f:
            json.dump(profiles, f, indent=2)
        st.write(f"📁 Saved {len(profiles)} VC profiles to {VC_PROFILE_PATH}")
    except Exception as e:
        st.error(f"❌ Error saving profiles: {str(e)}")

st.set_page_config(page_title="VC Hunter", layout="wide")
st.title("🧠 VC Hunter: Founder Intelligence Report")
st.markdown("Upload your startup concept and a CSV of VC firms to receive curated VC insights and a competitive landscape map.")

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
st.header("📄 Upload Your White Paper")
uploaded_file = st.file_uploader("Choose a PDF, TXT, or DOCX file", type=["pdf", "txt", "docx"])
founder_embedding = None
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
        st.subheader("Startup Summary")
        st.markdown(f"> {summary}")

        # Founder survey
        st.subheader("Founder Survey (Optional)")
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

        if isinstance(founder_embedding, list) and len(founder_embedding) == 1536:
            st.success(f"✅ Founder embedding created ({len(founder_embedding)} dimensions).")
        else:
            st.error(f"❌ Failed to create founder embedding: expected 1536 dimensions, got {len(founder_embedding) if isinstance(founder_embedding, list) else 'none'}.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Error processing document: {str(e)}")
        st.stop()

# VC CSV upload and processing
st.divider()
st.header("📥 Upload VC CSV")
vc_csv = st.file_uploader("Upload a CSV with a column named 'url'", type=["csv"])
if vc_csv and founder_embedding:
    try:
        df = pd.read_csv(vc_csv)
        if "url" not in df.columns:
            st.error("❌ CSV must have a 'url' column.")
            st.stop()

        urls = df["url"].dropna().unique().tolist()
        st.success(f"✅ Loaded {len(urls)} VC URLs.")

        for url in urls:
            with st.expander(url):
                try:
                    scraper = VCWebsiteScraperAgent()
                    enricher = PortfolioEnricherAgent()
                    interpreter = VCStrategicInterpreterAgent(api_key=openai_api_key)

                    vc_text = scraper.scrape_text(url)
                    st.write(f"📄 Scraped text length for {url}: {len(vc_text)} chars")
                    links = scraper.find_portfolio_links(url)
                    portfolio = (
                        enricher.extract_portfolio_entries_from_pages(links)
                        if links else enricher.extract_portfolio_entries(vc_text)
                    )
                    st.write(f"📈 Portfolio size for {url}: {len(portfolio)} companies")

                    try:
                        summary = interpreter.interpret_strategy(url, vc_text, portfolio)
                        st.write(f"🧠 Raw strategy summary for {url}: {summary[:100] if summary else 'None'}...")
                    except Exception as e:
                        st.error(f"❌ Failed to generate strategy summary for {url}: {str(e)}")
                        summary = f"Unable to generate strategy summary due to error: {str(e)}"

                    if not isinstance(summary, str) or not summary.strip() or summary is None:
                        st.error(f"❌ Invalid strategy summary for {url} (got: {summary}). Skipping profile.")
                        continue

                    st.markdown(f"🧠 Strategy: {summary[:300]}...")

                    tagger = StrategicTaggerAgent(api_key=openai_api_key)
                    tag_data = tagger.generate_tags_and_signals(summary)

                    vc_embedding = embed_vc_profile(vc_text, "\n".join([f"{e.get('name', '')}: {e.get('description', '')}" for e in portfolio]), summary, embedder)

                    if not isinstance(vc_embedding, list) or len(vc_embedding) != 1536:
                        st.error(f"❌ Invalid embedding for {url}: expected 1536 dimensions, got {len(vc_embedding) if isinstance(vc_embedding, list) else 'none'}")
                        continue

                    profile = {
                        "name": url.split("//")[-1].replace("www.", ""),
                        "url": url,
                        "embedding": vc_embedding,
                        "portfolio_size": len(portfolio),
                        "strategy_summary": summary,
                        "strategic_tags": tag_data.get("tags", []),
                        "motivational_signals": tag_data.get("motivational_signals", []),
                        "category": None,
                        "category_rationale": None,
                        "category_fit": None,
                        "cluster_id": None,
                        "pca_x": None,
                        "pca_y": None,
                    }

                    # Validate profile before saving
                    required_fields = ["name", "url", "embedding", "strategy_summary"]
                    if not all(key in profile and profile[key] for key in required_fields):
                        st.error(f"❌ Profile for {url} missing required fields: {[k for k in required_fields if k not in profile or not profile[k]]}. Skipping.")
                        continue

                    all_profiles = [p for p in load_vc_profiles() if p["url"] != url]
                    all_profiles.append(profile)
                    save_vc_profiles(all_profiles)
                    st.success("✅ Profile saved.")
                except Exception as e:
                    st.error(f"❌ Error processing {url}: {str(e)}")
                    continue
    except Exception as e:
        st.error(f"❌ Error reading CSV: {str(e)}")

    # Matching and display
    try:
        profiles = load_vc_profiles(expected_dim=1536)
        if not profiles:
            st.warning("⚠️ No valid VC profiles found. Please ensure CSV processing completed successfully.")
            st.stop()

        # Validate number of profiles for matching
        valid_profiles = [p for p in profiles if isinstance(p.get("embedding"), list)]
        if len(valid_profiles) < 1:
            st.warning("⚠️ At least 1 valid VC profile with embeddings is required for matching.")
            st.stop()

        # Match founder to VCs
        matcher = FounderMatcherAgent(founder_embedding)
        try:
            top_matches = matcher.match(top_k=5)
            top_vc_urls = [m["url"].strip().lower() for m in top_matches]
            st.write(f"✅ Found {len(top_matches)} top VC matches: {[m['name'] for m in top_matches]}")
        except Exception as e:
            st.error(f"❌ Error matching VCs: {str(e)}")
            top_matches = []
            top_vc_urls = []

        # Generate and display matches
        if top_matches:
            st.subheader("🎯 Top 5 VC Matches")
            valid_matches = []
            for match in top_matches:
                if not all(key in match and match[key] for key in ["name", "url", "strategy_summary", "score"]):
                    st.warning(f"⚠️ Skipping invalid match for {match.get('name', 'unknown')}: missing fields {[k for k in ['name', 'url', 'strategy_summary', 'score'] if k not in match or not match[k]]}")
                    continue
                valid_matches.append(match)

            for match in valid_matches:
                try:
                    prompt = f"""
You are a senior VC advisor helping a startup founder find the best venture capital firms for their company.

Founder Profile:
{combined_input}

VC Profile:
- Name: {match['name']}
- Strategy Summary: {match['strategy_summary'][:500] if match['strategy_summary'] else 'No strategy summary available'}
- Strategic Tags: {', '.join(match.get('strategic_tags', []))}
- Portfolio Size: {match.get('portfolio_size', 0)} companies
- Category: {match.get('category', 'Uncategorized')}

Your task is to explain why this VC is a strong match for the founder's startup. Be specific, referencing the founder's product stage, customer type, go-to-market strategy, or other relevant details from their profile. Highlight aspects of the VC’s strategy, focus, or portfolio that align with the founder’s needs.

Respond in this format:
**Why {match['name']} is a Match**:
This VC specializes in [area]. It is a strong match for your business because [detailed justification, 2–3 sentences].
"""
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a precise and insightful VC advisor."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=200
                    )
                    match['rationale'] = response.choices[0].message.content.strip()
                except Exception as e:
                    match['rationale'] = f"(Rationale generation failed: {str(e)})"
                    st.warning(f"⚠️ Rationale generation error for {match['name']}: {str(e)}")

            if valid_matches:
                st.dataframe(
                    pd.DataFrame([
                        {
                            "Name": m["name"],
                            "URL": m["url"],
                            "Category": m.get("category", "Uncategorized"),
                            "Score": f"{m['score']:.2f}",
                            "Rationale": m.get("rationale", "No rationale available.")
                        } for m in valid_matches
                    ]),
                    use_container_width=True
                )
                with st.expander("📝 Detailed Match Justifications"):
                    for match in valid_matches:
                        st.markdown(f"**{match['name']}** — [{match['url']}]({match['url']})")
                        st.markdown(f"• Category: {match.get('category', 'Uncategorized')}  |  Score: {match['score']:.2f}")
                        st.markdown(f"{match.get('rationale', 'No rationale available.')}")
                        st.markdown("---")
            else:
                st.warning("⚠️ No valid VC matches after filtering. Please ensure valid VC profiles were generated from the CSV.")
        else:
            st.warning("⚠️ No VC matches found. Please ensure valid VC profiles were generated from the CSV.")

    except KeyError as e:
        st.error(f"❌ Profile data error in matching: Missing field {str(e)}. Please upload a new CSV to reset VC profiles.")
        st.stop()

    # Clustering, categorization, and visualization
    try:
        if len(valid_profiles) < 2:
            st.warning("⚠️ At least 2 valid VC profiles with embeddings are required for clustering and visualization.")
            st.stop()

        # Apply K-means clustering
        clustering_agent = ClusterInterpreterAgent(api_key=openai_api_key)
        n_clusters = min(len(valid_profiles), 4)  # Ensure n_clusters <= n_samples
        profiles = clustering_agent.assign_kmeans_clusters(n_clusters=n_clusters)

        # Categorize clusters
        categorizer = CategorizerAgent(api_key=openai_api_key)
        profiles = categorizer.categorize_clusters()

        # Apply PCA for visualization
        valid_embeddings = [p["embedding"] for p in profiles if isinstance(p.get("embedding"), list)]
        pca = PCA(n_components=2)
        coords = pca.fit_transform(valid_embeddings)
        for i, p in enumerate(profiles):
            p["pca_x"], p["pca_y"] = float(coords[i][0]), float(coords[i][1])
        save_vc_profiles(profiles)

        # Log PCA variance for verification
        with st.expander("🔍 PCA Variance Details"):
            st.write(f"PC1 Variance: {pca.explained_variance_ratio_[0] * 100:.1f}%")
            st.write(f"PC2 Variance: {pca.explained_variance_ratio_[1] * 100:.1f}%")

        # Transform founder embedding
        try:
            founder_2d = pca.transform([founder_embedding])[0]
        except ValueError as e:
            st.error(f"❌ PCA transformation failed: {str(e)}. Please upload a new CSV to reset VC profiles.")
            st.stop()

        # Generate intuitive dimension labels
        dim_agent = DimensionExplainerAgent(api_key=openai_api_key)
        dim_labels = {
            "x_label": "Investment Stage Focus",
            "y_label": "Sector Preference",
            "x_description": "Distinguishes VCs by their focus on early-stage vs. growth-stage startups.",
            "y_description": "Separates VCs by their preference for technology-driven vs. non-tech sectors.",
            "x_variance": pca.explained_variance_ratio_[0],
            "y_variance": pca.explained_variance_ratio_[1],
        }
        try:
            dim_agent.generate_axis_labels(profiles=valid_profiles, pca=pca)
            loaded_labels = dim_agent.load_dimension_labels()
            dim_labels.update(loaded_labels)
            dim_labels["x_variance"] = pca.explained_variance_ratio_[0]
            dim_labels["y_variance"] = pca.explained_variance_ratio_[1]
        except Exception as e:
            st.warning(f"⚠️ Error generating dimension labels: {str(e)}. Using default labels.")

        # Generate cluster map
        viz_agent = VisualizationAgent(api_key=openai_api_key)
        fig, labels = viz_agent.generate_cluster_map(
            profiles=profiles,
            coords_2d=coords,
            pca=pca,
            dimension_labels=dim_labels,
            founder_embedding_2d=founder_2d,
            founder_cluster_id=None,
            top_match_names=top_vc_urls,
        )

        # Generate category narratives
        category_narratives = {}
        unique_categories = sorted(set(p["category"] for p in profiles if p.get("category")))
        for category in unique_categories:
            category_profiles = [p for p in profiles if p.get("category") == category]
            rationale = category_profiles[0].get("category_rationale", "No rationale provided.") if category_profiles else ""
            prompt = f"""
You are a senior VC analyst creating a narrative for a group of venture capital firms in the '{category}' category.

Input:
- Category Rationale: {rationale}
- Sample VCs: {[p['name'] for p in category_profiles[:3]]}

Your task is to write a concise narrative (2–3 sentences) describing what makes this category unique. Focus on their investment thesis, portfolio focus, or cultural mindset. Use founder-friendly language to help startups understand this group.

Return the narrative directly as plain text.
"""
            try:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a clear and insightful VC analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=150
                )
                category_narratives[category] = response.choices[0].message.content.strip()
            except Exception as e:
                category_narratives[category] = f"(Narrative generation failed: {str(e)})"

        # Display visualization
        st.subheader("📊 VC Landscape Visualization")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"**🧭 X-Axis ({labels['x_label']}, {labels.get('x_variance', 0.0) * 100:.1f}%):** {labels.get('x_description', 'Represents variance in investment focus.')}")
            st.markdown(f"**🧭 Y-Axis ({labels['y_label']}, {labels.get('y_variance', 0.0) * 100:.1f}%):** {labels.get('y_description', 'Represents variance in strategic approach.')}")

            # Display category narratives
            st.subheader("📚 VC Category Descriptions")
            for category in unique_categories:
                narrative = category_narratives.get(category, "No narrative available.")
                st.markdown(f"**{category}**: {narrative}")
                st.markdown("---")
        else:
            st.warning("⚠️ Failed to generate visualization.")
    except KeyError as e:
        st.error(f"❌ Profile data error in clustering/visualization: Missing field {str(e)}. Please upload a new CSV to reset VC profiles.")
    except Exception as e:
        st.error(f"❌ Error during clustering/visualization: {str(e)}")
else:
    if founder_embedding:
        st.info("ℹ️ Please upload a CSV with VC URLs to generate matches and visualization.")
