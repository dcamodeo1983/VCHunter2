import json
import os
import openai
from sklearn.decomposition import PCA
import numpy as np

VC_PROFILE_PATH = "outputs/vc_profiles.json"
DIMENSION_LABELS_PATH = "outputs/dimension_labels.json"

class DimensionExplainerAgent:
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)

    def generate_axis_labels(self, profiles=None, pca=None):
        if not profiles or pca is None:
            if not os.path.exists(VC_PROFILE_PATH):
                return
            with open(VC_PROFILE_PATH, "r") as f:
                profiles = json.load(f)
            embeddings = [p["embedding"] for p in profiles if isinstance(p.get("embedding"), list)]
            if len(embeddings) < 2:
                return
            X = np.array(embeddings)
            pca = PCA(n_components=2)
            pca.fit_transform(X)
        
        explained_variance = pca.explained_variance_ratio_
        # Get PCA loadings (feature contributions)
        loadings = pca.components_
        top_features_pc1 = np.argsort(np.abs(loadings[0]))[-3:]  # Top 3 features for PC1
        top_features_pc2 = np.argsort(np.abs(loadings[1]))[-3:]  # Top 3 features for PC2

        # Sample VC profiles for context
        sample_profiles = profiles[:5]  # Limit to 5 for brevity
        vc_summaries = "\n".join([f"- {p['name']}: {p.get('strategy_summary', '')[:200]}" for p in sample_profiles])
        vc_tags = "\n".join([f"- {p['name']}: {', '.join(p.get('strategic_tags', []))}" for p in sample_profiles])

        prompt = f"""
You are a senior data scientist interpreting principal components (PCs) for a startup founder. You have performed PCA on venture capital firm embeddings, reducing them to 2 dimensions (PC1 and PC2). Your task is to provide intuitive, founder-friendly labels and descriptions for PC1 and PC2 based on the VC profiles and PCA loadings.

PCA Details:
- PC1 explains {explained_variance[0]*100:.2f}% of the variance.
- PC2 explains {explained_variance[1]*100:.2f}% of the variance.
- Top contributing features for PC1 (embedding dimensions): {list(top_features_pc1)}.
- Top contributing features for PC2 (embedding dimensions): {list(top_features_pc2)}.

VC Profile Summaries (sample):
{vc_summaries}

Strategic Tags (sample):
{vc_tags}

For each component, provide:
- A short label (e.g., "Seed vs. Growth Stage").
- A 1–2 sentence description explaining what the component represents (e.g., "This axis distinguishes VCs focusing on early-stage startups from those investing in scaled companies.").

Respond in JSON format:
```json
{{
  "x_label": "<label>",
  "x_description": "<description>",
  "y_label": "<label>",
  "y_description": "<description>",
  "x_variance": {explained_variance[0]},
  "y_variance": {explained_variance[1]}
}}
"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a clear and insightful data scientist."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=300
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:-3].strip()  # Remove ```json and ```
            labels = json.loads(content)
            labels["x_variance"] = float(explained_variance[0])
            labels["y_variance"] = float(explained_variance[1])

            with open(DIMENSION_LABELS_PATH, "w") as f:
                json.dump(labels, f, indent=2)

        except Exception as e:
            st.warning(f"⚠️ Error generating axis labels: {str(e)}")

    def load_dimension_labels(self):
        if os.path.exists(DIMENSION_LABELS_PATH):
            with open(DIMENSION_LABELS_PATH, "r") as f:
                return json.load(f)
        return {
            "x_label": "PC1",
            "y_label": "PC2",
            "x_description": "First principal component",
            "y_description": "Second principal component",
            "x_variance": 0.0,
            "y_variance": 0.0,
        }
