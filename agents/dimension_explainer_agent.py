import os
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI

VC_PROFILE_PATH = "outputs/vc_profiles.json"
DIMENSION_LABELS_PATH = "outputs/dimension_labels.json"
EMBEDDING_DIM = 768
TOP_N_TERMS = 6

class DimensionExplainerAgent:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)

    def load_profiles(self):
        if os.path.exists(VC_PROFILE_PATH):
            with open(VC_PROFILE_PATH, "r") as f:
                return json.load(f)
        return []

    def save_labels(self, labels):
        with open(DIMENSION_LABELS_PATH, "w") as f:
            json.dump(labels, f, indent=2)

    def generate_axis_labels(self):
        profiles = self.load_profiles()
        embeddings = [p["embedding"] for p in profiles if isinstance(p.get("embedding"), list)]

        if not embeddings:
            return {}

        X = np.array(embeddings)
        pca = PCA(n_components=2, random_state=42)
        pca.fit(X)

        variance_ratios = pca.explained_variance_ratio_

        summaries = []
        for p in profiles:
            name = p.get("name", "")
            cat = (p.get("category") or "Unknown").split("\n")[0].replace("Category:", "").strip()
            rationale_line = next((line for line in p.get("strategy_summary", "").splitlines() if line.lower().startswith("rationale")), "")
            summaries.append(f"{name}: {cat} — {rationale_line.strip()}")

        prompt = f"""
You are a strategic analyst reviewing a 2D PCA map of venture capital firms.

The PCA dimensions have the following explained variances:
- Dimension 1: {variance_ratios[0]*100:.1f}% of variance
- Dimension 2: {variance_ratios[1]*100:.1f}% of variance

Each point represents a VC firm positioned by their investment strategies.

Here are some examples of VC summaries in this space:
{chr(10).join(summaries[:30])}

Propose:
- A short label for Dimension 1 (X axis)
- A short label for Dimension 2 (Y axis)
- Brief directional descriptions for each

Respond in this format:
Dimension 1: <short label> — <left to right interpretation>
Dimension 2: <short label> — <bottom to top interpretation>
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=600
            )
            result = response.choices[0].message.content.strip()

            # Parse GPT response
            lines = result.splitlines()
            x_label, x_description = lines[0].split("—")
            y_label, y_description = lines[1].split("—")

            labels = {
                "x_label": x_label.replace("Dimension 1:", "").strip(),
                "y_label": y_label.replace("Dimension 2:", "").strip(),
                "x_description": x_description.strip(),
                "y_description": y_description.strip(),
                "x_variance": variance_ratios[0],
                "y_variance": variance_ratios[1]
            }

            self.save_labels(labels)
            return labels

        except Exception as e:
            return {
                "x_label": "Dimension 1",
                "y_label": "Dimension 2",
                "x_description": "Left = ?, Right = ?",
                "y_description": "Bottom = ?, Top = ?",
                "error": str(e)
            }
