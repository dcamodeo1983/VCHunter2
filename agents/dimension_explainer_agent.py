import json
import os
from openai import OpenAI

VC_PROFILE_PATH = "outputs/vc_profiles.json"
DIMENSION_LABELS_PATH = "outputs/dimension_labels.json"

class DimensionExplainerAgent:
    def __init__(self, api_key):
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
        summaries = []

        for p in profiles:
            category = (p.get("category") or "Unknown").split("\n")[0].replace("Category:", "").strip()
            rationale_line = next((line for line in p.get("strategy_summary", "").splitlines() if line.lower().startswith("rationale")), "")
            summaries.append(f"{p['name']}: {category} — {rationale_line.strip()}")

        prompt = f"""
You are an expert data analyst and VC strategist reviewing the result of a PCA dimensionality reduction and clustering analysis of venture capital firms.

Each point represents a VC firm. The x and y dimensions are latent principal components derived from the embedding of their investment behavior and stated theses.

Below is a sample of the categorized firms and their rationale:

{chr(10).join(summaries[:30])}

Based on the firms and how they’re grouped, suggest short, human-readable names for the two principal axes that appear on a 2D visualization.

Format your response exactly like this:
Dimension 1: <label> — <brief description>
Dimension 2: <label> — <brief description>
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=500
            )
            result = response.choices[0].message.content.strip()

            lines = result.splitlines()
            labels = {
                "x_label": lines[0].replace("Dimension 1:", "").split("—")[0].strip(),
                "y_label": lines[1].replace("Dimension 2:", "").split("—")[0].strip()
            }
            self.save_labels(labels)
            return labels

        except Exception as e:
            return {"x_label": "Dimension 1", "y_label": "Dimension 2", "error": str(e)}
