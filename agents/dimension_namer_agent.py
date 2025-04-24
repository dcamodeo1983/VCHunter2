import json
import os
from openai import OpenAI

VC_PROFILE_PATH = "outputs/vc_profiles.json"
DIMENSION_LABELS_PATH = "outputs/pca_dimension_labels.json"

class DimensionNamerAgent:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def load_profiles(self):
        if os.path.exists(VC_PROFILE_PATH):
            with open(VC_PROFILE_PATH, "r") as f:
                return json.load(f)
        return []

    def name_pca_axes(self):
        profiles = self.load_profiles()
        df = [
            (p["name"], p["coordinates"], p["strategy_summary"])
            for p in profiles
            if p.get("coordinates") and p.get("strategy_summary")
        ]

        if not df:
            return {}

        dimension_labels = {}

        for i in range(2):  # For PC1 and PC2
            extremes = sorted(df, key=lambda x: x[1][i])
            low_end = extremes[:5]
            high_end = extremes[-5:]

            low_summary = "\n".join([f"- {x[0]}: {x[2]}" for x in low_end])
            high_summary = "\n".join([f"- {x[0]}: {x[2]}" for x in high_end])

            prompt = f"""
You are a strategic analyst. Based on the summaries of VCs at opposite ends of this dimension:

High end:
{high_summary}

Low end:
{low_summary}

What does this dimension represent? Return your answer as a short axis label and a one-sentence rationale.
Format:
AxisLabel: <label>
Rationale: <reason>
"""

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=200
            )
            content = response.choices[0].message.content.strip()
            dimension_labels[f"PC{i+1}"] = content

        with open(DIMENSION_LABELS_PATH, "w") as f:
            json.dump(dimension_labels, f, indent=2)

        return dimension_labels
