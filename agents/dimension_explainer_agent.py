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
            name = p.get("name", "")
            cat = (p.get("category") or "Unknown").split("\n")[0].replace("Category:", "").strip()
            rationale_line = next((line for line in p.get("strategy_summary", "").splitlines() if line.lower().startswith("rationale")), "")
            summaries.append(f"{name}: {cat} — {rationale_line.strip()}")

        prompt = f"""
You are a strategy analyst reviewing a 2D PCA visualization of venture capital firms.

Each point is a VC, embedded based on their investment behavior and strategy. The PCA reveals two dimensions.

Here are sample VC firms with their assigned strategic category and rationale:
{chr(10).join(summaries[:30])}

Please propose intuitive names and directional descriptions for the X and Y axes.

Respond in this format:
Dimension 1: <short label> — <left vs right description>
Dimension 2: <short label> — <bottom vs top description>
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=500
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
                "y_description": y_description.strip()
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
