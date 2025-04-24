import json
import os
import plotly.express as px
import pandas as pd

VC_PROFILE_PATH = "outputs/vc_profiles.json"
DIMENSION_LABELS_PATH = "outputs/dimension_labels.json"

class VisualizationAgent:
    def __init__(self):
        pass

    def load_profiles(self):
        if os.path.exists(VC_PROFILE_PATH):
            with open(VC_PROFILE_PATH, "r") as f:
                return json.load(f)
        return []

    def load_axis_labels(self):
        try:
            if os.path.exists(DIMENSION_LABELS_PATH):
                with open(DIMENSION_LABELS_PATH, "r") as f:
                    return json.load(f)
        except json.JSONDecodeError:
            print("⚠️ Invalid or empty dimension_labels.json — using defaults.")
        return {"x_label": "Dimension 1", "y_label": "Dimension 2"}

    def generate_cluster_map(self):
        profiles = self.load_profiles()
        labels = self.load_axis_labels()
        data = []

        for p in profiles:
            if p.get("coordinates") and p["coordinates"][0] is not None and p.get("cluster_id") is not None:
                tooltip = f"{p['name']}\nCategory: {p.get('category', 'N/A')}\nPortfolio Size: {p.get('portfolio_size', 0)}"
                strategy = p.get("strategy_summary", "")
                rationale_line = next((line for line in strategy.splitlines() if line.lower().startswith("rationale")), "")
                if rationale_line:
                    tooltip += f"\n{rationale_line.strip()}"

                data.append({
                    "name": p["name"],
                    "x": p["coordinates"][0],
                    "y": p["coordinates"][1],
                    "category": (p.get("category") or "").split("\n")[0].replace("Category:", "").strip(),
                    "tooltip": tooltip
                })

        if not data:
            return None

        df = pd.DataFrame(data)

        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="category",
            hover_name="name",
            hover_data={"x": False, "y": False, "tooltip": True},
            title="🧭 VC Landscape by Strategic Identity",
            labels={"x": labels["x_label"], "y": labels["y_label"]},
            color_discrete_sequence=px.colors.qualitative.Safe,
            width=950,
            height=600
        )

        fig.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
        fig.update_layout(legend_title_text='Cluster Category')

        return fig
