import json
import os
import plotly.express as px
import pandas as pd
from agents.dimension_explainer_agent import DimensionExplainerAgent

VC_PROFILE_PATH = "outputs/vc_profiles.json"
DIMENSION_LABELS_PATH = "outputs/dimension_labels.json"

class VisualizationAgent:
    def __init__(self, api_key=None):
        self.api_key = api_key

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

    def regenerate_axis_labels(self):
        if not self.api_key:
            return {"x_label": "Dimension 1", "y_label": "Dimension 2"}
        agent = DimensionExplainerAgent(api_key=self.api_key)
        return agent.generate_axis_labels()

    def generate_cluster_map(self, force_refresh=False):
        if force_refresh and self.api_key:
            self.regenerate_axis_labels()

        profiles = self.load_profiles()
        labels = self.load_axis_labels()
        data = []

        for p in profiles:
            if p.get("coordinates") and p["coordinates"][0] is not None and p.get("cluster_id") is not None:
                rationale_line = next((line for line in p.get("category", "").splitlines() if line.lower().startswith("rationale")), "")
                data.append({
                    "name": p["name"],
                    "x": p["coordinates"][0],
                    "y": p["coordinates"][1],
                    "category": (p.get("category") or "").split("\n")[0].replace("Category:", "").strip(),
                    "portfolio_size": p.get("portfolio_size", 0),
                    "rationale": rationale_line.strip()
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
            custom_data=["name", "category", "portfolio_size", "rationale"],
            title="🧭 VC Landscape by Strategic Identity",
            labels={
                "x": labels.get("x_label", "Dimension 1"),
                "y": labels.get("y_label", "Dimension 2")
            },
            color_discrete_sequence=px.colors.qualitative.Safe,
            width=950,
            height=600
        )

        fig.update_traces(
            marker=dict(size=10, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')),
            hovertemplate="<b>%{customdata[0]}</b><br>" +
                          "Category: %{customdata[1]}<br>" +
                          "Portfolio Size: %{customdata[2]}<br>" +
                          "%{customdata[3]}<extra></extra>"
        )

        fig.update_layout(legend_title_text='Cluster Category')
        return fig
