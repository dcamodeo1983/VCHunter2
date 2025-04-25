import json
import os
import plotly.express as px
import pandas as pd

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
            pass
        return {
            "x_label": "Dimension 1",
            "y_label": "Dimension 2",
            "x_description": "",
            "y_description": ""
        }

    def generate_cluster_map(self, founder_embedding_2d=None):
        profiles = self.load_profiles()
        labels = self.load_axis_labels()
        data = []

        for p in profiles:
            if p.get("coordinates") and p["coordinates"][0] is not None and p.get("cluster_id") is not None:
                rationale_line = next(
                    (line for line in p.get("strategy_summary", "").splitlines()
                     if line.lower().startswith("rationale")),
                    ""
                )
                tooltip = (
                    f"{p['name']}\n"
                    f"Category: {p.get('category', 'N/A')}\n"
                    f"Portfolio Size: {p.get('portfolio_size', 0)}\n"
                    f"{rationale_line.strip()}"
                )

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
            hover_data={"tooltip": True},
            labels={"x": labels["x_label"], "y": labels["y_label"]},
            title="🧭 VC Landscape by Strategic Identity",
            color_discrete_sequence=px.colors.qualitative.Safe,
            width=950,
            height=600
        )

        fig.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
        fig.update_layout(legend_title_text='Cluster Category')

        if founder_embedding_2d:
            fig.add_scatter(
                x=[founder_embedding_2d[0]],
                y=[founder_embedding_2d[1]],
                mode="markers+text",
                name="Your Startup",
                marker=dict(symbol='star', size=16, color='black'),
                text=["⭐ You"],
                textposition="top center",
                showlegend=True
            )

        return fig
