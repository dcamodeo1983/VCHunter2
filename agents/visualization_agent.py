import json
import os
import plotly.express as px
import pandas as pd
from openai import OpenAI

VC_PROFILE_PATH = "outputs/vc_profiles.json"
DIMENSION_LABELS_PATH = "outputs/dimension_labels.json"

class VisualizationAgent:
    def __init__(self, api_key=None):
        self.api_key = api_key
        if api_key:
            self.client = OpenAI(api_key=api_key)

    def load_profiles(self):
        if not os.path.exists(VC_PROFILE_PATH):
            return []
        try:
            with open(VC_PROFILE_PATH, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []

    def load_axis_labels(self):
        if not os.path.exists(DIMENSION_LABELS_PATH):
            return {}
        try:
            with open(DIMENSION_LABELS_PATH, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}

    def generate_cluster_map(self, founder_embedding_2d=None):
        profiles = self.load_profiles()
        data = [p for p in profiles if p.get("coordinates")]

        if not data:
            return None, {}

        df = pd.DataFrame({
            "name": [p["name"] for p in data],
            "x": [p["coordinates"][0] for p in data],
            "y": [p["coordinates"][1] for p in data],
            "cluster": [p.get("cluster_id", -1) for p in data],
            "label": [p.get("category", "Unknown") for p in data],
        })

        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="cluster",
            hover_data=["name", "label"],
            labels={"x": "PCA Dimension 1", "y": "PCA Dimension 2"},
            title="VC Strategy Landscape"
        )

        # Plot founder dot
        if founder_embedding_2d and len(founder_embedding_2d) == 2:
            fig.add_scatter(
                x=[founder_embedding_2d[0]],
                y=[founder_embedding_2d[1]],
                mode="markers+text",
                marker=dict(symbol="star", size=14, color="black"),
                text=["⭐ You"],
                textposition="top center",
                name="You"
            )

        fig.update_layout(showlegend=True)
        return fig, self.load_axis_labels()
