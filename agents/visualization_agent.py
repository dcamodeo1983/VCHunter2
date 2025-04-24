import json
import os
import plotly.express as px
import pandas as pd

VC_PROFILE_PATH = "outputs/vc_profiles.json"

class VisualizationAgent:
    def __init__(self):
        pass

    def load_profiles(self):
        if os.path.exists(VC_PROFILE_PATH):
            with open(VC_PROFILE_PATH, "r") as f:
                return json.load(f)
        return []

    def generate_cluster_map(self):
        profiles = self.load_profiles()
        data = [
            {
                "name": p["name"],
                "x": p["coordinates"][0] if p.get("coordinates") else None,
                "y": p["coordinates"][1] if p.get("coordinates") else None,
                "cluster_id": p.get("cluster_id"),
                "category": (p.get("category") or "").split("\n")[0].replace("Category:", "").strip()
            }
            for p in profiles
            if p.get("coordinates") and p.get("cluster_id") is not None
        ]

        df = pd.DataFrame(data)

        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="category",
            hover_name="name",
            title="VC Landscape by Strategic Category",
            labels={"x": "Dimension 1", "y": "Dimension 2"},
            width=900,
            height=600
        )

        fig.update_traces(marker=dict(size=10, opacity=0.7))
        return fig
