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
    data = []

    for p in profiles:
        coords = p.get("coordinates")
        if coords and isinstance(coords, list) and len(coords) == 2 and all(isinstance(c, (int, float)) for c in coords):
            category = (p.get("category") or "").split("\n")[0].replace("Category:", "").strip()
            rationale = ""
            if p.get("category") and "Rationale:" in p["category"]:
                rationale = p["category"].split("Rationale:")[-1].strip()

            data.append({
                "name": p.get("name", "Unnamed VC"),
                "x": coords[0],
                "y": coords[1],
                "cluster_id": p.get("cluster_id"),
                "category": category,
                "rationale": rationale
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
        hover_data=["rationale"],
        title="VC Landscape by Strategic Category",
        labels={"x": "Dimension 1", "y": "Dimension 2"},
        width=900,
        height=600,
        color_discrete_sequence=px.colors.qualitative.Bold  # 🌈 High-contrast palette
    )

    fig.update_traces(marker=dict(size=10, opacity=0.8))
    return fig

