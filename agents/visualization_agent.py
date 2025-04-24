import json
import os
import plotly.express as px
import pandas as pd

VC_PROFILE_PATH = "outputs/vc_profiles.json"

class VisualizationAgent:
    def __init__(self):
        pass

    def generate_cluster_map(self):
        profiles = self.load_profiles()
        data = []

        for p in profiles:
            coords = p.get("coordinates")
            if coords and isinstance(coords, list) and len(coords) == 2 and all(isinstance(c, (int, float)) for c in coords):
                data.append({
                    "name": p.get("name", "Unnamed VC"),
                    "x": coords[0],
                    "y": coords[1],
                    "cluster_id": p.get("cluster_id"),
                    "category": (p.get("category") or "").split("\n")[0].replace("Category:", "").strip()
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
            title="VC Landscape by Strategic Category",
            labels={"x": "Dimension 1", "y": "Dimension 2"},
            width=900,
            height=600
        )

        fig.update_traces(marker=dict(size=10, opacity=0.7))
        return fig


def generate_cluster_map(self):
    profiles = self.load_profiles()
    data = []

    for p in profiles:
        coords = p.get("coordinates")
        if coords and isinstance(coords, list) and len(coords) == 2 and all(isinstance(c, (int, float)) for c in coords):
            data.append({
                "name": p.get("name", "Unnamed VC"),
                "x": coords[0],
                "y": coords[1],
                "cluster_id": p.get("cluster_id"),
                "category": (p.get("category") or "").split("\n")[0].replace("Category:", "").strip()
            })

    if not data:
        return None  # Prevents crashing if no valid data

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

