import plotly.express as px
import pandas as pd
import numpy as np
import json
import os

VC_PROFILE_PATH = "outputs/vc_profiles.json"
DIMENSION_LABELS_PATH = "outputs/dimension_labels.json"
CLUSTER_LABELS_PATH = "outputs/cluster_labels.json"


class VisualizationAgent:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.color_palette = px.colors.qualitative.Safe

    def load_profiles(self):
        if os.path.exists(VC_PROFILE_PATH):
            with open(VC_PROFILE_PATH, "r") as f:
                return json.load(f)
        return []

    def load_dimension_labels(self):
        if os.path.exists(DIMENSION_LABELS_PATH):
            with open(DIMENSION_LABELS_PATH, "r") as f:
                return json.load(f)
        return {
            "x_label": "PC1",
            "y_label": "PC2",
            "x_description": "",
            "y_description": "",
        }

    def load_cluster_labels(self):
        if os.path.exists(CLUSTER_LABELS_PATH):
            with open(CLUSTER_LABELS_PATH, "r") as f:
                return json.load(f)
        return {}

    def generate_cluster_map(
        self,
        profiles=None,
        coords_2d=None,
        pca=None,
        dimension_labels=None,
        founder_embedding_2d=None,
        founder_cluster_id=None,
        top_match_names=None,
    ):
        if not profiles or coords_2d is None or pca is None:
            return None, {}

        # Assign PCA coordinates
        for profile, (x, y) in zip(profiles, coords_2d):
            profile["pca_x"] = float(x)
            profile["pca_y"] = float(y)

        cluster_labels = self.load_cluster_labels()
        dim_labels = dimension_labels or self.load_dimension_labels()

        df = pd.DataFrame(
            {
                "VC Name": [p.get("name") for p in profiles],
                "Cluster ID": [p.get("cluster_id", -1) for p in profiles],
                "Cluster Name": [
                    cluster_labels.get(str(p.get("cluster_id", -1)), {}).get(
                        "name", f"Cluster {p.get('cluster_id', -1)}"
                    )
                    for p in profiles
                ],
                "X": [p["pca_x"] for p in profiles],
                "Y": [p["pca_y"] for p in profiles],
                "Strategic Tags": [
                    ", ".join(p.get("strategic_tags", [])) for p in profiles
                ],
                "Motivational Signals": [
                    ", ".join(p.get("motivational_signals", [])) for p in profiles
                ],
            }
        )

        unique_clusters = sorted(df["Cluster Name"].unique())
        cluster_color_map = {
            cluster_name: self.color_palette[i % len(self.color_palette)]
            for i, cluster_name in enumerate(unique_clusters)
        }
        df["Color"] = df["Cluster Name"].map(cluster_color_map)

        top_match_names = top_match_names or []
        normalized_top_names = [name.strip().lower() for name in top_match_names]
        df["Normalized VC Name"] = df["VC Name"].str.strip().str.lower()
        df["Symbol"] = df["Normalized VC Name"].apply(
            lambda name: "star" if name in normalized_top_names else "circle"
        )
        df["Size"] = df["Normalized VC Name"].apply(
            lambda name: 10 if name in normalized_top_names else 6
        )

        fig = px.scatter(
            df,
            x="X",
            y="Y",
            color="Cluster Name",
            color_discrete_map=cluster_color_map,
            symbol="Symbol", showlegend=False,
            size="Size",
            size_max=15,
            hover_name="VC Name",
            custom_data=[
                "VC Name",
                "Cluster Name",
                "Strategic Tags",
                "Motivational Signals",
            ],
            labels={
                "X": f"{dim_labels.get('x_label', 'PC1')} ({dim_labels.get('x_variance', 0.0)*100:.1f}%)",
                "Y": f"{dim_labels.get('y_label', 'PC2')} ({dim_labels.get('y_variance', 0.0)*100:.1f}%)",
            },
            title="🧭 VC Landscape by Strategic Identity",
            width=950,
            height=650,
        )

        fig.update_traces(
            hovertemplate="<br>".join(
                [
                    "<b>%{customdata[0]}</b>",
                    "Cluster: %{customdata[1]}",
                    "Focus: %{customdata[2]}",
                    "Signals: %{customdata[3]}",
                ]
            )
        )

        if founder_embedding_2d is not None:
            founder_x, founder_y = founder_embedding_2d
            fig.add_scatter(
                x=[founder_x],
                y=[founder_y],
                mode="markers",
                marker=dict(
                    symbol="star",
                    size=22,
                    color="gold",
                    line=dict(color="black", width=2),
                ),
                name="Founder Idea",
            )

        return fig, dim_labels
