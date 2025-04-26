import plotly.express as px
import pandas as pd
import numpy as np
import os
import json
from sklearn.decomposition import PCA

VC_PROFILE_PATH = "outputs/vc_profiles.json"
DIMENSION_LABELS_PATH = "outputs/dimension_labels.json"
CLUSTER_LABELS_PATH = "outputs/cluster_labels.json"

class VisualizationAgent:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.color_palette = px.colors.qualitative.Safe  # Discrete color palette

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
            "y_description": ""
        }
    
    def load_cluster_labels(self):
        if os.path.exists(CLUSTER_LABELS_PATH):
            with open(CLUSTER_LABELS_PATH, "r") as f:
                return json.load(f)
        return {}

    def generate_cluster_map(self, founder_embedding_2d=None, founder_cluster_id=None):
        profiles = self.load_profiles()
        embeddings = [p["embedding"] for p in profiles if isinstance(p.get("embedding"), list)]

        if not embeddings:
            return None, {}

        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(embeddings)

        for profile, (x, y) in zip(profiles, coords):
            profile["pca_x"] = float(x)
            profile["pca_y"] = float(y)

        cluster_labels = self.load_cluster_labels()

        df = pd.DataFrame({
            "VC Name": [p["name"] for p in profiles],
            "Cluster ID": [p.get("cluster_id", -1) for p in profiles],
            "Cluster Name": [cluster_labels.get(str(p.get("cluster_id", -1)), {}).get("name", f"Cluster {p.get('cluster_id', -1)}") for p in profiles],
            "X": [p["pca_x"] for p in profiles],
            "Y": [p["pca_y"] for p in profiles],
            "Strategy Summary": [p.get("strategy_summary", "") for p in profiles],
            "Motivational Signals": [", ".join(p.get("motivational_signals", [])) for p in profiles],
            # "Portfolio Size": [p.get("portfolio_size") for p in profiles],
        })

        dim_labels = self.load_dimension_labels()
        unique_clusters = sorted(df["Cluster Name"].unique())
        cluster_color_map = {
            cluster_name: self.color_palette[i % len(self.color_palette)]
            for i, cluster_name in enumerate(unique_clusters)
        }

        df["Color"] = df["Cluster Name"].map(cluster_color_map)

        fig = px.scatter(
            df,
            x="X",
            y="Y",
            color="Cluster Name",
            color_discrete_map=cluster_color_map,
            

            labels={
                "X": dim_labels.get("x_label", "PC1"),
                "Y": dim_labels.get("y_label", "PC2")
            },
            title="🧭 VC Landscape by Strategic Identity",
            width=950,
            height=650
        )
        # Clean strategic tags into a readable string
df["Strategic Tags"] = df["Strategic Tags"].apply(lambda tags: ", ".join(tags) if isinstance(tags, list) else "")

# Update hover template
fig.update_traces(
    hovertemplate=
        "<b>%{customdata[0]}</b><br>" +  # VC Name
        "Strategic Category: %{customdata[1]}<br>" +
        "Strategic Tags: %{customdata[2]}<extra></extra>",
    customdata=np.stack((df["VC Name"], df["Strategy Summary"], df["Strategic Tags"]), axis=-1)
)
        fig.update_traces(
           hovertemplate="<br>".join([
               "Name: %{customdata[0]}",
               "Cluster: %{customdata[1]}",
               "Focus: %{customdata[2]}",
               "Signals: %{customdata[3]}"
          ])
        )


        if founder_embedding_2d is not None:
            founder_x, founder_y = founder_embedding_2d
            fig.add_scatter(
                x=[founder_x],
                y=[founder_y],
                mode="markers",
                marker_symbol="star",
                marker_size=20,
                marker_color="gold",
                name="Founder Idea"
            )

        fig.update_layout(
            xaxis_title=f"{dim_labels['x_label']} ({pca.explained_variance_ratio_[0]*100:.1f}% variance)",
            yaxis_title=f"{dim_labels['y_label']} ({pca.explained_variance_ratio_[1]*100:.1f}% variance)"
        )

        return fig, dim_labels
