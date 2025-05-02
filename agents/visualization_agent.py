
import plotly.express as px
import pandas as pd
import numpy as np
import json
import os
from collections import Counter

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

    def load_cluster_descriptions(self):
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

        for profile, (x, y) in zip(profiles, coords_2d):
            profile["pca_x"] = float(x)
            profile["pca_y"] = float(y)

        dim_labels = dimension_labels or self.load_dimension_labels()
        top_match_names = top_match_names or []
        normalized_top_names = [name.strip().lower() for name in top_match_names]

        df = pd.DataFrame({
            "VC Name": [p.get("name") for p in profiles],
            "Category": [p.get("category", "Uncategorized") for p in profiles],
            "X": [p["pca_x"] for p in profiles],
            "Y": [p["pca_y"] for p in profiles],
            "Strategic Tags": [", ".join(p.get("strategic_tags", [])) for p in profiles],
            "Motivational Signals": [", ".join(p.get("motivational_signals", [])) for p in profiles],
        })

        unique_categories = sorted(df["Category"].unique())
        category_color_map = {
            cat: self.color_palette[i % len(self.color_palette)]
            for i, cat in enumerate(unique_categories)
        }
        df["Color"] = df["Category"].map(category_color_map)

        df["Normalized VC Name"] = df["VC Name"].str.strip().str.lower()
        df["Symbol"] = df["Normalized VC Name"].apply(
            lambda name: "star" if name in normalized_top_names else "circle"
        )
        df["Size"] = df["Normalized VC Name"].apply(
            lambda name: 12 if name in normalized_top_names else 7
        )

        fig = px.scatter(
            df,
            x="X",
            y="Y",
            color="Category",
            color_discrete_map=category_color_map,
            symbol="Symbol",
            size="Size",
            size_max=15,
            hover_name="VC Name",
            custom_data=["VC Name", "Category", "Strategic Tags", "Motivational Signals"],
            labels={
                "X": f"{dim_labels.get('x_label', 'PC1')} ({dim_labels.get('x_variance', 0.0)*100:.1f}%)",
                "Y": f"{dim_labels.get('y_label', 'PC2')} ({dim_labels.get('y_variance', 0.0)*100:.1f}%)",
            },
            title="🧭 VC Landscape by Strategic Category",
            width=950,
            height=650,
        )

        fig.update_traces(
            hovertemplate="<br>".join([
                "<b>%{customdata[0]}</b>",
                "Category: %{customdata[1]}",
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
                marker=dict(
                    symbol="star",
                    size=22,
                    color="gold",
                    line=dict(color="black", width=2),
                ),
                name="Founder Idea",
                showlegend=True,
            )

        cluster_sizes = Counter(df["Category"])
        cluster_descriptions = self.load_cluster_descriptions()
        sorted_descriptions = sorted(cluster_descriptions.items(), key=lambda x: cluster_sizes.get(x[0], 0))

        description_block = "\n".join([f"**{cat}**: {desc}" for cat, desc in sorted_descriptions])

        return fig, {**dim_labels, "descriptions_markdown": description_block}
