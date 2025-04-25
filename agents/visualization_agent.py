
import json
import os
import plotly.express as px
import pandas as pd
from openai import OpenAI

VC_PROFILE_PATH = "outputs/vc_profiles.json"
DIMENSION_LABELS_PATH = "outputs/dimension_labels.json"
PCA_TERMS_PATH = "outputs/pca_terms.json"  # assumes we save top PCA terms there

class VisualizationAgent:
    def __init__(self, api_key=None):
        self.api_key = api_key
        if api_key:
            self.client = OpenAI(api_key=api_key)

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
            "x_label": "PC1",
            "x_description": "",
            "x_variance": 0.0,
            "y_label": "PC2",
            "y_description": "",
            "y_variance": 0.0
        }

    def generate_cluster_map(self, founder_embedding_2d=None):
        profiles = self.load_profiles()
        if not profiles:
            return None, {}

        data = [p for p in profiles if p.get("coordinates") and None not in p["coordinates"]]
        if not data:
            return None, {}

        df = pd.DataFrame([{
            "name": p["name"],
            "x": p["coordinates"][0],
            "y": p["coordinates"][1],
            "category": p.get("category", "Uncategorized"),
            "portfolio_size": p.get("portfolio_size", 0),
            "summary": p.get("strategy_summary", "")
        } for p in data])

        df["rationale"] = df["summary"].apply(
            lambda x: next((line for line in x.split("\n") if "Rationale:" in line), "")
        )

        labels = self.load_axis_labels()

        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="category",
            color_discrete_sequence=px.colors.qualitative.Safe,
            hover_data={
                "name": True,
                "category": True,
                "portfolio_size": True,
                "rationale": True,
                "x": False,
                "y": False
            },
            title="🧭 VC Landscape by Strategic Identity",
            width=1100,
            height=500,
            labels={"x": labels["x_label"], "y": labels["y_label"]}
        )

        if founder_embedding_2d and len(founder_embedding_2d) == 2:
            fig.add_scatter(
                x=[founder_embedding_2d[0]],
                y=[founder_embedding_2d[1]],
                mode="markers+text",
                marker=dict(symbol='star', size=16, color='black'),
                text=["⭐ You"],
                textposition="top center",
                name="You"
            )

        fig.update_layout(
            legend_title_text="Cluster Category",
            title_font_size=20,
            font=dict(size=13)
        )

        return fig, labels

    def regenerate_axis_labels(self):
        if not os.path.exists(PCA_TERMS_PATH):
            return  # Must include top terms for each principal component

        with open(PCA_TERMS_PATH, "r") as f:
            pca_terms = json.load(f)

        def query_label(pc_terms, axis):
            terms_str = ", ".join(pc_terms)
            prompt = f"You are a VC analyst. What does this semantic axis represent based on the terms: {terms_str}? Return a short label and a 1-2 sentence description."
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6
            )
            return response.choices[0].message.content.strip().split("\n", 1)

        x_label, x_description = query_label(pca_terms["PC1_terms"], "x")
        y_label, y_description = query_label(pca_terms["PC2_terms"], "y")

        axis_labels = {
            "x_label": x_label,
            "x_description": x_description,
            "x_variance": pca_terms.get("PC1_variance", 0.0),
            "y_label": y_label,
            "y_description": y_description,
            "y_variance": pca_terms.get("PC2_variance", 0.0)
        }

        with open(DIMENSION_LABELS_PATH, "w") as f:
            json.dump(axis_labels, f, indent=2)
