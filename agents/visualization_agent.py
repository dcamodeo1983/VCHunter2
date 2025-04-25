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
        if not profiles:
            return None

        # Filter out profiles with missing coordinates
        data = [p for p in profiles if p.get("coordinates") and None not in p["coordinates"]]
        if not data:
            return None

        df = pd.DataFrame([{
            "name": p["name"],
            "x": p["coordinates"][0],
            "y": p["coordinates"][1],
            "category": p.get("category", "Uncategorized"),
            "portfolio_size": p.get("portfolio_size", 0),
            "summary": p.get("strategy_summary", "")
        } for p in data])

        # Extract first rationale line from summary
        df["rationale"] = df["summary"].apply(lambda x: next((line for line in x.split("\n") if "Rationale:" in line), ""))

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
            labels={"x": "X", "y": "Y"},
            width=950,
            height=600
        )

        # Overlay founder position
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

        return fig

    def regenerate_axis_labels(self):
        # Stubbed method — replace with LLM logic if available
        labels = {
            "x_label": "Thesis Depth",
            "x_description": "Firms on the right articulate highly detailed investment theses, often with academic or technical framing. Left-side firms focus on generalist, opportunistic, or ambiguous strategies.",
            "y_label": "Stage Specialization",
            "y_description": "Higher values represent firms focused on early-stage, pre-seed, and innovation bets. Lower values lean toward growth-stage, scaling, or later-stage capital deployments."
        }
        with open(DIMENSION_LABELS_PATH, "w") as f:
            json.dump(labels, f, indent=2)
