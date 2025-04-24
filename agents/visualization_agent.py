import json
import os
import plotly.express as px
import pandas as pd

VC_PROFILE_PATH = "outputs/vc_profiles.json"
DIMENSION_LABELS_PATH = "outputs/dimension_labels.json"

class VisualizationAgent:
    def __init__(self):
        pass

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
            print("⚠️ Invalid or empty dimension_labels.json — using defaults.")
        return {"x_label": "Dimension 1", "y_label": "Dimension 2"}

def generate_cluster_map(self):
    profiles = self.load_profiles()
    labels = self.load_axis_labels()
    data = []

    for p in profiles:
        if p.get("coordinates") and p["coordinates"][0] is not None and p.get("cluster_id") is not None:
            rationale_line = next((line for line in p.get("category", "").splitlines() if line.lower().startswith("rationale")), "")
            data.append({
                "name": p["name"],
                "x": p["coordinates"][0],
                "y": p["coordinates"][1],
                "category": (p.get("category") or "").split("\n")[0].replace("Category:", "").strip(),
                "portfolio_size": p.get("portfolio_size", 0),
                "rationale": rationale_line.strip()
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
        custom_data=["name", "category", "portfolio_size", "rationale"],
        title="🧭 VC Landscape by Strategic Identity",
        labels={"x": labels.get("x_label", "Dimension 1"), "y": labels.get("y_label", "Dimension

# === VC Category Summary Explorer ===
st.divider()
st.subheader("📚 Strategic VC Categories")

profiles = load_vc_profiles()
by_category = {}
for p in profiles:
    cat = (p.get("category") or "").split("\n")[0].replace("Category:", "").strip()
    rationale = next((line for line in p.get("category", "").splitlines() if line.lower().startswith("rationale")), "")
    example = p.get("name", "")
    if cat not in by_category:
        by_category[cat] = {"rationale": rationale, "examples": set()}
    by_category[cat]["examples"].add(example)

for cat, details in by_category.items():
    st.markdown(f"### {cat}")
    if details["rationale"]:
        st.markdown(f"**Rationale:** {details['rationale']}")
    st.markdown(f"**Example Firms:** {', '.join(sorted(details['examples']))}")


