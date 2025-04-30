
import json
import os
from openai import OpenAI
from sklearn.decomposition import PCA
import numpy as np

VC_PROFILE_PATH = "outputs/vc_profiles.json"
DIMENSION_LABELS_PATH = "outputs/dimension_labels.json"


class DimensionExplainerAgent:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def generate_axis_labels(self):
        if not os.path.exists(VC_PROFILE_PATH):
            return

        with open(VC_PROFILE_PATH, "r") as f:
            profiles = json.load(f)

        embeddings = [p["embedding"] for p in profiles if isinstance(p.get("embedding"), list)]
        if len(embeddings) < 2:
            return

        X = np.array(embeddings)
        pca = PCA(n_components=2)
        pca.fit(X)
        explained_variance = pca.explained_variance_ratio_

        prompt = (
            "You are analyzing a PCA scatterplot of venture capital firms based on their investment thesis and behavior.\n"
            "Two principal components (PC1 and PC2) capture the semantic variation between firms.\n"
            "Interpret the meaning of these components in simple, intuitive terms for startup founders.\n"
            "Return a JSON with: x_label, y_label, x_description, y_description.\n"
            "Be concise and founder-friendly."
        )

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=300,
            )
            content = response.choices[0].message.content.strip()
            labels = json.loads(content)
            labels["x_variance"] = explained_variance[0]
            labels["y_variance"] = explained_variance[1]

            with open(DIMENSION_LABELS_PATH, "w") as out:
                json.dump(labels, out, indent=2)
        except Exception as e:
            print(f"Error generating dimension labels: {e}")
