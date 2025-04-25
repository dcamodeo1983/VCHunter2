import os
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI

VC_PROFILE_PATH = "outputs/vc_profiles.json"
DIMENSION_LABELS_PATH = "outputs/dimension_labels.json"
EMBEDDING_DIM = 768
TOP_N_TERMS = 6

class DimensionExplainerAgent:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)

    def load_profiles(self):
        if os.path.exists(VC_PROFILE_PATH):
            with open(VC_PROFILE_PATH, "r") as f:
                return json.load(f)
        return []

    def save_labels(self, labels):
        with open(DIMENSION_LABELS_PATH, "w") as f:
            json.dump(labels, f, indent=2)

    def generate_axis_labels(self):
        profiles = self.load_profiles()
        vectors = [p['embedding'] for p in profiles if isinstance(p.get('embedding'), list) and len(p['embedding']) == EMBEDDING_DIM]
        texts = [p['strategy_summary'] for p in profiles if p.get('strategy_summary')]

        if len(vectors) < 2 or len(texts) < 2:
            return {"error": "Not enough data to perform PCA and TF-IDF analysis."}

        X = np.array(vectors)
        pca = PCA(n_components=2)
        pca.fit(X)
        variance = pca.explained_variance_ratio_

        tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
        tfidf_matrix = tfidf.fit_transform(texts)
        terms = tfidf.get_feature_names_out()

        components = pca.components_
        term_scores_pc1 = np.dot(tfidf_matrix.toarray(), components[0])
        term_scores_pc2 = np.dot(tfidf_matrix.toarray(), components[1])
        top_pc1 = [terms[i] for i in np.argsort(term_scores_pc1)[-TOP_N_TERMS:][::-1]]
        top_pc2 = [terms[i] for i in np.argsort(term_scores_pc2)[-TOP_N_TERMS:][::-1]]

        prompt = f"""You are an expert analyst interpreting a PCA-reduced embedding space of VC firm strategies.
The X-axis is defined by: {', '.join(top_pc1)}.
The Y-axis is defined by: {', '.join(top_pc2)}.
Please assign an intuitive axis label and a one-sentence description for each dimension.

Respond in this format:
Dimension 1: <short label> — <left vs right description>
Dimension 2: <short label> — <bottom vs top description>"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            result = response.choices[0].message.content.strip()
            lines = result.splitlines()
            x_label, x_description = lines[0].split("—")
            y_label, y_description = lines[1].split("—")

            labels = {
                "x_label": x_label.replace("Dimension 1:", "").strip(),
                "x_description": x_description.strip(),
                "x_variance": float(variance[0]),
                "y_label": y_label.replace("Dimension 2:", "").strip(),
                "y_description": y_description.strip(),
                "y_variance": float(variance[1])
            }

            self.save_labels(labels)
            return labels

        except Exception as e:
            return {
                "x_label": "Dimension 1",
                "y_label": "Dimension 2",
                "x_description": "Left = ?, Right = ?",
                "y_description": "Bottom = ?, Top = ?",
                "x_variance": 0.0,
                "y_variance": 0.0,
                "error": str(e)
            }
