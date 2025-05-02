import json
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from openai import OpenAI

VC_PROFILE_PATH = "outputs/vc_profiles.json"
CLUSTER_LABELS_PATH = "outputs/cluster_labels.json"

class ClusterInterpreterAgent:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def load_profiles(self):
        if os.path.exists(VC_PROFILE_PATH):
            with open(VC_PROFILE_PATH, "r") as f:
                return json.load(f)
        return []

    def save_profiles(self, profiles):
        with open(VC_PROFILE_PATH, "w") as f:
            json.dump(profiles, f, indent=2)

    def assign_kmeans_clusters(self, n_clusters=4):
        profiles = self.load_profiles()
        vectors = [p["embedding"] for p in profiles if isinstance(p.get("embedding"), list)]
        if not vectors:
            return []

        X = normalize(np.array(vectors))
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        cluster_ids = model.fit_predict(X)

        cluster_label_map = {}
        for i, profile in enumerate(profiles):
            profile["cluster_id"] = int(cluster_ids[i])
            cluster_label_map[str(cluster_ids[i])] = {"name": f"Cluster {cluster_ids[i]}"}

        with open(VC_PROFILE_PATH, "w") as f:
            json.dump(profiles, f, indent=2)
        with open(CLUSTER_LABELS_PATH, "w") as f:
            json.dump(cluster_label_map, f, indent=2)

        return profiles

    def categorize_clusters(self):
        profiles = self.load_profiles()
        cluster_ids = sorted(set(p["cluster_id"] for p in profiles if p.get("cluster_id") is not None))

        for cluster_id in cluster_ids:
            cluster_profiles = [p for p in profiles if p["cluster_id"] == cluster_id]
            summarized_vcs = "\n".join(
                [f"- {p.get('name', 'Unnamed VC')}: {p.get('strategy_summary', '')[:250]}" for p in cluster_profiles]
            )

            prompt = f"""
You are a senior venture capital partner reviewing a group of VC firms that have been clustered together based on their investment behavior and strategy.

Your task is to:
1. Assign a short, founder-friendly name to this cluster.
2. Describe the shared investment style, thesis, or cultural mindset of the group.
3. Suggest what types of startups or founders are a natural fit for this cluster.

Input:
{summarized_vcs}

Format:
Category: <short name>
Rationale: <description>
Suggested Fit: <fit>
"""
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                    max_tokens=500,
                )
                result = response.choices[0].message.content.strip()

                category = ""
                rationale = ""
                fit = ""
                for line in result.splitlines():
                    if line.startswith("Category:"):
                        category = line.split("Category:")[1].strip()
                    elif line.startswith("Rationale:"):
                        rationale = line.split("Rationale:")[1].strip()
                    elif line.startswith("Suggested Fit:"):
                        fit = line.split("Suggested Fit:")[1].strip()

                for profile in cluster_profiles:
                    profile["category"] = category
                    profile["category_rationale"] = rationale
                    profile["category_fit"] = fit
            except Exception as e:
                for profile in cluster_profiles:
                    profile["category"] = f"[Error: {e}]"

        self.save_profiles(profiles)
        return profiles
