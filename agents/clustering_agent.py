import numpy as np
import umap
from sklearn.cluster import KMeans
import json
import os

VC_PROFILE_PATH = "outputs/vc_profiles.json"

class ClusteringAgent:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters

    def load_profiles(self):
        if os.path.exists(VC_PROFILE_PATH):
            with open(VC_PROFILE_PATH, "r") as f:
                return json.load(f)
        return []

    def save_profiles(self, profiles):
        with open(VC_PROFILE_PATH, "w") as f:
            json.dump(profiles, f, indent=2)

    def cluster(self):
        profiles = self.load_profiles()
        embeddings = [p['embedding'] for p in profiles if p['embedding'] and isinstance(p['embedding'], list)]
        names = [p['name'] for p in profiles if p['embedding'] and isinstance(p['embedding'], list)]

        if not embeddings:
            raise ValueError("No valid embeddings found in profiles.")

        embeddings = np.array(embeddings)

        reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=2, random_state=42)
        coordinates = reducer.fit_transform(embeddings)

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        cluster_ids = kmeans.fit_predict(embeddings)

        for i, profile in enumerate(profiles):
            if i < len(coordinates):
                profile["coordinates"] = coordinates[i].tolist()
                profile["cluster_id"] = int(cluster_ids[i])

        self.save_profiles(profiles)
        return profiles
