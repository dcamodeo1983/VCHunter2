import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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
        embeddings = [p['embedding'] for p in profiles if p.get('embedding') and isinstance(p['embedding'], list)]

        if not embeddings:
            raise ValueError("No valid embeddings found.")

        embeddings = np.array(embeddings)

        # PCA for 2D coordinates (for visualization)
        pca = PCA(n_components=2)
        coordinates = pca.fit_transform(embeddings)

        # KMeans clustering on full 1536-D embeddings
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        cluster_ids = kmeans.fit_predict(embeddings)

        for i, profile in enumerate(profiles):
            profile["coordinates"] = coordinates[i].tolist()
            profile["cluster_id"] = int(cluster_ids[i])

        self.save_profiles(profiles)
        return profiles
