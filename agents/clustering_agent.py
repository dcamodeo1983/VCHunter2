import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import json
import os

VC_PROFILE_PATH = "outputs/vc_profiles.json"

class ClusteringAgent:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.pca = None

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
        embeddings = [p["embedding"] for p in profiles if isinstance(p.get("embedding"), list)]

        if len(embeddings) < 2:
            raise ValueError("Not enough valid embeddings to perform clustering.")

        X = np.array(embeddings)

        # PCA for visualization
        self.pca = PCA(n_components=2)
        coords = self.pca.fit_transform(X)

        # KMeans clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        cluster_ids = kmeans.fit_predict(X)

        for i, profile in enumerate(profiles):
            profile["coordinates"] = coords[i].tolist()
            profile["cluster_id"] = int(cluster_ids[i])

        self.save_profiles(profiles)
        return profiles

    def transform(self, embedding):
        if self.pca:
            return self.pca.transform([embedding])[0].tolist()
        return [None, None]
