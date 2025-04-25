from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import json
import os

VC_PROFILE_PATH = "outputs/vc_profiles.json"

class ClusteringAgent:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.pca = None  # Save PCA model here for reuse

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

        if not embeddings:
            print("❌ No embeddings found for clustering.")
            return []

        X = np.array(embeddings)
        self.pca = PCA(n_components=2, random_state=42)
        coords = self.pca.fit_transform(X)

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)

        # Save coordinates and cluster ID
        for profile, label, (x, y) in zip(profiles, labels, coords):
            profile["cluster_id"] = int(label)
            profile["pca_x"] = float(x)
            profile["pca_y"] = float(y)

        self.save_profiles(profiles)
        return profiles

    def transform(self, embedding):
        if self.pca is None:
            raise ValueError("PCA not trained yet. Run cluster() first.")
        embedding = np.array(embedding).reshape(1, -1)  # Ensure 2D
        return self.pca.transform(embedding)[0]
