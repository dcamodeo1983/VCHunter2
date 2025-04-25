import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

VC_PROFILE_PATH = "outputs/vc_profiles.json"

class ClusteringAgent:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.pca = None

    def load_profiles(self):
        with open(VC_PROFILE_PATH, "r") as f:
            return json.load(f)

    def save_profiles(self, profiles):
        with open(VC_PROFILE_PATH, "w") as f:
            json.dump(profiles, f, indent=2)

    def cluster(self):
        profiles = self.load_profiles()
        embeddings = [p["embedding"] for p in profiles if isinstance(p.get("embedding"), list)]
        if len(embeddings) < 2:
            return []

        X = np.array(embeddings)

        # Run PCA
        self.pca = PCA(n_components=2)
        X_pca = self.pca.fit_transform(X)

        # Run KMeans clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_pca)

        for i, profile in enumerate(profiles):
            profile["cluster_id"] = int(clusters[i])
            profile["coordinates"] = X_pca[i].tolist()

        self.save_profiles(profiles)
        return profiles

    def transform(self, embedding):
        if self.pca:
            return self.pca.transform([embedding])[0].tolist()
        return [None, None]
