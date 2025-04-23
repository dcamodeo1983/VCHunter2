import numpy as np
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
        embeddings = [p['embedding']
