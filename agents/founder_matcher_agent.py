
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import os

VC_PROFILE_PATH = "outputs/vc_profiles.json"

class FounderMatcherAgent:
    def __init__(self, founder_embedding):
        self.founder_embedding = np.array(founder_embedding)

    def load_profiles(self):
        if os.path.exists(VC_PROFILE_PATH):
            with open(VC_PROFILE_PATH, "r") as f:
                return json.load(f)
        return []

    def match(self, top_k=5):
        profiles = self.load_profiles()
        results = []

        for p in profiles:
            if p.get("embedding") and isinstance(p["embedding"], list):
                vc_embedding = np.array(p["embedding"])
                sim = cosine_similarity([self.founder_embedding], [vc_embedding])[0][0]
                results.append({
                    "name": p["name"],
                    "url": p["url"],
                    "score": round(sim, 4),
                    "category": p.get("category", "Uncategorized"),
                    "rationale": p.get("strategy_summary", ""),
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
