from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import os

VC_PROFILE_PATH = "outputs/vc_profiles.json"

class FounderMatcherAgent:
    def __init__(self, api_key=None):
        self.api_key = api_key

    def load_profiles(self):
        if os.path.exists(VC_PROFILE_PATH):
            with open(VC_PROFILE_PATH, "r") as f:
                return json.load(f)
        return []

    def find_matches(self, founder_embedding, top_k=5):
        profiles = self.load_profiles()
        matches = []

        for profile in profiles:
            vc_embedding = profile.get("embedding")
            if isinstance(vc_embedding, list):
                sim = cosine_similarity([founder_embedding], [vc_embedding])[0][0]
                matches.append({
                    "name": profile["name"],
                    "url": profile["url"],
                    "score": round(sim, 4),
                    "category": profile.get("category", "Uncategorized"),
                    "rationale": profile.get("strategy_summary", "")
                })

        matches = sorted(matches, key=lambda x: x["score"], reverse=True)
        return matches[:top_k]
