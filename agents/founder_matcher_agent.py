
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .base_agent import BaseAgent

VC_PROFILE_PATH = "outputs/vc_profiles.json"

class FounderMatcherAgent(BaseAgent):
    def __init__(self, founder_embedding):
        super().__init__()
        self.founder_embedding = founder_embedding

    def load_profiles(self):
        if not self.founder_embedding:
            return []
        if not isinstance(self.founder_embedding, list):
            return []

        try:
            with open(VC_PROFILE_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            print("⚠️ Error: vc_profiles.json is missing or invalid.")
            return []

    def match(self, top_k=5):
        profiles = self.load_profiles()
        results = []

        for profile in profiles:
            vc_embedding = profile.get("embedding")
            if not isinstance(vc_embedding, list):
                print(f"⚠️ Skipping VC with invalid embedding: {profile.get('url')}")
                continue

            if (
                isinstance(vc_embedding, list)
                and isinstance(self.founder_embedding, list)
                and len(vc_embedding) == len(self.founder_embedding)
                and all(isinstance(x, (int, float)) for x in vc_embedding)
                and all(isinstance(x, (int, float)) for x in self.founder_embedding)
            ):
                try:
                    sim = cosine_similarity([self.founder_embedding], [vc_embedding])[0][0]
                    results.append({
                        "name": profile.get("name", ""),
                        "url": profile.get("url", ""),
                        "category": profile.get("category", "Unknown"),
                        "score": round(sim, 4),
                        "rationale": profile.get("strategy_summary", "")
                    })
                except Exception as e:
                    print(f"⚠️ Failed similarity for {profile.get('url')}: {str(e)}")

        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results[:top_k]
