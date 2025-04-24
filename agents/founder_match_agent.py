
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    import json
    import os
    from openai import OpenAI

    VC_PROFILE_PATH = "outputs/vc_profiles.json"

    class FounderMatcherAgent:
        def __init__(self, api_key):
            self.client = OpenAI(api_key=api_key)

        def load_vc_profiles(self):
            if os.path.exists(VC_PROFILE_PATH):
                with open(VC_PROFILE_PATH, "r") as f:
                    return json.load(f)
            return []

        def find_matches(self, founder_embedding, top_k=5):
            profiles = self.load_vc_profiles()
            matches = []
            vectors = []
            names = []

            for p in profiles:
                if isinstance(p.get("embedding"), list):
                    vectors.append(p["embedding"])
                    names.append(p["name"])
                else:
                    continue

            if not vectors:
                return []

            similarities = cosine_similarity([founder_embedding], vectors)[0]
            ranked = sorted(zip(names, profiles, similarities), key=lambda x: -x[2])[:top_k]

            explanations = []
            for name, profile, score in ranked:
                rationale = self.explain_match(founder_embedding, profile)
                explanations.append({
                    "name": name,
                    "url": profile.get("url", ""),
                    "category": profile.get("category", "Uncategorized"),
                    "score": round(float(score), 4),
                    "rationale": rationale
                })

            return explanations

        def explain_match(self, founder_embedding, profile):
            try:
                summary = profile.get("strategy_summary", "")
                prompt = f"""You are a VC analyst. A founder has submitted a business concept embedding that strongly matches this VC firm's strategic profile:

VC Strategy Summary:
{summary}

Explain why this match is strong and what themes the founder might emphasize when reaching out.
"""
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                    max_tokens=300
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                return f"[Explanation error: {e}]"
