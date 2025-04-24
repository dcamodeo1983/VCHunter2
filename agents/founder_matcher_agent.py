
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        import json
        import os
        from openai import OpenAI

        VC_PROFILE_PATH = "outputs/vc_profiles.json"

        class FounderMatcherAgent:
            def __init__(self, api_key=None):
                self.client = OpenAI(api_key=api_key)

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
                            "strategy_summary": profile.get("strategy_summary", ""),
                            "embedding": vc_embedding
                        })

                matches = sorted(matches, key=lambda x: x["score"], reverse=True)[:top_k]

                # Enrich with reasoning and messaging themes
                enriched = []
                for match in matches:
                    prompt = f"""
You are a VC analyst. A founder submitted a profile that scored {match['score']} in semantic similarity with this VC:

Category: {match['category']}
Strategy Summary:
{match['strategy_summary']}

Why is this VC a good fit for the founder? Suggest how the founder should engage with this firm.
Provide this format:

Fit Explanation: <why this VC is a match>
Suggested Messaging Themes: <bullet points>
"""
                    try:
                        response = self.client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.5,
                            max_tokens=500
                        )
                        explanation = response.choices[0].message.content.strip()
                    except Exception as e:
                        explanation = f"[Explanation unavailable: {e}]"

                    match["explanation"] = explanation
                    enriched.append(match)

                return enriched
