import json
import os
from openai import OpenAI
from collections import defaultdict

VC_PROFILE_PATH = "outputs/vc_profiles.json"

class CategorizerAgent:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def load_profiles(self):
        if os.path.exists(VC_PROFILE_PATH):
            with open(VC_PROFILE_PATH, "r") as f:
                return json.load(f)
        return []

    def save_profiles(self, profiles):
        with open(VC_PROFILE_PATH, "w") as f:
            json.dump(profiles, f, indent=2)

    def categorize_clusters(self):
        profiles = self.load_profiles()
        clusters = defaultdict(list)

        for p in profiles:
            if p.get("cluster_id") is not None:
                clusters[p["cluster_id"]].append(p)

        for cluster_id, members in clusters.items():
            firm_summaries = "\n".join([
                f"- {m['name']}: {m['strategy_summary'].replace('\\n', ' ')}"
                for m in members if m.get("strategy_summary")
            ])

            prompt = f"""
You are a seasoned venture capital partner tasked with interpreting clusters of similar VC firms.

You are given a list of firms in this cluster along with their investment strategies.

Your task is to:
- Assign a meaningful, founder-friendly name to this cluster
- Summarize what connects these firms (e.g., frontier tech, SaaS scalability, deep sector insight)
- Suggest which types of startups are a good fit
- Keep it short and crisp — this will appear in a UI for startup founders

Format:
Category: [Short label]
Rationale: [1–2 sentence explanation]
Suggested Fit: [What kind of founders should consider this group]

"""
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo"
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4,
                    max_tokens=300
                )
                result = response.choices[0].message.content.strip()
                for m in members:
                    m["category"] = result
            except Exception as e:
                for m in members:
                    m["category"] = f"[Error: {e}]"

        self.save_profiles(profiles)
        return profiles
