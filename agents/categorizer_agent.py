from openai import OpenAI
import json
import os

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
        cluster_ids = sorted(set(p["cluster_id"] for p in profiles if p.get("cluster_id") is not None))

        for cluster_id in cluster_ids:
            cluster_profiles = [p for p in profiles if p["cluster_id"] == cluster_id]

            summarized_vcs = "\n".join([
                f"- {p.get('name', 'Unnamed VC')}: {p.get('strategy_summary', 'No summary')[:250]}"
                for p in cluster_profiles
            ])

            prompt = f"""
You are a senior venture capital partner reviewing a group of VC firms that have been clustered together based on their investment behavior and strategy.

Your task is to:
1. Interpret what this group of VC firms has in common.
2. Assign a short, founder-friendly name to this cluster.
3. Describe the shared investment style, thesis, or cultural mindset of the group.
4. Suggest what types of startups or founders are a natural fit for this cluster.

Your response will be shown to startup founders exploring the VC landscape. Help them understand which types of investors they’re looking at.

Input:
Here is a list of VC firms in this cluster, along with a short summary of each firm’s investment strategy:

{summarized_vcs}

Format your answer like this:
Category: <short, intuitive label>
Rationale: <1–2 sentences explaining the common theme>
Suggested Fit: <1 sentence on what kind of startups or founders this group is right for>
"""

            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                    max_tokens=500
                )
                result = response.choices[0].message.content.strip()

                for profile in cluster_profiles:
                    profile["category"] = result

            except Exception as e:
                for profile in cluster_profiles:
                    profile["category"] = f"[Error during categorization: {e}]"

        self.save_profiles(profiles)
        return profiles
