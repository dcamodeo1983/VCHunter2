from openai import OpenAI

VC_PROFILE_PATH = "outputs/vc_profiles.json"
CLUSTER_LABELS_PATH = "outputs/cluster_labels.json"

class ClusterInterpreterAgent:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def load_profiles(self):
        import json, os
        if os.path.exists(VC_PROFILE_PATH):
            with open(VC_PROFILE_PATH, "r") as f:
                return json.load(f)
        return []

    def save_cluster_labels(self, cluster_labels):
        import json
        with open(CLUSTER_LABELS_PATH, "w") as f:
            json.dump(cluster_labels, f, indent=2)

    def interpret_clusters(self):
        profiles = self.load_profiles()

        # Group strategy summaries by cluster
        cluster_groups = {}
        for p in profiles:
            cid = p.get("cluster_id", -1)
            if cid not in cluster_groups:
                cluster_groups[cid] = []
            cluster_groups[cid].append(p.get("strategy_summary", ""))

        cluster_labels = {}

        for cluster_id, summaries in cluster_groups.items():
            combined_summary = "\n".join(summaries)

            prompt = f"""
You are analyzing a group of venture capital firms based on their investment strategies.

Given the following VC strategic summaries, propose:
- A short, intuitive name for this cluster (e.g., 'Frontier Deep Tech Investors')
- A 1-2 sentence description summarizing what unites these VCs.

Here are the VC strategic summaries:
{combined_summary}

Respond with this format:

Cluster Name: <name>
Cluster Description: <description>
"""

            try:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=400
                )
                content = response.choices[0].message.content.strip()

                # Parse the LLM response
                lines = content.splitlines()
                name_line = next((line for line in lines if line.lower().startswith("cluster name")), None)
                desc_line = next((line for line in lines if line.lower().startswith("cluster description")), None)

                if name_line and desc_line:
                    cluster_labels[str(cluster_id)] = {
                        "name": name_line.replace("Cluster Name:", "").strip(),
                        "description": desc_line.replace("Cluster Description:", "").strip()
                    }
                else:
                    cluster_labels[str(cluster_id)] = {
                        "name": f"Cluster {cluster_id}",
                        "description": "No description generated."
                    }

            except Exception as e:
                print(f"❌ Error interpreting cluster {cluster_id}: {e}")
                cluster_labels[str(cluster_id)] = {
                    "name": f"Cluster {cluster_id}",
                    "description": "Error during interpretation."
                }

        self.save_cluster_labels(cluster_labels)
        return cluster_labels

