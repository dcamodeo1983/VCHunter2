import streamlit as st
import json
import os
from openai import OpenAI
import numpy as np

class DimensionExplainerAgent:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.dimension_labels_path = "outputs/dimension_labels.json"

    def generate_axis_labels(self, profiles, pca):
        try:
            # Extract features for analysis
            embeddings = [p["embedding"] for p in profiles if isinstance(p.get("embedding"), list)]
            if not embeddings:
                st.error("❌ No valid embeddings for dimension labeling.")
                return

            # Analyze PCA components
            components = pca.components_
            feature_contributions = np.abs(components).argsort()[:, ::-1]

            prompt = f"""
You are a data scientist analyzing a PCA-based visualization of venture capital firms.

Your task is to generate intuitive labels for the X and Y axes based on the following:
- Each VC profile has an embedding capturing their investment strategy.
- PCA has reduced these embeddings to 2 dimensions.
- Top contributing features for PC1: {feature_contributions[0][:5].tolist()}
- Top contributing features for PC2: {feature_contributions[1][:5].tolist()}
- Sample VC strategies: {[p.get('strategy_summary', '')[:100] for p in profiles[:3]]}

Suggest labels and descriptions for the X and Y axes that would help a startup founder understand the VC landscape. Focus on investment stage, sector, or strategic focus.

Return in this JSON format:
{{
  "x_label": "<label>",
  "x_description": "<description>",
  "y_label": "<label>",
  "y_description": "<description>"
}}
"""
            st.write("📝 Generating dimension labels...")
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=200
            )
            labels = json.loads(response.choices[0].message.content.strip())
            
            # Save labels
            os.makedirs(os.path.dirname(self.dimension_labels_path), exist_ok=True)
            with open(self.dimension_labels_path, "w") as f:
                json.dump(labels, f, indent=2)
            st.write("✅ Dimension labels saved.")

        except Exception as e:
            st.error(f"❌ Failed to generate dimension labels: {str(e)}")
            raise

    def load_dimension_labels(self):
        try:
            if os.path.exists(self.dimension_labels_path):
                with open(self.dimension_labels_path, "r") as f:
                    return json.load(f)
            return {}
        except Exception as e:
            st.error(f"❌ Failed to load dimension labels: {str(e)}")
            return {}
