# agents/strategic_tagger_agent.py

import os
import json
from openai import OpenAI

class StrategicTaggerAgent:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def generate_tags(self, strategy_summary):
        if not strategy_summary:
            return ["#Generalist"]

        prompt = f"""
You are a venture capital research analyst.

Given the following investment strategy description, return 1–3 short hashtags that capture the focus areas, investment style, or specialization.

Each tag should:
- Be short (1–3 words)
- Use #hashtag style (e.g., #DeepTech, #ClimateInnovation, #ConsumerSaaS)
- Avoid full sentences.

Here is the strategy summary:
---
{strategy_summary}
---
Only output the hashtags, separated by commas. No extra text.
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=100
            )
            tags_text = response.choices[0].message.content.strip()
            tags = [tag.strip() for tag in tags_text.split(",") if tag.strip()]
            if not tags:
                return ["#Generalist"]
            return tags
        except Exception as e:
            print(f"❌ Error generating tags: {e}")
            return ["#Generalist"]
