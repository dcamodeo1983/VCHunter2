# agents/strategic_tagger_agent.py

import os
import json
from openai import OpenAI

class StrategicTaggerAgent:
    def __init__(self, api_key):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

    def generate_tags_and_signals(self, strategy_summary):
        prompt = f"""
Analyze the following VC investment strategy and return two outputs:
1. A short list of strategic focus tags (3-7 concise phrases).
2. A short list of motivational signals (3-5 concise phrases that describe what excites this VC about startups).

Strategy Summary:
\"\"\"
{strategy_summary}
\"\"\"

Return the result in this strict JSON format:

{{
  "tags": ["..."],
  "motivational_signals": ["..."]
}}
"""

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse and return JSON safely
        text_response = response.choices[0].message.content.strip()
        try:
            parsed_response = json.loads(text_response)
        except Exception as e:
            print(f"Failed to parse LLM output: {e}")
            parsed_response = {"tags": [], "motivational_signals": []}

        return parsed_response

