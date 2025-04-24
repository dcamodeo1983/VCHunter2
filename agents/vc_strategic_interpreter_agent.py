from openai import OpenAI

class VCStrategicInterpreterAgent:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def interpret_strategy(self, vc_name, site_text, portfolio_entries):
        try:
            formatted_entries = "\n".join(
                [f"- {entry['name']}: {entry['description']}" for entry in portfolio_entries if entry.get('name') and entry.get('description')]
            )
            
            prompt = f"""
You are a senior venture capital analyst tasked with interpreting a VC firm’s strategic thesis.

Your goal is to summarize how the firm positions itself in the investment ecosystem using:
- Website language
- Types of portfolio companies
- Any observable themes or biases

You must answer as if advising a startup founder evaluating this VC.

Firm Name: {vc_name}

Website Summary:
{site_text[:1000]}

Portfolio Companies:
{formatted_entries[:3000]}

Respond in this format:
Category: <Short high-level label>
Rationale: <3–5 sentences on their investing personality and thesis>
Motivational Signals: <Comma-separated themes like: contrarian, deeptech, early-stage SaaS, national resilience>
"""

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=750
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"[Error during strategy interpretation: {e}]"
