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
You are a senior VC analyst tasked with analyzing a venture capital firm's strategic thesis. Based on the firm's website and portfolio company list, describe the following:

1. What category best describes the firm's investment behavior?
2. What are the underlying motivations or patterns you observe?
3. Are they contrarian, domain-focused, mission-driven, or thesis-led?
4. Provide a list of 2-5 strategic signals or investment themes that define the firm's worldview.

Respond in this format:
Category: <label>
Rationale: <3-5 sentences>
Motivational Signals: <comma-separated list>

Firm Name: {vc_name}
Website Summary: {site_text[:1000]}
Portfolio Companies:
{formatted_entries[:3000]}
"""
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo"
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=750
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[Error during strategy interpretation: {e}]"

