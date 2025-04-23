from openai import OpenAI

class LLMSummarizerAgent:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def summarize(self, text):
        try:
            prompt = f"""
You are a venture capital analyst evaluating startup submissions. Based on the provided founder document, summarize the company’s goals, value proposition, and business model. Use terminology common to the VC industry, and emphasize key themes such as defensibility, differentiation, go-to-market strategy, and scalability.

If applicable, highlight any unique strategies, moats, or competitive advantages that emerge from the material. If there are gaps in the information provided that limit understanding of the opportunity, raise specific questions a VC might ask to complete the picture.

Founder Document:
{text}
"""
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[Error during summarization: {e}]"
