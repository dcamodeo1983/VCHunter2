from openai import OpenAI

class LLMSummarizerAgent:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def summarize(self, text):
        try:
            prompt = f"""
You are a startup analyst at a venture capital firm.

A founder has submitted the following white paper or concept document for investment consideration. Your task is to:
- Summarize what the company is building and why
- Highlight the core value proposition
- Infer business strategy including go to market, and possible moats.  
- Describe the intended market, customer, or user
- Note any standout features (tech, business model, traction)
- Identify gaps or ambiguities in the information
- Note any key financial information or proof of traction

Be concise but informative. Write in a tone that helps other VC colleagues understand the opportunity clearly and quickly.

Format:
Summary:
[Concise narrative summary]

Key Questions:
- [Optional bullet if anything is unclear]

Here is the founder's submitted text:

{text}
"""
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=1000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[Error during summarization: {e}]"

def summarize(self, text):
    try:
        prompt = f"""
You are a startup analyst at a venture capital firm.

A founder has submitted the following white paper or concept document for investment consideration. Your task is to:
- Summarize what the company is building and why
- Highlight the core value proposition
- Infer business strategy including go to market, and possible moats.  
- Describe the intended market, customer, or user
- Note any standout features (tech, business model, traction)
- Identify gaps or ambiguities in the information
- Note any key financial information or proof of traction

Be concise but informative. Write in a tone that helps other VC colleagues understand the opportunity clearly and quickly.

Format:
Summary:
[Concise narrative summary]

Key Questions:
- [Optional bullet if anything is unclear]

Here is the founder's submitted text:

{text}
"""
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error during summarization: {e}]"

