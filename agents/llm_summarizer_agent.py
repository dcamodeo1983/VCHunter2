from openai import OpenAI
from openai.types.chat import ChatCompletionMessage

class LLMSummarizerAgent:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def summarize(self, text):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    ChatCompletionMessage(role="user", content=f"Summarize this startup concept:\n{text}")
                ],
                temperature=0.5,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[Error during summarization: {e}]"
