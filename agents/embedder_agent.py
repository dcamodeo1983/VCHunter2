from openai import OpenAI

class EmbedderAgent:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def embed_text(self, text):
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=[text]
            )
            return response.data[0].embedding
        except Exception as e:
            return f"[Error during embedding: {e}]"
