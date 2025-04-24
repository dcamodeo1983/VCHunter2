from openai import OpenAI

class EmbedderAgent:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def embed_text(self, text):
        try:
            if not isinstance(text, str) or not text.strip():
                return "[Error: Empty or invalid input for embedding.]"

            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=[text.strip()]
            )

            # Check that we received a valid embedding
            embedding = response.data[0].embedding
            if isinstance(embedding, list) and all(isinstance(x, (float, int)) for x in embedding):
                return embedding
            else:
                return "[Error: Invalid embedding response format.]"

        except Exception as e:
            return f"[Error during embedding: {e}]"



