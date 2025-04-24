import re
import tiktoken

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def count_tokens(text, model="gpt-4"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def embed_vc_profile(site_text, portfolio_text, embedder):
    try:
        combined = site_text.strip() + "\n\n" + portfolio_text.strip()
        embedding = embedder.embed_text(combined)
        if isinstance(embedding, list) and all(isinstance(x, (float, int)) for x in embedding):
            return embedding
        else:
            print("❌ Invalid embedding response:", embedding)
            return None
    except Exception as e:
        print(f"❌ Exception during embedding: {e}")
        return None



