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
        combined = site_text + "\n\n" + portfolio_text
        return embedder.embed_text(combined)
    except Exception as e:
        return f"Embedding failed: {e}"


