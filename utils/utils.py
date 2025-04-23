import re
import tiktoken

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def count_tokens(text, model="gpt-4"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def embed_vc_profile(website_text, portfolio_text, embedder):
    """
    Combines website and portfolio content and returns a single 1536-d vector.
    """
    combined = f"VC THESIS & BACKGROUND:\n{website_text.strip()}\n\nPORTFOLIO BEHAVIOR:\n{portfolio_text.strip()}"
    return embedder.embed_text(combined)

