import re
import tiktoken

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def count_tokens(text, model="gpt-4"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))
