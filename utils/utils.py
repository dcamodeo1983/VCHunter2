import re
import tiktoken

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def count_tokens(text, model="gpt-4"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def embed_vc_profile(site_text, portfolio_text, interpreter_summary, embedder):
    """
    Generate an enriched embedding based on LLM interpretation (category + rationale),
    not just raw scraped content. This should improve clustering quality and
    interpretability of PCA dimensions.
    """
    if not interpreter_summary:
        return "[Error: Missing interpretation summary.]"

    # Basic metadata preview
    enriched_input = f"""
[Strategic Summary]
{interpreter_summary.strip()}

[Website Text Preview]
{site_text.strip()[:800]}

[Portfolio Highlights]
{portfolio_text.strip()[:1200]}
""".strip()

    return embedder.embed_text(enriched_input)





