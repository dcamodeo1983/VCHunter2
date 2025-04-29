import re
import tiktoken
import os
import json
from openai import OpenAI

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

    enriched_input = f"""
[Strategic Summary]
{interpreter_summary.strip()}

[Website Text Preview]
{site_text.strip()[:800]}

[Portfolio Highlights]
{portfolio_text.strip()[:1200]}
""".strip()

    return embedder.embed_text(enriched_input)

def interpret_pca_dimensions(components, explained_var):
    """
    Uses OpenAI GPT to interpret PCA components and generate intuitive axis labels + descriptions.
    Args:
        components (list): PCA components (list of lists)
        explained_var (list): Explained variance ratios for the first two principal components
    Returns:
        dict: Dictionary containing x/y labels, descriptions, and variance percentages
    """
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    prompt = f"""
You are an expert VC strategist analyzing a 2D PCA projection of venture capital firm investment strategies.

- Dimension 1 (explained variance: {explained_var[0]*100:.2f}%) and Dimension 2 ({explained_var[1]*100:.2f}%) are the primary axes.
- Each dimension is influenced by multiple latent factors extracted from VC investment theses.

Propose:
- A short, intuitive label for Dimension 1 (X-axis)
- A short, intuitive label for Dimension 2 (Y-axis)
- A 1-sentence left-to-right interpretation for Dimension 1
- A 1-sentence bottom-to-top interpretation for Dimension 2

Respond in this JSON format:
{
"x_label": "<short x label>",
"x_description": "<x interpretation>",
"y_label": "<short y label>",
"y_description": "<y interpretation>"
}
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=600
    )

    result = response.choices[0].message.content.strip()
    parsed = json.loads(result)

    parsed["x_variance"] = explained_var[0]
    parsed["y_variance"] = explained_var[1]

    return parsed
