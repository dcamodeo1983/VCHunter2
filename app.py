# VC Hunter Streamlit UI Upgrade (Narrative-Driven)

import streamlit as st
import os
from dotenv import load_dotenv
from agents.founder_doc_reader_agent import FounderDocReaderAgent
from agents.llm_summarizer_agent import LLMSummarizerAgent
from utils.utils import clean_text, count_tokens

st.set_page_config(page_title="VC Hunter", layout="wide")

st.title("🧠 VC Hunter: Founder Intelligence Report")
st.markdown("""
Upload your startup concept to receive curated insights and a clear summary of your business, powered by LLMs.
""")

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

# === Upload & Run ===
uploaded_file = st.file_uploader("📄 Upload Your White Paper", type=["pdf", "txt", "docx"])
if uploaded_file:
    reader = FounderDocReaderAgent()
    summarizer = LLMSummarizerAgent(api_key=openai_api_key)

    st.info("⏳ Extracting text from your file...")
    text = reader.extract_text(uploaded_file)
    if not text.strip():
        st.error("❌ No readable text found in the document.")
    else:
        cleaned_text = clean_text(text)
        token_count = count_tokens(cleaned_text)
        st.success(f"✅ Document processed. ({token_count} tokens)")

        st.info("🧠 Summarizing your concept using GPT...")
        summary = summarizer.summarize(cleaned_text)

        st.header("📄 Startup Summary")
        st.markdown(f"> {summary}")
