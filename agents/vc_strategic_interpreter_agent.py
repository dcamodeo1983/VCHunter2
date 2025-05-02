from openai import OpenAI
import streamlit as st
import re

class VCStrategicInterpreterAgent:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def interpret_strategy(self, vc_name, site_text, portfolio_entries):
        try:
            cleaned_site_text = re.sub(r"\s+", " ", site_text).strip() if site_text else ""
            if not cleaned_site_text or len(cleaned_site_text.split()) < 100:
                st.warning(f"⚠️ Insufficient website text for {vc_name} ({len(cleaned_site_text.split())} words): '{cleaned_site_text[:100]}...'. Using fallback.")
                return f"Category: Generalist\nRationale: Unable to generate detailed strategy for {vc_name} due to insufficient website data. The firm appears to invest across various sectors, but specific focus areas are unclear.\nMotivational Signals: generalist, startup-friendly"

            formatted_entries = "\n".join(
                [f"- {entry['name']}: {entry['description']}" for entry in portfolio_entries if entry.get('name') and entry.get('description')]
            )
            if not formatted_entries.strip():
                st.warning(f"⚠️ No valid portfolio entries for {vc_name}.")
                formatted_entries = "No portfolio data available."

            st.write(f"📝 Input for {vc_name}: site_text={len(cleaned_site_text)} chars, portfolio_entries={len(portfolio_entries)}")
            prompt = f"""
You are a senior venture capital analyst tasked with interpreting a VC firm’s strategic thesis.

Your goal is to summarize how the firm positions itself in the investment ecosystem using:
- Website language
- Types of portfolio companies
- Any observable themes or biases

You must answer as if advising a startup founder evaluating this VC. If insufficient data is provided, return a default summary indicating limited information.

Firm Name: {vc_name}

Website Summary:
{cleaned_site_text[:5000]}

Portfolio Companies:
{formatted_entries[:10000]}

Respond in this format:
Category: <Short high-level label>
Rationale: <3–5 sentences on their investing personality and thesis>
Motivational Signals: <Comma-separated themes like: contrarian, deeptech, early-stage SaaS, national resilience>
"""
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=750
            )
            summary = response.choices[0].message.content.strip()
            cleaned_summary = re.sub(r"\s+", " ", summary).strip() if summary else ""

            if not cleaned_summary or len(cleaned_summary.split()) < 30:
                st.warning(f"⚠️ Empty or insufficient strategy summary for {vc_name} ({len(cleaned_summary.split())} words): '{cleaned_summary[:100]}...'. Using fallback.")
                summary = f"Category: Generalist\nRationale: Unable to generate detailed strategy for {vc_name} due to limited or ambiguous website and portfolio data. The firm appears to invest across various sectors, but specific focus areas are unclear.\nMotivational Signals: generalist, startup-friendly"
            else:
                summary = cleaned_summary

            st.write(f"📝 Summary for {vc_name}: {summary[:100] if summary else 'None'}...")
            return summary

        except Exception as e:
            st.error(f"❌ Error during strategy interpretation for {vc_name}: {str(e)}")
            return f"Category: Generalist\nRationale: Unable to analyze {vc_name} due to error: {str(e)}. The firm’s strategy could not be determined from available data.\nMotivational Signals: none"
