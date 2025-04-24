class FounderSurveyAgent:
    def __init__(self):
        pass

    def format_survey_summary(self, responses: dict) -> str:
        """
        Take in a dictionary of Streamlit form responses and return a readable summary.
        """
        summary = f"""
[Founder Survey Summary]

- Product Stage: {responses.get('product_stage')}
- Revenue: {responses.get('revenue')}
- Team Size: {responses.get('team_size')} full-time founder(s)
- Product Type: {responses.get('product_type')}
- Headquarters: {responses.get('location')}
- Go-to-Market: {responses.get('gtm')}
- Customer Type: {responses.get('customer')}
- Moat: {responses.get('moat')}
""".strip()

        return summary
