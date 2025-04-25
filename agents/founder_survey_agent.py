class FounderSurveyAgent:
    def __init__(self):
        pass

    def format_survey_summary(self, responses: dict) -> str:
        """
        Convert structured survey responses into a human-readable paragraph for embedding.
        """
        return (
            f"The product is currently at the '{responses['product_stage']}' stage, "
            f"earning '{responses['revenue']}' in revenue. "
            f"It is led by a team of {responses['team_size']} full-time founder(s), "
            f"focused on building a '{responses['product_type']}' solution. "
            f"The company is headquartered in '{responses['location']}'. "
            f"Its primary go-to-market motion is '{responses['gtm']}', "
            f"targeting '{responses['customer']}' customers. "
            f"The perceived moat is '{responses['moat']}'."
        )

