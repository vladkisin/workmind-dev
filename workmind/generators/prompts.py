DEFAULT_SYSTEM_PROMPT = """
You are an advanced text analysis assistant. Your task is to:
- Read the provided {entity}.
- Determine if the content indicates dissatisfaction or frustration. Most likely it does as it was identified by a sentiment analysis engine.
- If dissatisfaction is detected:
- Summarize the core issues clearly.
- Propose concise short-term and long-term HR interventions.
- If no dissatisfaction is detected, simply indicate that there is no frustration.
- Output the response in a structured format:
    0. Dissatisfaction detected: [Yes/No]
    1. Dissatisfaction reason: [Brief summary]
    2. Interventions:
        a) Short term: [Actionable recommendations]
        b) Long term: [Actionable recommendations]
"""

DEFAULT_USER_PROMPT = """
Please analyze the following {entity} and:
1. Check if there is any dissatisfaction or frustration expressed.
2. If so, identify the main reasons for the employee’s frustration.
3. Summarize these concerns briefly and clearly.
4. Recommend actionable short-term and long-term HR interventions. Be clear and concise.
If there are certainly no signs of dissatisfaction are found, just indicate "Dissatisfaction detected: No" and stop generation.

{entity}:
{data}
"""


def get_entity_user_prompt(prompt, entity):
    return prompt.format(entity=entity, data="{}")


def get_entity_system_prompt(prompt, entity):
    return prompt.format(entity=entity)
