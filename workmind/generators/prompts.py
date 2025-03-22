DEFAULT_SYSTEM_PROMPT: str = (
    "You are an advanced text analysis assistant. Your task is to:\n"
    "Read the provided {entity}.\n"
    "Determine if the content indicates dissatisfaction or frustration. Most likely it does as it was identified by a sentiment analysis engine.\n"
    "If dissatisfaction is detected:\n"
    "Summarize the core issues clearly.\n"
    "Propose concise short-term and long-term HR interventions.\n"
    "If no dissatisfaction is detected, simply indicate that there is no frustration.\n"
    "Output the response in a structured format:\n"
    "0. Dissatisfaction detected: [Yes/No]\n"
    "1. Dissatisfaction reason: [Brief summary]\n"
    "2. Interventions:\n"
    "    a) Short term: [Actionable recommendations]\n"
    "    b) Long term: [Actionable recommendations]"
)


DEFAULT_USER_PROMPT: str = (
    "Please analyze the following {entity} and:\n"
    "1. Check if there is any dissatisfaction or frustration expressed.\n"
    "2. If so, identify the main reasons for the employee’s frustration.\n"
    "3. Summarize these concerns briefly and clearly.\n"
    "4. Recommend actionable short-term and long-term HR interventions. Be clear and concise.\n"
    'If there are certainly no signs of dissatisfaction, just indicate "Dissatisfaction detected: No" and stop generation.\n'
    "\n"
    "{entity}:\n"
    "{data}"
)


def get_entity_user_prompt(prompt: str, entity: str) -> str:
    """
    Format a user prompt by inserting the entity and a placeholder for data.

    Parameters:
        prompt (str): The prompt template.
        entity (str): The entity name.

    Returns:
        str: The formatted prompt with a placeholder.
    """
    return prompt.format(entity=entity, data="{}")


def get_entity_system_prompt(prompt: str, entity: str) -> str:
    """
    Format a system prompt by inserting the entity.

    Parameters:
        prompt (str): The prompt template.
        entity (str): The entity name.

    Returns:
        str: The formatted system prompt.
    """
    return prompt.format(entity=entity)
