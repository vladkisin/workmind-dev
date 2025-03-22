from workmind.generators.prompts import get_entity_user_prompt, get_entity_system_prompt


def test_get_entity_user_prompt():
    prompt_template = "Analyze the {entity}: {data}"
    user_prompt = get_entity_user_prompt(prompt_template, "email")
    assert "email" in user_prompt
    assert "{}" in user_prompt


def test_get_entity_system_prompt():
    prompt_template = "System: analyze {entity} now."
    system_prompt = get_entity_system_prompt(prompt_template, "chat")
    assert "chat" in system_prompt
    assert "{}" not in system_prompt
