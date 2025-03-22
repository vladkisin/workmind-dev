import pytest
from workmind.generators.interventions import (
    InterventionGenerator,
    RAGInterventionGenerator,
)


def test_construct_prompt_intervention_generator():
    generator = InterventionGenerator(model_name="dummy-model")
    emails = ["Email 1 text.", "Email 2 text."]
    prompt = generator.construct_prompt(emails)
    assert isinstance(prompt, list)
    assert len(prompt) == 2
    for message in prompt:
        assert "role" in message
        assert "content" in message


def test_analyze_emails_intervention_generator(monkeypatch):
    def dummy_analyze_emails(batch_of_emails):
        return ["dummy intervention response" for _ in batch_of_emails]

    generator = InterventionGenerator(model_name="dummy-model")
    monkeypatch.setattr(generator, "analyze_emails", dummy_analyze_emails)
    batch = [["Email text 1"], ["Email text 2"]]
    responses = generator.predict(batch)
    assert len(responses) == len(batch)
    for response in responses:
        assert response == "dummy intervention response"


def test_construct_prompt_rag():
    emails = ["Email 1 text", "Email 2 text"]
    prompt = RAGInterventionGenerator.construct_prompt(emails)
    assert "1. Email 1 text" in prompt
    assert "2. Email 2 text" in prompt
