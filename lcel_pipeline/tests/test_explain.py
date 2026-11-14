"""Tests for lcel_pipeline/explain.py."""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from lcel_pipeline.config import PipelineSettings
from lcel_pipeline.explain import (
    build_explain_chain,
    build_explain_prompt,
    format_prompt_display,
    run_explain,
)


@pytest.fixture
def dummy_settings():
    return PipelineSettings(
        groq_api_key="gsk_dummy_test_key_12345",
        model="openai/gpt-oss-20b",
        chunk_size=800,
        chunk_overlap=120,
    )


def test_build_explain_prompt_placeholders():
    """Verify ChatPromptTemplate has topic, audience, and format variables."""
    prompt = build_explain_prompt()
    assert isinstance(prompt, ChatPromptTemplate)
    assert set(prompt.input_variables) == {"topic", "audience", "format"}

    formatted = prompt.format_messages(
        topic="Asyncio",
        audience="undergrads",
        format="two sentences",
    )
    assert len(formatted) == 2
    assert "undergrads" in formatted[0].content
    assert "two sentences" in formatted[0].content
    assert "Asyncio" in formatted[1].content


def test_lcel_chain_composition(dummy_settings):
    """Verify chain composition: prompt | ChatGroq | StrOutputParser."""
    chain = build_explain_chain(dummy_settings, temperature=0.0)

    # In LangChain, 'prompt | model | parser' is a RunnableSequence
    steps = chain.steps
    assert len(steps) == 3
    assert isinstance(steps[0], ChatPromptTemplate)
    assert isinstance(steps[1], ChatGroq)
    assert isinstance(steps[2], StrOutputParser)
    # ChatGroq internally clamps temperature 0.0 to 1e-8 for API compatibility
    assert steps[1].temperature == pytest.approx(0.0, abs=1e-6)


def test_show_prompt_prints_without_api_call(dummy_settings):
    """--show-prompt must format prompt without invoking ChatGroq."""
    with patch("lcel_pipeline.explain.ChatGroq") as mock_groq:
        display = run_explain(
            topic="Semaphore",
            audience="backend engineers",
            format_spec="short bullet points",
            settings=dummy_settings,
            show_prompt=True,
        )

        # ChatGroq must not have been instantiated or called
        mock_groq.assert_not_called()

        assert "=== FORMATTED PROMPT (No API Call) ===" in display
        assert "Target Audience: backend engineers" in display
        assert "Format Spec:     short bullet points" in display
        assert "Topic:           Semaphore" in display
        assert "[System Message]:" in display
        assert "[Human Message]:" in display


def test_empty_topic_rejected(dummy_settings):
    """Empty or whitespace-only topic raises ValueError before chain execution."""
    for empty_val in ["", "   ", "\t\n", None]:
        with pytest.raises(ValueError, match="Topic cannot be empty or whitespace-only"):
            run_explain(
                topic=empty_val,
                audience="engineers",
                format_spec="bullets",
                settings=dummy_settings,
                show_prompt=False,
            )


def test_same_chain_serves_three_audiences(dummy_settings):
    """The same chain serves 3 distinct audiences, changing the system prompt payload."""
    from langchain_core.runnables import RunnableLambda

    audiences = ["5-year-old", "systems engineer", "executive"]
    captured_audiences = []

    def fake_model_step(prompt_value):
        # Inspect system message formatted by prompt
        sys_msg = prompt_value.messages[0].content
        for aud in audiences:
            if f"target audience: {aud}" in sys_msg:
                captured_audiences.append(aud)
                return AIMessage(content=f"Explanation tailored for {aud}")
        return AIMessage(content="Default response")

    prompt = build_explain_prompt()
    parser = StrOutputParser()
    chain = prompt | RunnableLambda(fake_model_step) | parser

    results = []
    for aud in audiences:
        out = chain.invoke({
            "topic": "Event Loop",
            "audience": aud,
            "format": "one line",
        })
        results.append(out)

    assert captured_audiences == ["5-year-old", "systems engineer", "executive"]
    assert results[0] == "Explanation tailored for 5-year-old"
    assert results[1] == "Explanation tailored for systems engineer"
    assert results[2] == "Explanation tailored for executive"
