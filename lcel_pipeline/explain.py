"""Explain mode using LangChain Expression Language (LCEL).

Composes ChatPromptTemplate | ChatGroq | StrOutputParser with placeholders
for topic, audience, and format.
"""

from __future__ import annotations

import sys
from typing import Optional
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq

from lcel_pipeline.config import PipelineSettings

SYSTEM_PROMPT_TEMPLATE = (
    "You are an expert technical explainer. Explain the requested technical topic "
    "clearly and accurately, tailoring the depth, terminology, and tone specifically "
    "for the target audience: {audience}.\n"
    "Follow this format specification strictly: {format}."
)

HUMAN_PROMPT_TEMPLATE = "Topic to explain: {topic}"


def build_explain_prompt() -> ChatPromptTemplate:
    """Create ChatPromptTemplate with topic, audience, and format placeholders."""
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_TEMPLATE),
        ("human", HUMAN_PROMPT_TEMPLATE),
    ])


def build_explain_chain(
    settings: PipelineSettings,
    temperature: float = 0.0,
) -> Runnable:
    """Build the LCEL pipeline: prompt | model | parser.

    Stages:
        1. ChatPromptTemplate: receives dict (topic, audience, format) -> returns ChatPromptValue
        2. ChatGroq: receives ChatPromptValue -> returns AIMessage
        3. StrOutputParser: receives AIMessage -> returns str
    """
    prompt = build_explain_prompt()
    model = ChatGroq(
        model=settings.model,
        temperature=temperature,
        groq_api_key=settings.groq_api_key,
    )
    parser = StrOutputParser()
    return prompt | model | parser


def format_prompt_display(topic: str, audience: str, format_spec: str) -> str:
    """Format and return the prompt messages without invoking the model."""
    prompt = build_explain_prompt()
    formatted = prompt.format_messages(topic=topic, audience=audience, format=format_spec)
    lines = [
        "=== FORMATTED PROMPT (No API Call) ===",
        f"Target Audience: {audience}",
        f"Format Spec:     {format_spec}",
        f"Topic:           {topic}",
        "--- Messages ---",
    ]
    for msg in formatted:
        role = msg.type.capitalize()
        lines.append(f"[{role} Message]:\n{msg.content}\n")
    lines.append("=" * 38)
    return "\n".join(lines)


def run_explain(
    topic: Optional[str],
    audience: str,
    format_spec: str,
    settings: PipelineSettings,
    show_prompt: bool = False,
) -> str:
    """Execute explain mode or show formatted prompt.

    Args:
        topic: Technical topic to explain.
        audience: Target audience.
        format_spec: Desired output format.
        settings: PipelineSettings containing model and API key.
        show_prompt: If True, prints formatted prompt and skips API call.

    Returns:
        Generated explanation string or formatted prompt string.

    Raises:
        ValueError: If topic is missing or whitespace-only.
    """
    cleaned_topic = topic.strip() if topic else ""
    if not cleaned_topic:
        raise ValueError("Topic cannot be empty or whitespace-only.")

    if show_prompt:
        return format_prompt_display(cleaned_topic, audience, format_spec)

    chain = build_explain_chain(settings)
    result = chain.invoke({
        "topic": cleaned_topic,
        "audience": audience,
        "format": format_spec,
    })
    return str(result).strip()

