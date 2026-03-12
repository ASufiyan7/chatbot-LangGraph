"""
llm_factory.py  –  Returns a ready-to-use Gemini chat model
"""
from __future__ import annotations
from functools import lru_cache
from config.settings import GEMINI_API_KEY, GEMINI_MODEL


@lru_cache(maxsize=4)
def get_llm(temperature: float = 0.1):
    """Return a cached LangChain Gemini chat model."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=temperature,
        convert_system_message_to_human=True,  # Gemini requires this
    )