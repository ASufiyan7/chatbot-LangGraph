"""
reviewer.py  –  NEXUS Reviewer Agent
"""
from __future__ import annotations
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from config.llm_factory import get_llm
from graph.state import NexusState

SYSTEM_PROMPT = """You are NEXUS Reviewer — a senior software engineer conducting a thorough code review.

Analyse the provided code across these dimensions:

1. CORRECTNESS   – Does it solve the task? Any logical errors?
2. SECURITY      – Dangerous calls? Injection risks? Hardcoded secrets? Unsafe eval/exec?
3. PERFORMANCE   – O(n²) where O(n) is possible? Unnecessary loops or allocations?
4. STYLE         – Naming conventions, readability, docstrings, dead code?
5. EDGE CASES    – Unhandled None/null, empty inputs, integer overflow?

Format your response EXACTLY:
CORRECTNESS: <short assessment>
SECURITY: <short assessment>
PERFORMANCE: <short assessment>
STYLE: <short assessment>
EDGE_CASES: <short assessment>
OVERALL_NOTES: <2-3 sentences summarising the most important improvements needed>
"""

def reviewer_node(state: NexusState) -> dict:
    llm = get_llm(temperature=0.1)

    code     = state.get("generated_code", "")
    language = state.get("language", "python")
    task     = state["messages"][0].content

    if not code:
        return {
            "review_notes": "No code available to review.",
            "messages": [AIMessage(content="⚠️ Reviewer: No code to analyse.")],
        }

    context = (
        f"LANGUAGE: {language}\n\n"
        f"TASK: {task}\n\n"
        f"CODE TO REVIEW:\n```{language}\n{code}\n```"
    )

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=context),
    ])

    review_text = response.content.strip()

    return {
        "review_notes": review_text,
        "messages":     [AIMessage(content=f"📋 **Code Review**\n\n{review_text}")],
    }