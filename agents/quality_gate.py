
from __future__ import annotations
import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from config.llm_factory import get_llm, invoke_with_limit
from config.settings import MAX_REVISION_LOOPS, QUALITY_PASS_THRESHOLD
from graph.state import NexusState
from memory.memory import save_memory

SYSTEM_PROMPT = """\
You are NEXUS Quality Gate — a senior engineer and strict code reviewer.

Analyse the code below and respond with ONLY a valid JSON object — no preamble, no markdown fences, nothing else.

JSON schema (all fields required):
{
  "correctness_notes": "<1-2 sentences: does it solve the task?>",
  "security_notes":    "<1-2 sentences: dangerous patterns, injection, unsafe calls?>",
  "performance_notes": "<1-2 sentences: obvious inefficiencies?>",
  "style_notes":       "<1-2 sentences: naming, readability, docstrings?>",
  "edge_case_notes":   "<1-2 sentences: unhandled None/empty/overflow?>",
  "correctness": <float 0.0-1.0>,
  "security":    <float 0.0-1.0>,
  "style":       <float 0.0-1.0>,
  "overall":     <float: 0.5*correctness + 0.3*security + 0.2*style>,
  "feedback":    "<one sentence: biggest issue, or 'Looks great!'>"
}

Scoring guide:
  1.0 = perfect   0.8 = minor issues   0.6 = needs work   0.4 = significant problems
Pass threshold is """ + str(QUALITY_PASS_THRESHOLD) + """. Be honest — do not inflate scores.
"""

_FALLBACK_PASS: dict = {
    "correctness": 0.80, "security": 0.80, "style": 0.80, "overall": 0.80,
    "feedback": "Auto-passed (JSON scoring unavailable).",
    "correctness_notes": "Could not parse structured review.",
    "security_notes": "", "performance_notes": "", "style_notes": "", "edge_case_notes": "",
    "review": "Quality gate scoring skipped — Groq did not return valid JSON.",
}


def _parse_score(text: str) -> dict | None:
    """Extract the first JSON object from the model's response."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        data = json.loads(text[start:end])
        # Validate required numeric keys
        for k in ("correctness", "security", "style", "overall"):
            data[k] = float(data[k])
        return data
    except Exception:
        return None


def quality_gate_node(state: NexusState, config: RunnableConfig) -> dict:
    code = state.get("generated_code", "")
    if not code:
        return {
            "quality_score": None,
            "messages": [AIMessage(content="⚠️ Quality Gate: no code to review.")],
        }

    llm  = get_llm(provider="groq", temperature=0.0)
    task = state["messages"][0].content
    exec_res = state.get("execution_result", "(no execution result)")
    exec_ok  = state.get("execution_ok", False)

    context = (
        f"TASK:\n{task}\n\n"
        f"LANGUAGE: {state.get('language', 'python')}\n\n"
        f"CODE:\n```\n{code}\n```\n\n"
        f"EXECUTION (ok={exec_ok}):\n{exec_res}"
    )

    try:
        response = invoke_with_limit(
            llm,
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=context)],
            provider="groq",
        )
        score = _parse_score(response.content)
    except Exception as exc:
        score = None
        import logging
        logging.getLogger("nexus.quality_gate").warning("LLM call failed: %s", exc)

    if score is None:
        score = dict(_FALLBACK_PASS)  # safe copy

    # Build full review text for the responder
    review_text = (
        f"CORRECTNESS : {score.get('correctness_notes', '')}\n"
        f"SECURITY    : {score.get('security_notes', '')}\n"
        f"PERFORMANCE : {score.get('performance_notes', '')}\n"
        f"STYLE       : {score.get('style_notes', '')}\n"
        f"EDGE CASES  : {score.get('edge_case_notes', '')}"
    )
    score["review"] = review_text

    # Save passing solutions to memory
    session_id = config.get("configurable", {}).get("thread_id", "default")
    if score["overall"] >= QUALITY_PASS_THRESHOLD:
        save_memory(
            session_id=session_id,
            task=task,
            code=code,
            outcome=exec_res,
            score=score["overall"],
        )

    badge = "✅ PASS" if score["overall"] >= QUALITY_PASS_THRESHOLD else "🔁 REVISE"
    card = (
        f"🎯 **Quality Gate** — {badge}\n\n"
        f"{review_text}\n\n"
        f"**Score**: correctness={score['correctness']:.0%}  "
        f"security={score['security']:.0%}  "
        f"style={score['style']:.0%}  "
        f"**overall={score['overall']:.0%}**\n"
        f"**Feedback**: {score['feedback']}"
    )

    return {
        "quality_score": score,
        "messages": [AIMessage(content=card)],
    }


def quality_gate_router(state: NexusState) -> str:
    score    = state.get("quality_score")
    revision = state.get("revision_count", 0)
    if score is None:
        return "respond"
    if score["overall"] < QUALITY_PASS_THRESHOLD and revision < MAX_REVISION_LOOPS:
        return "revise"
    return "respond"
