"""
critic.py  –  NEXUS Critic / Reflector Agent
"""
from __future__ import annotations
import re
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from config.llm_factory import get_llm
from config.settings import CRITIC_PASS_THRESHOLD, MAX_REVISION_LOOPS
from graph.state import NexusState, CriticScore
from memory.memory import save_memory

SYSTEM_PROMPT = """You are NEXUS Critic — a ruthlessly objective quality gate.

Score the code on three dimensions (each 0.0 – 1.0):
  • CORRECTNESS : Does the code correctly solve the stated task? (weighted 0.5)
  • SECURITY    : Is the code free of dangerous patterns? (weighted 0.3)
  • STYLE       : Is the code clean and readable? (weighted 0.2)

Compute: OVERALL = 0.5*correctness + 0.3*security + 0.2*style

Also write one FEEDBACK sentence explaining the biggest remaining issue (or "Looks great!" if overall ≥ 0.85).

Respond in EXACTLY this format (numbers only, no % signs):
CORRECTNESS: <0.0-1.0>
SECURITY: <0.0-1.0>
STYLE: <0.0-1.0>
OVERALL: <0.0-1.0>
FEEDBACK: <one sentence>
"""

def critic_node(state: NexusState) -> dict:
    llm = get_llm(temperature=0.0)

    code     = state.get("generated_code", "")
    task     = state["messages"][0].content
    review   = state.get("review_notes", "")
    exec_res = state.get("execution_result", "")
    exec_ok  = state.get("execution_ok", False)

    context = (
        f"TASK: {task}\n\n"
        f"CODE:\n{code}\n\n"
        f"EXECUTION RESULT (ok={exec_ok}):\n{exec_res}\n\n"
        f"REVIEWER NOTES:\n{review}"
    )

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=context),
    ])

    raw   = response.content.strip()
    score = _parse_score(raw)

    if score["overall"] >= CRITIC_PASS_THRESHOLD:
        save_memory(
            session_id="nexus",
            task=task,
            code=code,
            outcome=exec_res,
            score=score["overall"],
        )

    badge = "✅ PASS" if score["overall"] >= CRITIC_PASS_THRESHOLD else "🔁 NEEDS REVISION"
    score_card = (
        f"🎯 **Critic Score** — {badge}\n"
        f"  Correctness : {score['correctness']:.0%}\n"
        f"  Security    : {score['security']:.0%}\n"
        f"  Style       : {score['style']:.0%}\n"
        f"  **Overall   : {score['overall']:.0%}**\n"
        f"  Feedback    : {score['feedback']}"
    )

    return {
        "critic_score": score,
        "messages": [AIMessage(content=score_card)],
    }

def critic_router(state: NexusState) -> str:
    score    = state.get("critic_score")
    revision = state.get("revision_count", 0)
    if score is None:
        return "respond"
    if score["overall"] < CRITIC_PASS_THRESHOLD and revision < MAX_REVISION_LOOPS:
        return "revise"
    return "respond"

def _parse_score(text: str) -> CriticScore:
    def _f(pattern: str) -> float:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                return min(max(float(m.group(1)), 0.0), 1.0)
            except ValueError:
                pass
        return 0.5

    correctness = _f(r"CORRECTNESS:\s*([\d.]+)")
    security    = _f(r"SECURITY:\s*([\d.]+)")
    style       = _f(r"STYLE:\s*([\d.]+)")
    overall     = _f(r"OVERALL:\s*([\d.]+)")

    m_fb = re.search(r"FEEDBACK:\s*(.+)", text, re.IGNORECASE)
    feedback = m_fb.group(1).strip() if m_fb else "No feedback."

    return CriticScore(
        correctness=correctness,
        security=security,
        style=style,
        overall=overall,
        feedback=feedback,
    )