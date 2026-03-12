"""
responder.py  –  NEXUS Final Responder
"""
from __future__ import annotations
from langchain_core.messages import SystemMessage, HumanMessage
from config.llm_factory import get_llm
from graph.state import NexusState

SYSTEM_PROMPT = """You are NEXUS — an expert AI software engineer.

Synthesise everything below into a clean, helpful final response for the user.

Structure your response as:
1. Brief explanation of what was built / found / fixed
2. The final code (in a properly fenced code block)
3. Example usage or output (if relevant)
4. Any important caveats or next steps

Be concise, professional, and genuinely helpful. Do NOT repeat the agent discussion logs.
"""

def responder_node(state: NexusState) -> dict:
    llm = get_llm(temperature=0.3)

    task     = state["messages"][0].content
    code     = state.get("generated_code", "")
    exec_res = state.get("execution_result", "")
    review   = state.get("review_notes", "")
    score    = state.get("critic_score")
    language = state.get("language", "python")
    plan     = state.get("plan", "")

    score_str = ""
    if score:
        score_str = (
            f"Quality score: {score['overall']:.0%} "
            f"(correctness={score['correctness']:.0%}, "
            f"security={score['security']:.0%}, "
            f"style={score['style']:.0%})"
        )

    context = "\n\n".join(filter(None, [
        f"USER TASK: {task}",
        f"PLAN FOLLOWED:\n{plan}" if plan else "",
        f"FINAL CODE ({language}):\n```{language}\n{code}\n```" if code else "",
        f"EXECUTION OUTPUT:\n{exec_res}" if exec_res else "",
        f"REVIEW SUMMARY:\n{review}" if review else "",
        score_str,
    ]))

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=context),
    ])

    return {"messages": [response]}