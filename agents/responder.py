
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from config.llm_factory import get_llm, invoke_with_limit
from graph.state import NexusState

SYSTEM_PROMPT = """\
You are NEXUS — an expert AI software engineer delivering a final answer.

Synthesise the context below into a clean, helpful response.

Structure:
1. Brief explanation of what was built / found / fixed (2-4 sentences)
2. The final code in a properly fenced code block (if any)
3. Example usage or expected output (if relevant)
4. Important caveats or next steps (if any)

Rules:
  - Be concise and professional.
  - Do NOT repeat agent logs or intermediate scores.
  - Do NOT say "As an AI" or similar filler.
  - If no code was generated, give a direct thorough prose answer.
"""

from langchain_core.messages import HumanMessage, SystemMessage

def responder_node(state: NexusState) -> dict:
    llm = get_llm(provider="groq", temperature=0.3)
    
    # FIX: Get the latest Human message from history instead of the first one [state["messages"][0]]
    # This ensures it answers the current question, not the first one you ever asked.
    human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    task = human_messages[-1].content if human_messages else "No task found."
    
    code     = state.get("generated_code", "")
    exec_res = state.get("execution_result", "")
    language = state.get("language", "python")
    plan     = state.get("plan", "")
    score    = state.get("quality_score")

    score_str = ""
    if score:
        badge = "✅" if score.get("overall", 0) >= 0.70 else "⚠️"
        score_str = (
            f"Quality: {badge} {score.get('overall', 0):.0%} overall. "
            f"Note: {score.get('feedback', '')}"
        )

    # Build context: If task_type is 'direct', plan and code will be empty strings 
    # due to the Orchestrator reset above, allowing for a clean prose answer.
    context = "\n\n".join(filter(None, [
        f"USER TASK:\n{task}",
        f"PLAN:\n{plan}"                                          if plan     else "",
        f"FINAL CODE ({language}):\n```{language}\n{code}\n```"  if code     else "",
        f"EXECUTION OUTPUT:\n{exec_res}"                          if exec_res else "",
        score_str,
    ]))

    response = invoke_with_limit(
        llm,
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=context)],
        provider="groq",
    )

    # LangGraph automatically appends this message to state["messages"]
    return {"messages": [response]}