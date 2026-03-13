"""
critic.py  –  NEXUS Critic / Reflector Agent
"""
from __future__ import annotations
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from config.llm_factory import get_llm
from config.settings import CRITIC_PASS_THRESHOLD, MAX_REVISION_LOOPS
from graph.state import NexusState
from memory.memory import save_memory

# 1. Define the exact schema we want the LLM to output
class CriticOutput(BaseModel):
    correctness: float = Field(description="Does the code correctly solve the stated task? Score 0.0 to 1.0.")
    security: float = Field(description="Is the code free of dangerous patterns? Score 0.0 to 1.0.")
    style: float = Field(description="Is the code clean and readable? Score 0.0 to 1.0.")
    overall: float = Field(description="Weighted average: 0.5*correctness + 0.3*security + 0.2*style")
    feedback: str = Field(description="One sentence explaining the biggest remaining issue, or 'Looks great!'")

SYSTEM_PROMPT = """You are NEXUS Critic — a ruthlessly objective quality gate.

Score the provided code on Correctness (weight 0.5), Security (weight 0.3), and Style (weight 0.2).
Compute the OVERALL score using those weights.
"""

def critic_node(state: NexusState, config: RunnableConfig) -> dict:
    llm = get_llm(temperature=0.0)
    
    # 2. Bind the Pydantic model to force structured JSON output
    structured_llm = llm.with_structured_output(CriticOutput)

    code     = state.get("generated_code", "")
    task     = state["messages"][0].content
    review   = state.get("review_notes", "")
    exec_res = state.get("execution_result", "")
    exec_ok  = state.get("execution_ok", False)

    session_id = config.get("configurable", {}).get("thread_id", "default_nexus_session")

    context = (
        f"TASK: {task}\n\n"
        f"CODE:\n{code}\n\n"
        f"EXECUTION RESULT (ok={exec_ok}):\n{exec_res}\n\n"
        f"REVIEWER NOTES:\n{review}"
    )

    # 3. Invoke directly returns our Pydantic object, no regex needed!
    result = structured_llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=context),
    ])
    
    # Convert Pydantic object back to dictionary for the LangGraph state
    score = result.model_dump()

    if score["overall"] >= CRITIC_PASS_THRESHOLD:
        save_memory(
            session_id=session_id,
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
