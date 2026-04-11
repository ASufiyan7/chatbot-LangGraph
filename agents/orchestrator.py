
from __future__ import annotations
import re

from langchain_core.messages import HumanMessage, SystemMessage
from config.llm_factory import get_llm, invoke_with_limit
from config.settings import GROQ_API_KEY
from graph.state import NexusState
from memory.memory import recall_memory

SYSTEM_PROMPT = """\
You are NEXUS Orchestrator — the strategic planner of an autonomous coding system.

Your job (do ALL three):
1. Classify the request into EXACTLY one task type:
   - code_gen   → write new code
   - debug      → fix broken/buggy code the user provided
   - review     → analyse code for quality / security
   - explain    → explain or document existing code
   - direct     → general question, no code needed

2. Detect the programming language (python, javascript, java, etc., or "unknown").

3. Write a numbered PLAN (3–5 steps) describing how the agent team will solve this.
   Keep each step to one short sentence.

Respond in EXACTLY this format — no preamble, no extra text:
TASK_TYPE: <type>
LANGUAGE: <language>
PLAN:
1. <step>
2. <step>
3. <step>
"""


def orchestrator_node(state: NexusState) -> dict:
    provider = "groq" if GROQ_API_KEY else "gemini"
    llm = get_llm(provider=provider, temperature=0.1)

    # Gets the LATEST message sent by the user
    user_message = state["messages"][-1].content

    # Pull relevant past episodes from memory 
    memory_ctx = recall_memory(user_message)
    memory_block = ""
    if memory_ctx:
        memory_block = f"\n\n[RELEVANT PAST EPISODES]\n{memory_ctx}\n[END MEMORY]\n"

    response = invoke_with_limit(
        llm,
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_message + memory_block),
        ],
        provider=provider,
    )

    raw = str(response.content).strip()

    task_type = _extract(r"TASK_TYPE:\s*(\w+)", raw, default="code_gen")
    language  = _extract(r"LANGUAGE:\s*(\w+)",  raw, default="python")
    plan      = _extract_plan(raw)

    valid_types = {"code_gen", "debug", "review", "explain", "direct"}
    if task_type not in valid_types:
        task_type = "code_gen"

    # CRITICAL FIX: We must return empty values for code/result/score 
    # to "clear" the state from the previous turn.
    return {
        "task_type":        task_type,
        "language":        language,
        "plan":            plan,
        "memory_context":   memory_ctx,
        "generated_code":   "",    # Reset old code
        "execution_result": "",    # Reset old results
        "execution_ok":     False, # Reset old status
        "quality_score":    None,  # Reset old score
        "revision_count":   0,
        "exec_retries":     0,
    }


#  helpers

def _extract(pattern: str, text: str, default: str = "") -> str:
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1).lower().strip() if m else default


def _extract_plan(text: str) -> str:
    m = re.search(r"PLAN:\s*([\s\S]+)", text, re.IGNORECASE)
    return m.group(1).strip() if m else "No plan generated."
