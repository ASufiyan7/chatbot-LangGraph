"""
orchestrator.py  –  NEXUS Orchestrator Agent
"""
from __future__ import annotations
import re
import time
from langchain_core.messages import SystemMessage, HumanMessage
from config.llm_factory import get_llm
from graph.state import NexusState
from memory.memory import recall_memory

SYSTEM_PROMPT = """You are NEXUS Orchestrator — the strategic brain of an autonomous software engineering system.

Your job is to:
1. Classify the user request into EXACTLY one task type:
   - code_gen   : user wants new code written
   - debug      : user has broken/buggy code to fix
   - review     : user wants code quality / security analysis
   - explain    : user wants code explained or documented
   - direct     : general question, no code work needed

2. Write a numbered step-by-step PLAN (3-5 steps) for how the agent team will solve this.

3. Detect the programming language (python, javascript, java, etc. — or "unknown").

Respond in EXACTLY this format (no extra text):
TASK_TYPE: <one of the types above>
LANGUAGE: <language or unknown>
PLAN:
1. <step one>
2. <step two>
3. <step three>
"""

def orchestrator_node(state: NexusState) -> dict:
    llm = get_llm(provider="groq", temperature=0.1)
    user_message = state["messages"][-1].content

    memory_ctx = recall_memory(user_message)
    memory_block = ""
    if memory_ctx:
        memory_block = (
            "\n\n[RELEVANT PAST EPISODES FROM MEMORY]\n"
            + memory_ctx
            + "\n[END MEMORY]\n"
        )

    full_user_content = user_message + memory_block

    time.sleep(2) 

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=full_user_content),
    ])

    raw = response.content.strip()

    task_type = _extract(r"TASK_TYPE:\s*(\w+)", raw, default="code_gen")
    language  = _extract(r"LANGUAGE:\s*(\w+)",  raw, default="python")
    plan      = _extract_plan(raw)

    valid_types = {"code_gen", "debug", "review", "explain", "direct"}
    if task_type not in valid_types:
        task_type = "code_gen"

    return {
        "task_type":      task_type,
        "language":       language,
        "plan":           plan,
        "memory_context": memory_ctx,
        "revision_count": 0,
    }

def _extract(pattern: str, text: str, default: str = "") -> str:
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1).lower().strip() if m else default

def _extract_plan(text: str) -> str:
    m = re.search(r"PLAN:\s*([\s\S]+)", text, re.IGNORECASE)
    return m.group(1).strip() if m else "No plan generated."