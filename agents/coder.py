"""
coder.py  –  NEXUS Coder Agent
"""
from __future__ import annotations
import re
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode
from config.llm_factory import get_llm
from graph.state import NexusState
from tools.sandbox import execute_code

CODER_TOOLS = [execute_code]
coder_tool_node = ToolNode(CODER_TOOLS)

SYSTEM_PROMPT = """You are NEXUS Coder — an expert software engineer.

Your mission:
- Write clean, efficient, well-commented {language} code that solves the user's request.
- Follow the execution plan provided by the orchestrator.
- After writing code, you MUST call the execute_code tool to verify it runs correctly.
- If memory context is provided, learn from past episodes to avoid previous mistakes.

Code quality rules:
  • Use meaningful variable names
  • Add a brief docstring to every function
  • Handle edge cases and exceptions
  • Keep it concise — no bloat

IMPORTANT: Output the code in a single ```{language} ... ``` block, then call execute_code.
"""

def coder_node(state: NexusState) -> dict:
    llm = get_llm(provider="ollama", model_name="deepseek-coder:6.7b", temperature=0.2)
    llm_with_tools = llm.bind_tools(CODER_TOOLS)

    system = SYSTEM_PROMPT.format(language=state.get("language", "python"))

    context_parts = [f"ORCHESTRATOR PLAN:\n{state.get('plan', '')}"]
    if state.get("memory_context"):
        context_parts.append(f"RELEVANT PAST EPISODES:\n{state['memory_context']}")
    if state.get("review_notes"):
        context_parts.append(f"REVIEWER FEEDBACK (previous attempt):\n{state['review_notes']}")

    context = "\n\n".join(context_parts)

    response = llm_with_tools.invoke([
        SystemMessage(content=system),
        HumanMessage(content=context),
        *state["messages"],
    ])

    generated_code = _extract_code(response.content)

    return {
        "messages":       [response],
        "generated_code": generated_code,
    }

def _extract_code(text: str) -> str:
    m = re.search(r"```(?:\w+)?\n([\s\S]+?)```", text)
    return m.group(1).strip() if m else text.strip()