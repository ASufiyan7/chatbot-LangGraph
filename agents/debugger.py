"""
debugger.py  –  NEXUS Debugger Agent
"""
from __future__ import annotations
import re
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode
from config.llm_factory import get_llm
from graph.state import NexusState
from tools.sandbox import execute_code

DEBUGGER_TOOLS = [execute_code]
debugger_tool_node = ToolNode(DEBUGGER_TOOLS)

SYSTEM_PROMPT = """You are NEXUS Debugger — a world-class bug hunter and code fixer.

You will receive:
  • The original user task
  • The code that was written
  • The execution result (stdout / stderr / exit code)

Your job:
  1. Read the error output carefully.
  2. Identify the ROOT CAUSE in one sentence.
  3. Write a FIXED version of the code.
  4. Call execute_code with the fixed code to confirm it now works.

Be surgical — change only what is broken. Do not rewrite working parts.

Format your response:
ROOT CAUSE: <one sentence>

FIXED CODE:
```{language}
<corrected code here>
```
"""

def debugger_node(state: NexusState) -> dict:
    llm = get_llm(temperature=0.1)
    llm_with_tools = llm.bind_tools(DEBUGGER_TOOLS)

    language = state.get("language", "python")
    system   = SYSTEM_PROMPT.format(language=language)

    context = (
        f"ORIGINAL TASK:\n{state['messages'][0].content}\n\n"
        f"WRITTEN CODE:\n{state.get('generated_code', '(no code found)')}\n\n"
        f"EXECUTION RESULT:\n{state.get('execution_result', '(no result)')}"
    )

    response = llm_with_tools.invoke([
        SystemMessage(content=system),
        HumanMessage(content=context),
    ])

    fixed_code = _extract_code(response.content)

    return {
        "messages":       [response],
        "generated_code": fixed_code,
    }

def _extract_code(text: str) -> str:
    m = re.search(r"```(?:\w+)?\n([\s\S]+?)```", text)
    return m.group(1).strip() if m else text.strip()