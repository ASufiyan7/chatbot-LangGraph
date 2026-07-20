"""
agents/coder.py — NEXUS Coder + Executor

Provider: Groq (llama-3.3-70b-versatile)
Why:  Groq runs at ~400 tokens/second on cloud hardware — 20-50x faster
     than any local Ollama model on a CPU-only machine. On an 8 GB laptop
     without a dedicated GPU, local inference is simply too slow for a
     good user experience. Groq's free tier (14,400 req/day) is more than
     enough for all four agents combined.

Ollama is kept as an OPTIONAL provider — set CODER_PROVIDER=ollama in
.env if you have a machine with a dedicated GPU where local inference
is actually fast. Otherwise leave it as "groq".
"""
from __future__ import annotations
import logging
import re

from langchain_core.messages import AIMessage, SystemMessage

from config.llm_factory import get_llm, invoke_with_limit
from config.settings import CODER_PROVIDER, MAX_EXEC_RETRIES, OLLAMA_MODEL
from graph.state import NexusState
from tools.sandbox import execute_code

log = logging.getLogger("nexus.coder")

WRITE_PROMPT = """\
You are NEXUS Coder — an expert software engineer.

Task plan from orchestrator:
{plan}

{memory_block}\
{feedback_block}\

Rules:
  - Write clean, readable {language} code that fully solves the task.
  - Add a brief docstring to every function.
  - Handle edge cases and exceptions gracefully.
  - Output your code in a SINGLE fenced block: ```{language} ... ```
  - Do NOT add explanations outside the code block.
"""

FIX_PROMPT = """\
You are NEXUS Coder fixing a bug.

Original task:
{task}

Code that failed:
```{language}
{code}
```

Execution error:
{error}

Instructions:
  1. Identify the ROOT CAUSE in one comment line at the top.
  2. Output the COMPLETE fixed code in a single fenced block.
  3. Change only what is broken — do not rewrite working parts.
"""


def _make_llm():
    """Return the configured coder LLM."""
    if CODER_PROVIDER == "ollama":
        return get_llm(provider="ollama", model_name=OLLAMA_MODEL, temperature=0.2), "ollama"
    return get_llm(provider="groq", temperature=0.2), "groq"


def coder_node(state: NexusState) -> dict:
    llm, provider = _make_llm()
    language      = state.get("language", "python")
    task          = state["messages"][0].content

    # ── Build context ─────────────────────────────────────────────────────────
    memory_block = ""
    if state.get("memory_context"):
        memory_block = f"Relevant past episodes:\n{state['memory_context']}\n\n"

    feedback_block = ""
    if state.get("quality_score") and state.get("revision_count", 0) > 0:
        fb  = state["quality_score"].get("feedback", "")
        rev = state["quality_score"].get("review", "")
        if fb or rev:
            feedback_block = (
                f"Quality gate feedback from previous attempt:\n{rev}\n{fb}\n\n"
            )

    write_prompt = WRITE_PROMPT.format(
        plan=state.get("plan", ""),
        language=language,
        memory_block=memory_block,
        feedback_block=feedback_block,
    )

    # ── Generate code ─────────────────────────────────────────────────────────
    response = invoke_with_limit(
        llm,
        [SystemMessage(content=write_prompt), *state["messages"]],
        provider=provider,
    )

    code = _extract_code(response.content)

    # ── Execute + self-fix loop ───────────────────────────────────────────────
    exec_retries = state.get("exec_retries", 0)
    exec_result  = ""
    exec_ok      = False

    for attempt in range(MAX_EXEC_RETRIES + 1):
        exec_result = execute_code.invoke({"code": code, "language": language})
        exec_ok     = _is_ok(exec_result)

        if exec_ok or attempt >= MAX_EXEC_RETRIES:
            break

        fix_response = invoke_with_limit(
            llm,
            [SystemMessage(content=FIX_PROMPT.format(
                task=task, language=language, code=code, error=exec_result,
            ))],
            provider=provider,
        )
        code          = _extract_code(fix_response.content)
        exec_retries += 1

    return {
        "messages":         [AIMessage(content=response.content)],
        "generated_code":   code,
        "execution_result": exec_result,
        "execution_ok":     exec_ok,
        "exec_retries":     exec_retries,
    }


def _extract_code(text: str) -> str:
    m = re.search(r"```(?:\w+)?\n([\s\S]+?)```", text)
    return m.group(1).strip() if m else text.strip()


def _is_ok(result: str) -> bool:
    return "Exit code: 0" in result or "✅" in result