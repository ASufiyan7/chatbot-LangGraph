"""
nexus_graph.py  –  NEXUS LangGraph definition

Full agent topology:

  START
    │
    ▼
  orchestrator
    │
    ├─── code_gen/debug/review ──► coder
    │                                │
    │                    ┌─── tool call? ───┐
    │                    ▼                  ▼
    │              coder_tools          debugger
    │                    │                  │
    │              sync_coder_exec          │
    │                    │                  │
    │              ┌─ ok? ─┐               │
    │              ▼       ▼               │
    │          reviewer  debugger          │
    │              │       │               │
    │              │  debugger_tools       │
    │              │       │               │
    │              │  sync_dbg_exec        │
    │              │       │               │
    │              └───────┴───────────────┘
    │                      │
    │                   reviewer
    │                      │
    │                   critic
    │                      │
    │           ┌── pass? ─┴── revise? ──┐
    │           ▼                        ▼
    │       responder             bump_revision
    │           │                        │
    │          END                    coder (loop)
    │
    └─── direct/explain ──► responder ──► END
"""
from __future__ import annotations
from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from graph.state import NexusState
from agents.orchestrator import orchestrator_node
from agents.coder        import coder_node, coder_tool_node
from agents.debugger     import debugger_node, debugger_tool_node
from agents.reviewer     import reviewer_node
from agents.critic       import critic_node, critic_router
from agents.responder    import responder_node


# ── routers ───────────────────────────────────────────────────────────────────

def orchestrator_router(state: NexusState) -> Literal["coder", "responder"]:
    """Skip the pipeline for simple direct/explain tasks."""
    if state["task_type"] in ("direct", "explain"):
        return "responder"
    return "coder"


def coder_router(state: NexusState) -> Literal["coder_tools", "debugger"]:
    """If coder called execute_code, run it; otherwise skip straight to debugger."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "coder_tools"
    return "debugger"


def execution_router(state: NexusState) -> Literal["debugger", "reviewer"]:
    """After sandbox run — did it succeed or fail?"""
    if not state.get("execution_ok", True):
        return "debugger"
    return "reviewer"


def debugger_router(state: NexusState) -> Literal["debugger_tools", "reviewer"]:
    """If debugger wants to test a fix, run it; then go to reviewer."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "debugger_tools"
    return "reviewer"


def revision_router(state: NexusState) -> Literal["coder"]:
    """After bump_revision, always loop back to coder."""
    return "coder"


# ── helper nodes ──────────────────────────────────────────────────────────────

def _bump_revision(state: NexusState) -> dict:
    """Increment the revision counter before looping back to coder."""
    return {"revision_count": state.get("revision_count", 0) + 1}


def _sync_execution_result(state: NexusState) -> dict:
    """
    After a ToolNode runs execute_code, pull the result out of the last
    ToolMessage and store it in state fields the other agents can read.
    """
    from langchain_core.messages import ToolMessage
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            content = msg.content or ""
            exec_ok = "Exit code: 0" in content or "✅" in content
            return {
                "execution_result": content,
                "execution_ok":     exec_ok,
            }
    return {"execution_result": "", "execution_ok": False}


# ── graph assembly ────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(NexusState)

    # ── register nodes ────────────────────────────────────────────────────────
    g.add_node("orchestrator",    orchestrator_node)
    g.add_node("coder",           coder_node)
    g.add_node("coder_tools",     coder_tool_node)
    g.add_node("sync_coder_exec", _sync_execution_result)
    g.add_node("debugger",        debugger_node)
    g.add_node("debugger_tools",  debugger_tool_node)
    g.add_node("sync_dbg_exec",   _sync_execution_result)
    g.add_node("reviewer",        reviewer_node)
    g.add_node("critic",          critic_node)
    g.add_node("bump_revision",   _bump_revision)
    g.add_node("responder",       responder_node)

    # ── edges ─────────────────────────────────────────────────────────────────
    g.add_edge(START, "orchestrator")

    g.add_conditional_edges(
        "orchestrator",
        orchestrator_router,
        {"coder": "coder", "responder": "responder"},
    )

    g.add_conditional_edges(
        "coder",
        coder_router,
        {"coder_tools": "coder_tools", "debugger": "debugger"},
    )

    g.add_edge("coder_tools", "sync_coder_exec")

    g.add_conditional_edges(
        "sync_coder_exec",
        execution_router,
        {"debugger": "debugger", "reviewer": "reviewer"},
    )

    g.add_conditional_edges(
        "debugger",
        debugger_router,
        {"debugger_tools": "debugger_tools", "reviewer": "reviewer"},
    )

    g.add_edge("debugger_tools", "sync_dbg_exec")
    g.add_edge("sync_dbg_exec",  "reviewer")

    g.add_edge("reviewer", "critic")

    g.add_conditional_edges(
        "critic",
        critic_router,
        {"respond": "responder", "revise": "bump_revision"},
    )

    g.add_conditional_edges(
        "bump_revision",
        revision_router,
        {"coder": "coder"},
    )

    g.add_edge("responder", END)

    return g


# ── compiled singleton ────────────────────────────────────────────────────────

nexus_graph = build_graph().compile(checkpointer=InMemorySaver())