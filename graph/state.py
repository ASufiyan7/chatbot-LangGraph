"""
state.py  –  NEXUS shared graph state
Every field that any agent reads or writes lives here.
"""
from __future__ import annotations
from typing import Annotated, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class CriticScore(TypedDict):
    correctness: float   # 0-1
    security:    float   # 0-1
    style:       float   # 0-1
    overall:     float   # weighted average
    feedback:    str     # improvement notes


class NexusState(TypedDict):
    # ── conversation ──────────────────────────────────────────────────────────
    messages: Annotated[List[BaseMessage], add_messages]

    # ── orchestrator decision ─────────────────────────────────────────────────
    task_type: str       # "code_gen" | "debug" | "review" | "explain" | "direct"
    plan:      str       # step-by-step plan from orchestrator

    # ── code artefacts ────────────────────────────────────────────────────────
    generated_code:   str
    language:         str
    execution_result: str
    execution_ok:     bool

    # ── review & reflection ───────────────────────────────────────────────────
    review_notes:   str
    critic_score:   Optional[CriticScore]
    revision_count: int

    # ── memory ────────────────────────────────────────────────────────────────
    memory_context: str