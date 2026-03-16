
from __future__ import annotations
from typing import Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agents.coder         import coder_node
from agents.orchestrator  import orchestrator_node
from agents.quality_gate  import quality_gate_node, quality_gate_router
from agents.responder     import responder_node
from graph.state          import NexusState


#  Routers 

def orchestrator_router(state: NexusState) -> Literal["coder", "responder"]:
    if state["task_type"] in ("direct", "explain"):
        return "responder"
    return "coder"


def _bump_revision(state: NexusState) -> dict:
    return {
        "revision_count": state.get("revision_count", 0) + 1,
        "exec_retries":   0,  
    }


#  Graph  
def build_graph() -> StateGraph:
    g = StateGraph(NexusState)

    # Nodes
    g.add_node("orchestrator",  orchestrator_node)
    g.add_node("coder",         coder_node)
    g.add_node("quality_gate",  quality_gate_node)
    g.add_node("bump_revision", _bump_revision)
    g.add_node("responder",     responder_node)

    # Edges
    g.add_edge(START, "orchestrator")

    g.add_conditional_edges(
        "orchestrator",
        orchestrator_router,
        {"coder": "coder", "responder": "responder"},
    )

    g.add_edge("coder", "quality_gate")

    g.add_conditional_edges(
        "quality_gate",
        quality_gate_router,
        {"respond": "responder", "revise": "bump_revision"},
    )

    g.add_edge("bump_revision", "coder")  

    g.add_edge("responder", END)

    return g


nexus_graph = build_graph().compile(checkpointer=InMemorySaver())
