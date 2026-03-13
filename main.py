"""
main.py  –  NEXUS FastAPI server

Endpoints:
  POST /chat              – standard request/response
  WS   /ws/chat/{tid}     – streaming WebSocket (agent events in real-time)
  GET  /graph/schema      – returns the graph structure for frontend viz
  GET  /health            – health check
"""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asyncio
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.messages import HumanMessage

from graph.nexus_graph import nexus_graph

# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NEXUS — Autonomous Multi-Agent Software Engineer",
    description="Self-correcting, memory-augmented AI coding system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── schemas ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:   str
    thread_id: str = "default"


class ChatResponse(BaseModel):
    thread_id:        str
    response:         str
    task_type:        str
    language:         str
    plan:             str
    generated_code:   str
    execution_result: str
    execution_ok:     bool
    review_notes:     str
    critic_score:     dict | None
    revision_count:   int


# ── REST endpoint ─────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    config = {"configurable": {"thread_id": req.thread_id}}

    try:
        output = await asyncio.to_thread(
            nexus_graph.invoke,
            {"messages": [HumanMessage(content=req.message)]},
            config,
        )
    except Exception as exc:
        # Common failure mode: quota / rate limit on Gemini API.
        raise HTTPException(
            status_code=503,
            detail=(
                "LLM request failed (possible quota/rate-limit issue). "
                "Check your Gemini billing/quota or change model via GEMINI_MODEL. "
                f"Original error: {exc}"
            ),
        )

    return ChatResponse(
        thread_id        = req.thread_id,
        response         = output["messages"][-1].content,
        task_type        = output.get("task_type", ""),
        language         = output.get("language", ""),
        plan             = output.get("plan", ""),
        generated_code   = output.get("generated_code", ""),
        execution_result = output.get("execution_result", ""),
        execution_ok     = output.get("execution_ok", False),
        review_notes     = output.get("review_notes", ""),
        critic_score     = output.get("critic_score"),
        revision_count   = output.get("revision_count", 0),
    )


# ── WebSocket streaming ───────────────────────────────────────────────────────
_AGENT_NODES = {
    "orchestrator", "coder", "debugger",
    "reviewer",     "critic", "responder",
}

@app.websocket("/ws/chat/{thread_id}")
async def ws_chat(websocket: WebSocket, thread_id: str):
    """
    Stream agent events to the frontend in real-time.
    Message format: { "event": str, "node": str, "data": str }
    """
    await websocket.accept()
    try:
        raw          = await websocket.receive_text()
        payload      = json.loads(raw)
        user_message = payload.get("message", "")
        config       = {"configurable": {"thread_id": thread_id}}

        async for event in nexus_graph.astream_events(
            {"messages": [HumanMessage(content=user_message)]},
            config=config,
            version="v2",
        ):
            kind = event.get("event", "")
            name = event.get("name", "")

            if kind == "on_chain_start" and name in _AGENT_NODES:
                await websocket.send_json({
                    "event": "node_start",
                    "node":  name,
                    "data":  f"🔄 {name} is working...",
                })

            elif kind == "on_chain_end" and name in _AGENT_NODES:
                msgs    = event.get("data", {}).get("output", {}).get("messages", [])
                preview = msgs[-1].content[:200] if msgs else ""
                await websocket.send_json({
                    "event": "node_end",
                    "node":  name,
                    "data":  preview,
                })

            elif kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    await websocket.send_json({
                        "event": "token",
                        "node":  name,
                        "data":  chunk.content,
                    })

        await websocket.send_json({"event": "done", "node": "END", "data": ""})

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        await websocket.send_json({
            "event": "error",
            "node":  "system",
            "data":  str(exc),
        })


# ── graph schema for frontend visualisation ───────────────────────────────────
@app.get("/graph/schema")
def graph_schema():
    return {
        "nodes": [
            {"id": "orchestrator", "label": "Orchestrator", "type": "planner",  "color": "#6366f1"},
            {"id": "coder",        "label": "Coder",         "type": "worker",   "color": "#22c55e"},
            {"id": "coder_tools",  "label": "Sandbox",       "type": "tool",     "color": "#f59e0b"},
            {"id": "debugger",     "label": "Debugger",      "type": "worker",   "color": "#ef4444"},
            {"id": "reviewer",     "label": "Reviewer",      "type": "reviewer", "color": "#8b5cf6"},
            {"id": "critic",       "label": "Critic",        "type": "critic",   "color": "#f97316"},
            {"id": "responder",    "label": "Responder",     "type": "output",   "color": "#14b8a6"},
        ],
        "edges": [
            {"from": "orchestrator", "to": "coder",       "label": "code task"},
            {"from": "orchestrator", "to": "responder",   "label": "direct"},
            {"from": "coder",        "to": "coder_tools", "label": "execute"},
            {"from": "coder",        "to": "debugger",    "label": "skip exec"},
            {"from": "coder_tools",  "to": "debugger",    "label": "error"},
            {"from": "coder_tools",  "to": "reviewer",    "label": "ok"},
            {"from": "debugger",     "to": "reviewer",    "label": "fixed"},
            {"from": "reviewer",     "to": "critic",      "label": "reviewed"},
            {"from": "critic",       "to": "responder",   "label": "pass ✅"},
            {"from": "critic",       "to": "coder",       "label": "revise 🔁"},
            {"from": "responder",    "to": "END",         "label": "done"},
        ],
    }


# ── health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":  "NEXUS online",
        "agents":  list(_AGENT_NODES),
        "version": "1.0.0",
    }