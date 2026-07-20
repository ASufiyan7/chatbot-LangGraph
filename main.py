from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asyncio
import json
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from graph.nexus_graph import nexus_graph

log = logging.getLogger("nexus.api")


app = FastAPI(
    title="NEXUS — Autonomous Multi-Agent Software Engineer (v2)",
    description="4-agent, rate-limit-aware, resource-efficient coding system",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    quality_score:    dict | None
    revision_count:   int
    exec_retries:     int


#  REST endpoint 

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
        log.exception("Agent pipeline failed")
        return ChatResponse(
            thread_id=req.thread_id,
            response=(
                "I could not complete that request because the model backend failed. "
                "If this mentions Ollama memory, restart Ollama and the FastAPI server; "
                "the app is now configured to use a smaller context and a local fallback model."
            ),
            task_type="direct",
            language="unknown",
            plan="",
            generated_code="",
            execution_result=str(exc),
            execution_ok=False,
            quality_score=None,
            revision_count=0,
            exec_retries=0,
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
        quality_score    = output.get("quality_score"),
        revision_count   = output.get("revision_count", 0),
        exec_retries     = output.get("exec_retries", 0),
    )


#  WebSocket streaming

_AGENT_NODES = {"orchestrator", "coder", "quality_gate", "responder"}


@app.websocket("/ws/chat/{thread_id}")
async def ws_chat(websocket: WebSocket, thread_id: str):
    """
    Stream agent events to the frontend.
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


#  Graph schema for frontend visualisation 
@app.get("/graph/schema")
def graph_schema():
    return {
        "nodes": [
            {"id": "orchestrator",  "label": "Orchestrator",   "type": "planner",  "color": "#6366f1", "provider": "groq"},
            {"id": "coder",         "label": "Coder+Executor", "type": "worker",   "color": "#14b8a6", "provider": "ollama"},
            {"id": "quality_gate",  "label": "Quality Gate",   "type": "reviewer", "color": "#f97316", "provider": "gemini"},
            {"id": "responder",     "label": "Responder",      "type": "output",   "color": "#6366f1", "provider": "gemini"},
        ],
        "edges": [
            {"from": "orchestrator", "to": "coder",        "label": "code task"},
            {"from": "orchestrator", "to": "responder",    "label": "direct / explain"},
            {"from": "coder",        "to": "quality_gate", "label": "review + score"},
            {"from": "quality_gate", "to": "responder",    "label": "pass ✅"},
            {"from": "quality_gate", "to": "coder",        "label": "revise 🔁"},
            {"from": "responder",    "to": "END",          "label": "done"},
        ],
        "providers": {
            "groq":   {"color": "#22c55e", "label": "Groq (fast routing)"},
            "ollama": {"color": "#3b82f6", "label": "Ollama (local coding)"},
            "gemini": {"color": "#a855f7", "label": "Gemini (quality + response)"},
        },
    }


#  Health check 

@app.get("/health")
def health():
    return {
        "status":  "NEXUS v2 online",
        "agents":  list(_AGENT_NODES),
        "version": "2.0.0",
        "providers": {
            "orchestrator":  "groq",
            "coder":         "ollama",
            "quality_gate":  "gemini",
            "responder":     "gemini",
        },
    }
