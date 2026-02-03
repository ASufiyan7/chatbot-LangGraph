import os
from typing import List, TypedDict, Annotated, Literal

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.messages import (
    HumanMessage,
    BaseMessage,
    SystemMessage,
    AIMessage,
)
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode

# ENV
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")

app = FastAPI(title="Multi-Agent LangGraph (Safe & Robust)")

# MODEL
llm_engine = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.1,
    huggingfacehub_api_token=HF_TOKEN,
)

llm = ChatHuggingFace(llm=llm_engine)

# TOOLS
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"The weather in {city} is sunny with 25°C."


math_tools = [multiply]
research_tools = [get_weather]

math_tool_node = ToolNode(math_tools)
research_tool_node = ToolNode(research_tools)

# STATE
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    task_type: str

# SUPERVISOR (ROUTER ONLY — NO TOOLS)
def supervisor_node(state: AgentState):
    user_input = state["messages"][-1].content

    prompt = (
        "You are a supervisor.\n"
        "Decide who should handle the task.\n\n"
        "Return ONLY one word:\n"
        "- research (info / weather)\n"
        "- math (calculations)\n"
        "- both (needs info + math)\n"
        "- direct (just answer)\n\n"
        f"User request: {user_input}"
    )

    decision = llm.invoke(prompt).content.lower().strip()

    if "both" in decision:
        decision = "both"
    elif "math" in decision:
        decision = "math"
    elif "research" in decision:
        decision = "research"
    else:
        decision = "direct"

    return {"task_type": decision}

# RESEARCHER AGENT
def researcher_agent(state: AgentState):
    llm_with_tools = llm.bind_tools(research_tools)

    response = llm_with_tools.invoke(
        [SystemMessage(content="You are a researcher. Use tools if needed.")]
        + state["messages"]
    )

    return {"messages": [response]}

# MATH AGENT (SAFE)
def math_agent(state: AgentState):
    user_text = state["messages"][-1].content

    # 🚫 Invalid math → graceful failure (NO tools)
    if not any(ch.isdigit() for ch in user_text):
        return {
            "messages": [
                AIMessage(
                    content=(
                        "I can’t perform multiplication because the request "
                        "does not contain valid numbers."
                    )
                )
            ]
        }

    llm_with_tools = llm.bind_tools(math_tools)

    response = llm_with_tools.invoke(
        [SystemMessage(content="You are a math expert. Use tools only when valid.")]
        + state["messages"]
    )

    return {"messages": [response]}

# FINAL ANSWER
def final_node(state: AgentState):
    response = llm.invoke(
        state["messages"]
        + [SystemMessage(content="Provide a clear final answer to the user.")]
    )
    return {"messages": [response]}

# ROUTERS
def supervisor_router(state: AgentState) -> Literal["research", "math", "both", "direct"]:
    return state["task_type"]


def math_router(state: AgentState) -> Literal["tools", "final"]:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "final"

# GRAPH
graph = StateGraph(AgentState)

graph.add_node("supervisor", supervisor_node)
graph.add_node("researcher", researcher_agent)
graph.add_node("research_tools", research_tool_node)
graph.add_node("math_agent", math_agent)
graph.add_node("math_tools", math_tool_node)
graph.add_node("final", final_node)

graph.add_edge(START, "supervisor")

graph.add_conditional_edges(
    "supervisor",
    supervisor_router,
    {
        "research": "researcher",
        "math": "math_agent",
        "both": "researcher",
        "direct": "final",
    },
)

# Research → tools → math
graph.add_edge("researcher", "research_tools")
graph.add_edge("research_tools", "math_agent")

# Math → (router)
graph.add_conditional_edges(
    "math_agent",
    math_router,
    {
        "tools": "math_tools",
        "final": "final",
    },
)

graph.add_edge("math_tools", "final")
graph.add_edge("final", END)

app_graph = graph.compile(checkpointer=InMemorySaver())

# API
class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    config = {"configurable": {"thread_id": req.thread_id}}

    output = app_graph.invoke(
        {"messages": [HumanMessage(content=req.message)]},
        config=config,
    )

    return {"response": output["messages"][-1].content}


@app.get("/")
def root():
    return {"status": "Multi-Agent LangGraph system running safely"}
