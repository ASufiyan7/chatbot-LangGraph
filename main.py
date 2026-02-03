import os
from typing import List, TypedDict, Annotated, Literal

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode

# ENV
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")

app = FastAPI(title="Robust Planner–Executor LangGraph Agent")

# MODEL
llm_engine = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.1,
    huggingfacehub_api_token=HF_TOKEN,
)

chat_model = ChatHuggingFace(llm=llm_engine)

# TOOLS
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"The weather in {city} is sunny with 25°C."

tools = [multiply, get_weather]
tool_node = ToolNode(tools)

# STATE
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    plan: List[str]
    current_step: int

# NODES
def planner_node(state: AgentState):
    """
    Create a strict plan.
    If request is invalid, detect it explicitly.
    """
    user_input = state["messages"][-1].content

    prompt = (
        "You are a strict planner.\n"
        "If the user request is invalid, ambiguous, or impossible, "
        "respond with exactly: INVALID_REQUEST.\n\n"
        "Otherwise, return a numbered step-by-step plan.\n\n"
        f"User request: {user_input}"
    )

    response = chat_model.invoke(prompt)

    if "INVALID_REQUEST" in response.content:
        return {
            "plan": ["Explain clearly why the request is invalid."],
            "current_step": 0,
        }

    plan = [s.strip() for s in response.content.split("\n") if s.strip()]
    return {"plan": plan, "current_step": 0}


def executor_node(state: AgentState):
    """
    Execute ONE step safely.
    Tools are guarded.
    """
    idx = state["current_step"]
    plan = state["plan"]

    if idx >= len(plan):
        return {}

    step = plan[idx]

    # Block math tools if no numbers exist
    if "multiply" in step.lower() and not any(ch.isdigit() for ch in step):
        return {
            "messages": [
                SystemMessage(
                    content="I cannot perform multiplication because no valid numbers were provided."
                )
            ],
            "current_step": len(plan),
        }

    system_msg = SystemMessage(
        content=f"Execute the following step carefully:\n{step}"
    )

    llm_with_tools = chat_model.bind_tools(tools)

    response = llm_with_tools.invoke(
        [system_msg] + state["messages"]
    )

    return {
        "messages": [response],
        "current_step": idx + 1,
    }


def final_answer_node(state: AgentState):
    """Produce the final answer for the user."""
    response = chat_model.invoke(
        state["messages"]
        + [SystemMessage(content="Provide a final clear answer to the user.")]
    )
    return {"messages": [response]}

# ROUTERS
def executor_router(state: AgentState) -> Literal["tools", "check_done"]:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "check_done"


def check_done(state: AgentState) -> Literal["execute", "final"]:
    if state["current_step"] < len(state["plan"]):
        return "execute"
    return "final"

# GRAPH
graph = StateGraph(AgentState)

graph.add_node("planner", planner_node)
graph.add_node("executor", executor_node)
graph.add_node("tools", tool_node)
graph.add_node("final", final_answer_node)
graph.add_node("check_done", lambda x: x)

graph.add_edge(START, "planner")
graph.add_edge("planner", "executor")

graph.add_conditional_edges(
    "executor",
    executor_router,
    {
        "tools": "tools",
        "check_done": "check_done",
    },
)

graph.add_conditional_edges(
    "check_done",
    check_done,
    {
        "execute": "executor",
        "final": "final",
    },
)

graph.add_edge("tools", "executor")
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
    return {"status": "Planner-Executor LangGraph Agent running"}
