import os
import re
from typing import List, TypedDict, Annotated, Literal

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.tools import Tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# ENV SETUP
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
if not HF_TOKEN:
    raise ValueError("HUGGINGFACE_API_KEY not found")

app = FastAPI(title="LangGraph Tool Calling Chatbot")

# LLM
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.1,
    max_new_tokens=512,
    huggingfacehub_api_token=HF_TOKEN,
)

# STATE
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# TOOL
@Tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

# TOOL NODE
def tool_node(state: ChatState) -> ChatState:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_message = msg.content
            break
    else:
        return {"messages": [AIMessage(content="No input found.")]}

    numbers = list(map(int, re.findall(r"\d+", last_message)))

    if len(numbers) >= 2:
        result = multiply(numbers[0], numbers[1])
        return {
            "messages": [
                AIMessage(
                    content=f"The result of multiplying {numbers[0]} and {numbers[1]} is {result}."
                )
            ]
        }

    return {"messages": [AIMessage(content="Please provide two numbers.")]}

# CHAT NODE
def chat_node(state: ChatState) -> ChatState:
    ai_response = llm.invoke(state["messages"])
    return {"messages": [AIMessage(content=ai_response)]}

# GOODBYE NODE
def goodbye_node(state: ChatState) -> ChatState:
    return {"messages": [AIMessage(content="Goodbye!")]}

# TOOL ROUTER
def tool_router(state: ChatState) -> Literal["use_tool", "no_tool"]:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            text = msg.content.lower()
            break
    else:
        return "no_tool"

    if "multiply" in text or "product" in text:
        return "use_tool"

    return "no_tool"

# EXIT ROUTER
def exit_router(state: ChatState) -> Literal["end_chat", "continue"]:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            text = msg.content.lower()
            break
    else:
        return "continue"

    if any(word in text for word in ["bye", "exit", "quit", "goodbye"]):
        return "end_chat"

    return "continue"

# GRAPH
graph = StateGraph(ChatState)

graph.add_node("chat", chat_node)
graph.add_node("tool", tool_node)
graph.add_node("goodbye", goodbye_node)

graph.add_edge(START, "chat")

graph.add_conditional_edges(
    "chat",
    tool_router,
    {
        "use_tool": "tool",
        "no_tool": END,
    },
)

graph.add_conditional_edges(
    END,
    exit_router,
    {
        "end_chat": "goodbye",
        "continue": END,
    },
)

graph.add_edge("tool", END)
graph.add_edge("goodbye", END)

# MEMORY
memory = InMemorySaver()
app_graph = graph.compile(checkpointer=memory)

# API MODELS
class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default_user"

class ChatResponse(BaseModel):
    response: str

# API
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
    return {"status": "LangGraph Tool Chatbot Running"}
