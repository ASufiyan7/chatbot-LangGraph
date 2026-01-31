import os
from typing import List, TypedDict, Annotated, Literal

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_huggingface import HuggingFaceEndpoint
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
if not HF_TOKEN:
    raise ValueError("HUGGINGFACE_API_KEY not found in environment variables.")

app = FastAPI(title="LangGraph HuggingFace Chatbot API")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.7,
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=512,
)

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def chat_node(state: ChatState) -> ChatState:
    print("Generating AI response...")
    ai_response = llm.invoke(state["messages"])
    return {"messages": [AIMessage(content=ai_response)]}

def goodbye_node(state: ChatState) -> ChatState:
    print("Ending conversation.")
    return {"messages": [AIMessage(content="Goodbye!")]}

def router(state: ChatState) -> Literal["end_chat", "continue_chat"]:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_message = msg.content.lower()
            break

    else:
        return "continue_chat"

    if any(words in last_message for words in ["bye", "exit", "quit", "goodbye"]):
        return "end_chat"
    
    return "continue_chat"

graph = StateGraph(ChatState)

graph.add_node("chat", chat_node)
graph.add_node("goodbye", goodbye_node)

graph.add_edge(START, "chat")

graph.add_conditional_edges(
    "chat",
    router,
    {
        "end_chat": "goodbye",
        "continue_chat": END
    }
)

graph.add_edge("goodbye", END)

class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default_user"

class ChatResponse(BaseModel):
    response: str

memory = InMemorySaver()
app_graph = graph.compile(checkpointer=memory)

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):

    config = {"configurable": {"thread_id": req.thread_id}}

    system_prompt = SystemMessage(content="You are a helpful AI assistant. Be concise, accurate, and clear.")

    output = app_graph.invoke(
        {
            "messages": [
                system_prompt,
                HumanMessage(content=req.message)
            ]
        },
        config=config
    )

    return {'response': output["messages"][-1].content}

@app.get("/")
def root():
    return {"status": "LangGraph + HuggingFace Chatbot API Running"}
