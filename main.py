import os
from typing import List, TypedDict, Annotated

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_huggingface import HuggingFaceEndpoint

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")

app = FastAPI(title="LangGraph HuggingFace Chatbot API")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.7,
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=512
)

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def chat_node(state: ChatState) -> ChatState:
    response = llm.invoke(state["messages"])
    return {"messages": [AIMessage(content=response)]}

graph = StateGraph(ChatState)

graph.add_node("chat", chat_node)
graph.add_edge(START, "chat")
graph.add_edge("chat", "chat")

app_graph = graph.compile()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

chat_state: ChatState = {"messages": []}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):

    global chat_state

    chat_state["messages"].append(HumanMessage(content=req.message))

    chat_state = app_graph.invoke(chat_state)

    ai_msg = chat_state["messages"][-1]

    return {"response": ai_msg.content}

@app.get("/")
def root():
    return {"status": "LangGraph + HuggingFace Chatbot API Running"}
