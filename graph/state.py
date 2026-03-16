
from __future__ import annotations
from typing import Annotated, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class QualityScore(TypedDict):
    correctness: float   
    security:    float   
    style:       float  
    overall:     float   
    feedback:    str    
    review:      str     


class NexusState(TypedDict):
    #  conversation 
    messages: Annotated[List[BaseMessage], add_messages]

    #  orchestrator decision 
    task_type: str  
    language:  str   
    plan:      str   

    #  code artefacts 
    generated_code:   str
    execution_result: str
    execution_ok:     bool
    exec_retries:     int   

    #  quality 
    quality_score:  Optional[QualityScore]
    revision_count: int  

    #  memory 
    memory_context: str
