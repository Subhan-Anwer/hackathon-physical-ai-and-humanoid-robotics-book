from pydantic import BaseModel
from typing import List, Optional

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5

class RetrievedChunk(BaseModel):
    text: str
    path: str
    score: float

class RetrieveResponse(BaseModel):
    results: List[RetrievedChunk]
    
class AgentQueryRequest(BaseModel):
    query: str
    selected_text: Optional[str] = None