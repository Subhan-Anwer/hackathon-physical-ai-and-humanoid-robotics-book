from fastapi import FastAPI
from .schemas import RetrieveRequest, RetrieveResponse, AgentQueryRequest
from .retrieval import retrieve_chunks
from app.agent_query import run_agent



app = FastAPI(title="Book RAG Retrieval API")

@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    results = retrieve_chunks(req.query, req.top_k)
    return {"results": results}

@app.post("/agent_query")
def agent_query(req: AgentQueryRequest):
    results = run_agent(
        req.query,
        selected_text=req.selected_text,
    )
    return {"results": results}