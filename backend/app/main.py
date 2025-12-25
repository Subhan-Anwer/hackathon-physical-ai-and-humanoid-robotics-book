from fastapi import FastAPI
from .schemas import RetrieveRequest, RetrieveResponse
from .retrieval import retrieve_chunks

app = FastAPI(title="Book RAG Retrieval API")

@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    results = retrieve_chunks(req.query, req.top_k)
    return {"results": results}
