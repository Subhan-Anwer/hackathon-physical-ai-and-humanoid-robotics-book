from typing import List
from qdrant_client.http.models import SearchParams
from .config import cohere_client, qdrant_client, QDRANT_COLLECTION_NAME
from .schemas import RetrievedChunk


def retrieve_chunks(query: str, top_k: int) -> List[RetrievedChunk]:
    # 1. Embed query
    embedding = cohere_client.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query",
    ).embeddings[0]

    # 2. Search Qdrant
    response = qdrant_client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        query=embedding,
        limit=top_k,
        with_payload=True,
    )

    # 3. Format response
    results = []
    for point in response.points:
        payload = point.payload or {}
        results.append(
            RetrievedChunk(
                text=payload.get("text", ""),
                path=payload.get("path", ""),
                score=point.score,
            )
        )

    return results