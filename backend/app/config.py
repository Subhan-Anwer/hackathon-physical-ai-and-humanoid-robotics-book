import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import cohere

load_dotenv()


COHERE_API_KEY = os.getenv("COHERE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")

if not all([COHERE_API_KEY, QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME]):
    raise RuntimeError("Missing required environment variables")

cohere_client = cohere.Client(COHERE_API_KEY)


qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)