import requests
import xml.etree.ElementTree as ET
import trafilatura
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import cohere
import dotenv
import os
import time

dotenv.load_dotenv()

# -----------------------------------
# CONFIG
# -----------------------------------
SITEMAP_URL = os.getenv("SITEMAP_URL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
EMBED_MODEL = "embed-english-v3.0"

# Connect to Qdrant Cloud
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# -----------------------------------
# URL FILTERING (CRITICAL)
# -----------------------------------
def should_skip_url(url: str) -> bool:
    """
    Skip non-content pages that break RAG quality
    """
    skip_patterns = [
        "/docs/category/",
        "/blog",
        "/tags",
        "/authors",
        "/archive",
        "/page/",
    ]
    return any(p in url for p in skip_patterns)


# -----------------------------------
# Step 1: Extract URLs from Sitemap
# -----------------------------------
def get_all_urls(sitemap_url):
    print("\nFetching sitemap...")
    xml = requests.get(sitemap_url, timeout=20).text
    root = ET.fromstring(xml)

    urls = []
    for child in root:
        loc_tag = child.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
        if loc_tag is None:
            continue

        url = loc_tag.text.strip()

        if should_skip_url(url):
            print("[SKIP]", url)
            continue

        urls.append(url)

    print(f"\n✅ {len(urls)} valid content pages found")
    return urls


# -----------------------------------
# Step 2: Download page + extract text
# -----------------------------------
def extract_text_from_url(url, max_chars=50_000):
    try:
        html = requests.get(url, timeout=20).text
        text = trafilatura.extract(html)

        if not text:
            print("[WARNING] No text extracted:", url)
            return None

        # HARD safety limit
        if len(text) > max_chars:
            print(f"[TRUNCATE] {url} ({len(text)} chars)")
            text = text[:max_chars]

        return text

    except Exception as e:
        print("[ERROR] Failed to extract:", url, e)
        return None


# -----------------------------------
# Step 3: Chunk text (SAFE)
# -----------------------------------
def chunk_text(text, max_chars=1000):
    """
    Simple fixed-size chunking (memory safe)
    """
    return [
        text[i : i + max_chars]
        for i in range(0, len(text), max_chars)
    ]


# -----------------------------------
# Step 4: Create embedding
# -----------------------------------
def embed(text):
    response = cohere_client.embed(
        model=EMBED_MODEL,
        input_type="search_document",  # ✅ CORRECT for ingestion
        texts=[text]
    )
    
    time.sleep(1.8)
    return response.embeddings[0]


# -----------------------------------
# Step 5: Store in Qdrant
# -----------------------------------
def create_collection():
    print("\nCreating Qdrant collection...")
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=1024,
            distance=Distance.COSINE
        )
    )


def save_chunk_to_qdrant(chunk, chunk_id, url):
    vector = embed(chunk)

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=chunk_id,
                vector=vector,
                payload={
                    "url": url,
                    "text": chunk,
                    "chunk_id": chunk_id,
                    "source": "book"
                }
            )
        ]
    )


# -----------------------------------
# MAIN INGESTION PIPELINE
# -----------------------------------
def ingest_book():
    urls = get_all_urls(SITEMAP_URL)
    create_collection()

    global_id = 1

    for url in urls:
        print("\nProcessing:", url)

        text = extract_text_from_url(url)
        if not text:
            continue

        chunks = chunk_text(text)

        for chunk in chunks:
            save_chunk_to_qdrant(chunk, global_id, url)
            print(f"Saved chunk {global_id}")
            global_id += 1

    print("\n✅ Ingestion completed!")
    print("Total chunks stored:", global_id - 1)


if __name__ == "__main__":
    ingest_book()
