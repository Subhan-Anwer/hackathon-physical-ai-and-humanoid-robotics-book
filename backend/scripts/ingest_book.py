"""
One-time script to ingest book markdown files into Qdrant vector database.
This script reads markdown files from the site/docs directory, chunks the text,
and stores the chunks in Qdrant with metadata.
"""

import os
import uuid
from pathlib import Path
from typing import List
from dotenv import load_dotenv
import time

import cohere
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
import markdown
from bs4 import BeautifulSoup

load_dotenv()



# -------------------------
# Helpers
# -------------------------


def read_markdown_file(file_path: Path) -> str:
    """Read and extract text content from a markdown file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        md = f.read()
    html = markdown.markdown(md)
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text(separator=" ", strip=True)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
    
    chunks = []
    text_length = len(text)
    start = 0

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        
        if chunk:
            chunks.append(chunk)
            
        # move forward safely
        start += chunk_size - overlap

            
    return chunks


# -------------------------
# Main
# -------------------------


def main():
    """Main function to ingest book markdown files into Qdrant."""
    
    # ---- env ----
    cohere_api_key = os.getenv("COHERE_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection_name = os.getenv("QDRANT_COLLECTION_NAME")
    
    
    if not all([cohere_api_key, qdrant_url, qdrant_api_key]):
        raise RuntimeError("COHERE_API_KEY, QDRANT_URL, QDRANT_API_KEY are required")


    # ---- clients ----
    co = cohere.Client(cohere_api_key)

    qdrant = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
    )
    
    # ---- recreate collection cleanly ----
    if qdrant.collection_exists(collection_name):
        print(f"Collection '{collection_name}' exists, clearing it...")
        qdrant.delete_collection(collection_name)

    print(f"Creating collection '{collection_name}'...")
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=1024,
            distance=Distance.COSINE,
        ),
    )

    # ---- locate book ----
    book_dir = Path("../site/docs")  # Relative to backend directory
    if not book_dir.exists():
        # Try alternative path if the above doesn't exist
        raise RuntimeError("Book directory not found: ../site/docs")

    markdown_files = list(book_dir.rglob("*.md"))
    print(f"Found {len(markdown_files)} markdown files")

    # points = []
    # chunk_index = 0
    texts = []
    metadatas = []


    # ---- read + chunk ----
    for md_file  in markdown_files:
        print(f"Processing: {md_file}")
        text = read_markdown_file(md_file)
        chunks = chunk_text(text)
        
        for idx, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append({
                "source": "book",
                "path": str(md_file.relative_to(book_dir)),
                "chunk_index": idx,
                "text": chunk,
            })
            
            
    print(f"Total chunks: {len(texts)}")
    
    
    # ---- batch embed ----
    
    BATCH_SIZE = 32
    
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        batch_meta = metadatas[i:i + BATCH_SIZE]
        
        embed = co.embed(
            texts=batch_texts,
            model="embed-english-v3.0",
            input_type="search_document",
        ).embeddings
        
        time.sleep(2)  # To respect rate limits
        
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=meta,
            )
            for vector, meta in zip(embed, batch_meta)
        ]
        
        qdrant.upsert(collection_name=collection_name, points=points)
        print(f"Ingested {i + len(points)} / {len(texts)}")
        
    print("Ingestion completed successfully.")
    
    
# -------------------------
if __name__ == "__main__":
    main()