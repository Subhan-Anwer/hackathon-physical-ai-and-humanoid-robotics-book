PROJECT: Physical AI & Humanoid Robotics Book – RAG Chatbot Backend

GOAL:
Build a FastAPI backend for a Retrieval-Augmented Generation (RAG) chatbot
embedded into a Docusaurus-based online book.

FUNCTIONAL REQUIREMENTS:

1. The backend MUST expose a REST API using FastAPI.

2. The chatbot MUST answer questions using ONLY:
   - Retrieved content from Qdrant (default mode), OR
   - User-selected text (selection mode)

3. The backend MUST integrate:
   - Qdrant Cloud (vector search)
   - Neon Serverless Postgres (chat sessions & metadata)
   - Embedding model compatible with existing ingestion
   - LLM Agent abstraction (OpenAI Agents SDK compatible)

4. The backend MUST support two query modes:
   A. Global Book RAG
      - Embed user query
      - Retrieve top-k chunks from Qdrant
      - Generate answer grounded in retrieved chunks only

   B. Selected Text RAG
      - Accept user-selected text
      - Disable vector search
      - Generate answer using ONLY selected text

5. The backend MUST store:
   - Chat session ID
   - User question
   - Retrieved chunk IDs (if any)
   - LLM response
   in Neon Postgres.

NON-FUNCTIONAL REQUIREMENTS:

- Clear folder structure
- Environment variables for all secrets
- Rate-limit safe design
- Easy to swap LLM provider (Groq → OpenAI)

SUGGESTED FOLDER STRUCTURE:

backend/
├── app/
│   ├── main.py
│   ├── api/
│   │   ├── chat.py
│   │   └── health.py
│   ├── core/
│   │   ├── config.py
│   │   └── database.py
│   ├── services/
│   │   ├── retriever.py
│   │   ├── agent.py
│   │   └── embeddings.py
│   ├── models/
│   │   └── schemas.py
│   └── utils/
│       └── logging.py
├── requirements.txt
└── README.md

OUT OF SCOPE (FOR NOW):
- Authentication
- Streaming responses
- Frontend widget
