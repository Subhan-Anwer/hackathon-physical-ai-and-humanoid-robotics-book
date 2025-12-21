# RAG Chatbot Backend - Detailed Specification

## Project Overview

**Project**: Physical AI & Humanoid Robotics Book – RAG Chatbot Backend
**Goal**: Build a FastAPI backend for a Retrieval-Augmented Generation (RAG) chatbot embedded into a Docusaurus-based online book.

## Functional Requirements

### 1. REST API using FastAPI
- The backend MUST expose a REST API using FastAPI
- API endpoints MUST follow REST conventions
- All endpoints MUST return appropriate HTTP status codes
- All endpoints MUST include proper request/response validation

### 2. RAG Chatbot Functionality
The chatbot MUST answer questions using ONLY:
- Retrieved content from Qdrant (default mode), OR
- User-selected text (selection mode)

### 3. Integration Requirements
The backend MUST integrate:
- Qdrant Cloud (vector search) - READ-ONLY access to pre-existing collection
- Neon Serverless Postgres (chat sessions & metadata)
- Embedding model compatible with existing ingestion
- LLM Agent abstraction (OpenAI Agents SDK compatible)

**IMPORTANT**: The book content has ALREADY been ingested into Qdrant Cloud using a separate offline ingestion script (ingest_book.py). The backend MUST NOT re-ingest content, recreate the Qdrant collection, or generate embeddings for book pages at runtime.

### 4. Query Modes
The backend MUST support two query modes:

#### A. Global Book RAG
- Embed user query using the same embedding model as the ingestion process
- Perform vector search against the existing Qdrant collection (NO ingestion at runtime)
- Retrieve top-k chunks from the pre-existing collection
- Generate answer grounded in retrieved chunks only
- Return relevant chunk IDs, URLs, and sources in response

#### B. Selected Text RAG
- Accept user-selected text
- Disable vector search
- Generate answer using ONLY selected text
- No chunk retrieval required

### 5. Data Storage Requirements
The backend MUST store in Neon Postgres:
- Chat session ID
- User question
- Retrieved chunk IDs (if any)
- LLM response
- Timestamp of interaction
- Query mode used (Global Book RAG vs Selected Text RAG)

## Non-Functional Requirements

### Performance
- API response time should be under 5 seconds for 95% of requests
- Support for at least 100 concurrent users
- Handle vector search requests efficiently

### Security
- Environment variables for all secrets
- Rate-limit safe design
- Input validation to prevent injection attacks

### Maintainability
- Clear folder structure following suggested organization
- Easy to swap LLM provider (Groq → OpenAI)
- Proper logging and error handling

## API Specification

### Chat Endpoints

#### POST /api/chat
**Purpose**: Main endpoint for chat interactions with RAG functionality

**Request Body**:
```json
{
  "session_id": "string",
  "message": "string",
  "query_mode": "global_book_rag" | "selected_text_rag",
  "selected_text": "string" | null,
  "top_k": "integer" | null
}
```

**Response**:
```json
{
  "response": "string",
  "session_id": "string",
  "query_mode": "global_book_rag" | "selected_text_rag",
  "retrieved_chunks": [
    {
      "chunk_id": "string",
      "content": "string",
      "score": "number"
    }
  ] | null,
  "timestamp": "string"
}
```

**Status Codes**:
- 200: Success
- 400: Invalid request parameters
- 500: Internal server error

#### POST /api/chat/session
**Purpose**: Create a new chat session

**Request Body**:
```json
{
  "session_id": "string" | null
}
```

**Response**:
```json
{
  "session_id": "string",
  "created_at": "string"
}
```

#### GET /api/chat/session/{session_id}
**Purpose**: Get chat history for a session

**Response**:
```json
{
  "session_id": "string",
  "history": [
    {
      "question": "string",
      "response": "string",
      "timestamp": "string",
      "query_mode": "string",
      "retrieved_chunks": ["string"] | null
    }
  ]
}
```

### Health Check Endpoint

#### GET /health
**Purpose**: Health check endpoint

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "string"
}
```

## Database Schema

### chat_sessions Table
```sql
CREATE TABLE chat_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### chat_messages Table
```sql
CREATE TABLE chat_messages (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) REFERENCES chat_sessions(session_id),
    question TEXT NOT NULL,
    response TEXT NOT NULL,
    query_mode VARCHAR(50) NOT NULL,
    retrieved_chunk_ids TEXT[],
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Technical Architecture

### Folder Structure
```
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
│   │   └── logging.py
├── requirements.txt
└── README.md
```

### Core Components

#### 1. Configuration (core/config.py)
- Environment variable loading
- Database connection settings
- Qdrant connection settings
- LLM provider configuration
- Embedding model settings (MUST match the model used in the ingestion process)

#### 2. Database (core/database.py)
- PostgreSQL connection handling
- Session management
- Chat history persistence

#### 3. API Endpoints (api/)
- Chat endpoints with request/response validation
- Health check endpoints
- Error handling middleware

#### 4. Services (services/)
- **Retriever**: Performs vector search against existing Qdrant collection ONLY - NO ingestion or collection creation
- **Agent**: LLM interaction abstraction
- **Embeddings**: Query text embedding generation (for search, NOT for ingestion)

#### 5. Models (models/schemas.py)
- Pydantic models for request/response validation
- Database model definitions
- API response structures

#### 6. Utilities (utils/logging.py)
- Application logging
- Error tracking
- Performance monitoring

## Implementation Constraints

### Qdrant Runtime Requirements (CRITICAL)
**IMPORTANT**: The book content has ALREADY been ingested into a Qdrant Cloud cluster using a separate offline ingestion script (ingest_book.py).

The backend MUST NOT at runtime:
- Re-ingest content into Qdrant
- Recreate the Qdrant collection
- Generate embeddings for book pages
- Modify the existing Qdrant collection in any way

At runtime, the backend should ONLY:
- Embed user queries using the same embedding model as the ingestion process
- Perform vector search against the existing Qdrant collection
- Use stored payload fields (url, text, chunk_id, source) from Qdrant search results

### Out of Scope (For Now)
- Authentication
- Streaming responses
- Frontend widget

### Security Considerations
- Input sanitization for all user-provided content
- Rate limiting to prevent abuse
- Environment variables for all sensitive data
- SQL injection prevention through parameterized queries

## Error Handling

### Expected Error Scenarios
1. Database connection failures
2. Qdrant vector search failures
3. LLM provider unavailability
4. Invalid user inputs
5. Rate limiting exceeded

### Error Response Format
```json
{
  "error": {
    "type": "string",
    "message": "string",
    "details": "object" | null
  }
}
```

## Testing Requirements

### Unit Tests
- Service layer functionality (retriever, agent, embeddings)
- API endpoint validation
- Database operations
- Configuration loading

### Integration Tests
- Full API request/response cycle
- Database persistence
- Qdrant integration
- LLM provider integration

### Performance Tests
- API response time under load
- Vector search performance
- Database query performance

## Deployment Requirements

### Environment Variables
- `DATABASE_URL`: Neon Postgres connection string
- `QDRANT_URL`: Qdrant Cloud endpoint
- `QDRANT_API_KEY`: Qdrant authentication key
- `LLM_PROVIDER_API_KEY`: API key for LLM provider
- `LLM_PROVIDER_BASE_URL`: Base URL for LLM provider
- `EMBEDDING_MODEL`: Name of embedding model to use
- `TOP_K_DEFAULT`: Default number of chunks to retrieve

### Infrastructure
- FastAPI application server
- Neon Serverless Postgres database
- Qdrant Cloud vector database
- LLM provider access (OpenAI, Groq, etc.)

## Acceptance Criteria

### Core Functionality
- [ ] API responds to chat requests with appropriate responses
- [ ] Global Book RAG mode retrieves relevant chunks from Qdrant
- [ ] Selected Text RAG mode generates responses using only provided text
- [ ] Chat history is properly stored in database
- [ ] Both query modes work as specified

### Integration
- [ ] Successfully connects to Qdrant Cloud
- [ ] Successfully connects to Neon Postgres
- [ ] Properly handles embedding generation
- [ ] Integrates with LLM provider

### Error Handling
- [ ] Gracefully handles database connection failures
- [ ] Properly validates input parameters
- [ ] Returns appropriate error messages
- [ ] Handles rate limiting

### Performance
- [ ] API responds within 5 seconds for 95% of requests
- [ ] Supports expected concurrent user load
- [ ] Efficient vector search operations