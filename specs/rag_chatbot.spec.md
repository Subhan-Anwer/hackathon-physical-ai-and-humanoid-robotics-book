# RAG Chatbot Specification

## 1. System Overview

Goal: Build a backend Retrieval-Augmented Generation (RAG) chatbot
that answers questions about a published technical book.

Core Capabilities:
- Answer questions using the entire book (vector search)
- Answer questions using only user-selected text
- Persist chat sessions and messages
- Expose a clean HTTP API for frontend embedding

---

## 2. Runtime Environment

- OS: Ubuntu (WSL-compatible)
- Language: Python 3.11+
- Framework: FastAPI
- Python environment: uv with local .venv
- Dependency management: requirements.txt installed via uv
- Deployment target: Free-tier PaaS

---

## 3. External Services (MANDATORY)

- Cohere: embeddings + text generation
- Qdrant Cloud (Free Tier): vector search
- Neon Serverless Postgres: chat/session storage

Only Cohere may be used for LLM calls.

## 3A. Agent Orchestration (MANDATORY)

- The system must use the OpenAI Agents SDK for agent orchestration.
- The Agents SDK is used only for:
  - tool orchestration
  - control flow
  - decision logic
- No OpenAI-hosted LLM models are used.
- All embeddings and text generation are performed using Cohere.
- No OpenAI LLM model usage

---

## 4. Data Flow

### Whole-Book Question Flow

1. Receive user question
2. Embed question using Cohere
3. Query Qdrant for relevant chunks
4. Construct prompt with retrieved chunks
5. Generate answer using Cohere
6. Persist chat session and messages
7. Return answer

---

### Selected-Text Question Flow

1. Receive user question and selected_text
2. Skip vector search
3. Construct prompt using ONLY selected_text
4. Generate answer using Cohere
5. Persist chat session and messages
6. Return answer

---

## 5. Vector Storage

Collection name: book_chunks

Payload metadata:
- source
- path
- chunk_index
- text

---

## 6. Database Schema

### chat_sessions
- id (UUID)
- created_at (TIMESTAMP)

### chat_messages
- id (UUID)
- session_id (UUID)
- role (user | assistant)
- content (TEXT)
- created_at (TIMESTAMP)

---

## 7. API Contract

POST /chat

Request:
- session_id (UUID or null)
- question (string)
- selected_text (string or null)

Behavior:
- selected_text present → constrained mode
- selected_text null → retrieval mode

Response:
- session_id
- answer

---

## 8. One-Time Scripts

- ingest_book.py
- create_tables.py

---

## 9. Non-Goals

- No authentication
- No streaming
- No multi-LLM routing
- No frontend logic
