# RAG Backend API Specification

## Project Overview

A Retrieval-Augmented Generation (RAG) backend API for a Docusaurus-based technical book on Physical AI & Humanoid Robotics. The backend will be built with FastAPI (Python), using Cohere as the LLM provider, Qdrant Cloud as the vector database, and Neon Serverless Postgres for persistence.

## Functional Requirements

### Core Features
- Expose a POST `/chat` endpoint for conversational AI interactions
- Input: user question as a string
- Output: grounded answer based ONLY on book content
- Use Retrieval-Augmented Generation (RAG) to ensure answers are based on book content
- No citations required in output (clean, natural responses)
- Support future extension for:
  - "answer based on selected text" functionality
  - User personalization features

### API Contract

#### POST `/chat`
**Description**: Process user questions and return grounded responses based on book content

**Request Body**:
```json
{
  "question": "string",
  "context": {
    "selected_text": "string (optional)",
    "user_id": "string (optional)"
  }
}
```

**Response**:
```json
{
  "answer": "string",
  "sources": ["string"] (optional, for future use),
  "model_used": "string",
  "timestamp": "ISO 8601 datetime string"
}
```

**Success Response (200)**:
- Returns a well-formed answer based on book content
- Answer is grounded in retrieved documents
- No hallucinations or external information

**Error Responses**:
- 400: Invalid request format
- 422: Unprocessable entity (e.g., empty question)
- 500: Internal server error
- 503: Service unavailable (external API issues)

## Data Flow Description

### High-Level Flow
1. User submits a question via POST `/chat`
2. Input validation occurs
3. Question is embedded using Cohere's embedding model
4. Vector search performed in Qdrant Cloud to retrieve relevant book content
5. Retrieved context and original question sent to Cohere's generative model
6. Generated response returned to user

### Detailed Flow
1. **Request Reception**: FastAPI receives POST request with user question
2. **Validation**: Validate input format and content
3. **Embedding Generation**: Use Cohere API to generate embedding for user question
4. **Vector Search**: Query Qdrant Cloud with the embedding to retrieve top-k relevant documents
5. **Context Formation**: Combine retrieved documents into a context prompt
6. **Response Generation**: Use Cohere's generative model to create answer based on context and question
7. **Response Formatting**: Structure response according to API contract
8. **Response Delivery**: Return formatted response to client

## RAG Pipeline Steps

### Step 1: Document Ingestion & Indexing
- Extract content from Docusaurus documentation
- Chunk documents into manageable pieces (e.g., paragraphs, sections)
- Generate embeddings for each chunk using Cohere
- Store embeddings in Qdrant Cloud with metadata

### Step 2: Query Processing
1. **Input Validation**: Ensure question is non-empty and properly formatted
2. **Question Embedding**: Convert user question to vector representation using Cohere
3. **Similarity Search**: Find most relevant document chunks in vector database
4. **Context Assembly**: Combine top-k retrieved chunks into context for generation

### Step 3: Answer Generation
1. **Prompt Construction**: Create a prompt containing:
   - Retrieved context documents
   - Original user question
   - Instructions to ground response in provided context
2. **Model Inference**: Send prompt to Cohere's generative model
3. **Response Processing**: Clean and format the model's output

### Step 4: Response Delivery
1. **Validation**: Ensure response is grounded in provided context
2. **Formatting**: Structure response according to API contract
3. **Delivery**: Return response to client

## Environment Variables

### Required Variables
```env
# Cohere Configuration
COHERE_API_KEY="your-cohere-api-key"
COHERE_EMBED_MODEL="embed-multilingual-v2.0"  # or appropriate model
COHERE_GENERATE_MODEL="command-r-plus"  # or appropriate model

# Qdrant Cloud Configuration
QDRANT_HOST="your-qdrant-cluster-url"
QDRANT_API_KEY="your-qdrant-api-key"
QDRANT_COLLECTION_NAME="book_content"

# Database Configuration
NEON_DATABASE_URL="your-neon-db-connection-string"

# Application Configuration
APP_ENV="development|production"
LOG_LEVEL="debug|info|warning|error"
```

### Optional Variables
```env
# API Configuration
QDRANT_TOP_K=5  # Number of documents to retrieve
COHERE_TEMPERATURE=0.3  # Generation temperature (lower for more deterministic)
COHERE_MAX_TOKENS=1000  # Maximum tokens in response
MAX_QUESTION_LENGTH=1000  # Maximum allowed question length
```

## Error Handling Expectations

### Input Validation Errors
- **Empty Question**: Return 422 with message "Question cannot be empty"
- **Invalid JSON**: Return 400 with appropriate error message
- **Question Too Long**: Return 422 with message about length limits

### External Service Errors
- **Cohere API Unavailable**: Return 503 with message "LLM service temporarily unavailable"
- **Qdrant Unavailable**: Return 503 with message "Search service temporarily unavailable"
- **Database Unavailable**: Return 503 with message "Database service temporarily unavailable"

### Internal Errors
- **Processing Error**: Return 500 with generic error message
- **Timeout**: Return 500 with message "Request timed out"

### Graceful Degradation
- If vector search fails, attempt to return a general response
- If LLM generation fails, return error with suggestion to try again
- Implement circuit breaker pattern for external service calls

## Non-Functional Requirements

### Performance
- API response time: < 3 seconds for typical queries
- Support for concurrent users: at least 10 simultaneous requests
- Embedding generation and search should complete within 1.5 seconds

### Reliability
- 99% uptime during hackathon period
- Proper retry logic for external API calls
- Circuit breaker pattern for external services

### Maintainability
- Simple, readable code without complex frameworks
- Deterministic and debuggable components
- Clear logging for troubleshooting
- Well-documented code and architecture

### Scalability
- Designed for hackathon scale (hundreds of users)
- Easy to extend with additional features
- Modular architecture allowing component replacement

## Future Extensibility

### Selected Text Feature
- Accept optional `selected_text` parameter in context
- Modify RAG pipeline to prioritize provided text in generation
- Adjust prompt engineering to incorporate selected text

### User Personalization
- Accept optional `user_id` parameter
- Store user preferences in Neon Postgres
- Track conversation history per user
- Personalize responses based on user context

## Security Considerations

### Input Sanitization
- Validate and sanitize all user inputs
- Prevent prompt injection attacks
- Implement rate limiting to prevent abuse

### API Security
- Secure API keys with environment variables
- Implement authentication if needed for future features
- Log all requests for audit purposes