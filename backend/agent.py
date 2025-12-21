from agents import Agent, OpenAIChatCompletionsModel, AsyncOpenAI, Runner
from agents import set_tracing_disabled, function_tool, enable_verbose_stdout_logging
from dotenv import load_dotenv
import os
import cohere
from qdrant_client import QdrantClient

load_dotenv()
set_tracing_disabled(True)
enable_verbose_stdout_logging()

# Groq LLM
# client = AsyncOpenAI(
#     api_key=os.getenv("GROQ_API_KEY"),
#     base_url=os.getenv("GROQ_BASE_URL")
# )

# model = OpenAIChatCompletionsModel(
#     model="llama-3.3-70b-versatile",
#     openai_client=client,
# )

# OpenAI LLM
client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

model = OpenAIChatCompletionsModel(
    model="gpt-4o-mini",  # cheap + correct
    openai_client=client,
)

# Cohere + Qdrant
cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)


def get_embedding(text):
    response = cohere_client.embed(
        model="embed-english-v3.0",
        input_type="search_query",
        texts=[text],
    )
    return response.embeddings[0]


@function_tool
def retrieve(query: str):
    embedding = get_embedding(query)

    hits = qdrant.search(
        collection_name=os.getenv("COLLECTION_NAME"),
        query_vector=embedding,
        limit=5
    )
    
    return [hit.payload["text"] for hit in hits]

agent = Agent(
    name="Assistant",
    instructions="""
    You are an AI tutor for the Physical AI & Humanoid Robotics textbook.

Rules:
1. Always call `retrieve` first.
2. Use ONLY retrieved content to answer.
3. If the answer is not present, say exactly: "I don't know".

    """,
    model=model,
    tools=[retrieve],
)

res = Runner.run_sync(agent, input="how many modules are there in the book?")
print(res.final_output)