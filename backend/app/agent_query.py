from agents import Agent, Runner, function_tool
from app.retrieval import retrieve_chunks
from app.db import save_chat
from dotenv import load_dotenv

load_dotenv()

@function_tool
def retrieval_tool(query: str):
    return retrieve_chunks(query, top_k = 5)




agent = Agent(
    name="book-rag-agent",
    instructions=(
        "You MUST call the retrieval_tool to answer. "
        "Use ONLY retrieved book chunks. "
        "Do not answer from prior knowledge."
    ),
    tools=[retrieval_tool],
)

def run_agent(query: str) -> str:
    result = Runner.run_sync(
        agent,
        input=query,
    )

    answer = result.final_output
    save_chat(query, answer)

    return answer