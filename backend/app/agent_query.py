from agents import Agent, Runner
from app.retrieval import retrieve_chunks
from app.db import save_chat
from dotenv import load_dotenv

load_dotenv()



agent = Agent(
    name="book-rag-agent",
    instructions=(
        "Answer the question using ONLY the provided book content. "
        "If the answer is not in the content, say you don't know."
    ),
)

def run_agent(query: str, selected_text: str = None) -> str:
    # 1. Always retrieve from Qdrant
    chunks = retrieve_chunks(query, top_k=5)
    
    context = "\n\n".join(
        f"[Source: {c.path}]\n{c.text}" for c in chunks
    )
    
    # 2. Selected text is OPTIONAL bias
    if selected_text:
        prompt = f"""
        Selected text (may be incomplete or noisy):
        {selected_text}

        Book content:
        {context}

        Question:
        {query}

        Answer ONLY using the book content above.
        """
    else:
        prompt = f"""
        Book content:
        {context}

        Question:
        {query}

        Answer ONLY using the book content above.
        """
    
    result = Runner.run_sync(
        agent,
        input=prompt,
    )

    answer = result.final_output
    save_chat(query, answer)

    return answer