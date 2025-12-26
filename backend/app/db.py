import os
import psycopg

DATABASE_URL = os.getenv("DATABASE_URL")

def save_chat(query: str, answer: str):
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_logs (query, answer)
                VALUES (%s, %s)
                """,
                (query, answer),
            )
        conn.commit()