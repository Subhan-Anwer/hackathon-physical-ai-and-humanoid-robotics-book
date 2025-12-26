"""
One-time script to initialize Neon Postgres database.
Creates tables required for the RAG application.
"""

import os
from dotenv import load_dotenv
import psycopg

load_dotenv()


def main():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")

    print("Connecting to Neon Postgres...")

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            print("Creating chat_logs table...")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_logs (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                );
                """
            )

        conn.commit()

    print("Neon database initialized successfully ✅")


if __name__ == "__main__":
    main()
