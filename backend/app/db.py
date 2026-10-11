import os

import psycopg
from pgvector.psycopg import register_vector

DSN = os.getenv("DATABASE_URL")
LITE = not DSN  # sans DATABASE_URL : mode léger en mémoire (voir lite.py)
DIM = int(os.getenv("EMBED_DIM", "384"))  # paraphrase-multilingual-MiniLM-L12-v2 -> 384

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS sites (
    id   text PRIMARY KEY,
    cat  text NOT NULL,
    zone text NOT NULL,
    lat  double precision,
    lon  double precision,
    data jsonb NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id        serial PRIMARY KEY,
    site_id   text NOT NULL REFERENCES sites(id) ON DELETE CASCADE,
    lang      text NOT NULL,
    text      text NOT NULL,
    embedding vector({DIM}) NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_emb_idx ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS chat_logs (
    id         serial PRIMARY KEY,
    ts         timestamptz NOT NULL DEFAULT now(),
    lang       text,
    question   text,
    answer     text,
    sources    text[],
    ctx_site   text,
    mode       text,
    latency_ms int
);

CREATE TABLE IF NOT EXISTS feedback (
    id      serial PRIMARY KEY,
    ts      timestamptz NOT NULL DEFAULT now(),
    chat_id int REFERENCES chat_logs(id) ON DELETE SET NULL,
    value   int NOT NULL
);
"""


def init_db() -> None:
    # connexion « nue » : l'extension doit exister avant register_vector()
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute(SCHEMA)


def connect() -> psycopg.Connection:
    conn = psycopg.connect(DSN, autocommit=True)
    register_vector(conn)
    return conn
