"""Ingestion : data/sites.json -> tables sites + chunks (embeddings).

Usage : python -m app.ingest      (lance aussi automatiquement au 1er démarrage)
"""
import json
from pathlib import Path

from psycopg.types.json import Jsonb

from .db import connect, init_db
from .embeddings import embed_passages

DATA = Path(__file__).resolve().parent.parent / "data" / "sites.json"
LANGS = ("fr", "en", "ar")


def ingest(path: Path = DATA) -> int:
    sites = json.loads(path.read_text(encoding="utf-8"))
    texts, meta = [], []
    for s in sites:
        for lang in LANGS:
            # un chunk par site et par langue : nom + description + conseil
            texts.append(f"{s['name'][lang]}. {s['d'][lang]} {s['tip'][lang]}")
            meta.append((s["id"], lang))
    vectors = embed_passages(texts)

    with connect() as conn, conn.transaction():
        conn.execute("DELETE FROM chunks")
        for s in sites:
            conn.execute(
                """INSERT INTO sites (id, cat, zone, lat, lon, data) VALUES (%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (id) DO UPDATE SET cat=EXCLUDED.cat, zone=EXCLUDED.zone,
                   lat=EXCLUDED.lat, lon=EXCLUDED.lon, data=EXCLUDED.data""",
                (s["id"], s["cat"], s["zone"], s["lat"], s["lon"], Jsonb(s)),
            )
        for (sid, lang), text, vec in zip(meta, texts, vectors):
            conn.execute(
                "INSERT INTO chunks (site_id, lang, text, embedding) VALUES (%s,%s,%s,%s)",
                (sid, lang, text, vec),
            )
    return len(texts)


if __name__ == "__main__":
    init_db()
    print(f"{ingest()} chunks indexés")
