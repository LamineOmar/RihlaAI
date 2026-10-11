"""Mode léger : sans PostgreSQL ni Docker.

Actif quand DATABASE_URL n'est pas défini. Les embeddings sont gardés en mémoire
(9 fiches = trivial) et les journaux sont écrits dans rihla_logs.jsonl.
"""
import json
import os
import threading
from pathlib import Path

import numpy as np

from .embeddings import embed_passages, embed_query

DATA = Path(__file__).resolve().parent.parent / "data" / "sites.json"
LOG = Path(os.getenv("LITE_LOG", "rihla_logs.jsonl"))
LANGS = ("fr", "en", "ar")

_sites: list[dict] = []
_ids: list[str] = []
_mat: np.ndarray | None = None
_chats: list[dict] = []
_fb: list[dict] = []
_lock = threading.Lock()


def load() -> int:
    global _sites, _ids, _mat
    _sites = json.loads(DATA.read_text(encoding="utf-8"))
    texts, ids = [], []
    for s in _sites:
        for lang in LANGS:
            texts.append(f"{s['name'][lang]}. {s['d'][lang]} {s['tip'][lang]}")
            ids.append(s["id"])
    m = np.stack(embed_passages(texts)).astype(np.float32)
    _mat = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
    _ids = ids
    if LOG.exists():  # restaure l'historique
        for line in LOG.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            (_chats if e.get("t") == "chat" else _fb).append(e)
    return len(texts)


def retrieve(question: str, site_id: str | None, k: int = 8, top_sites: int = 3) -> list[str]:
    q = embed_query(question).astype(np.float32)
    q = q / (np.linalg.norm(q) + 1e-9)
    sims = _mat @ q
    order: list[str] = [site_id] if site_id else []
    for i in np.argsort(-sims)[:k]:
        if _ids[i] not in order:
            order.append(_ids[i])
    return order[:top_sites]


def load_sites(ids: list[str]) -> list[dict]:
    by_id = {s["id"]: s for s in _sites}
    return [by_id[i] for i in ids if i in by_id]


def all_sites() -> list[dict]:
    return list(_sites)


def _append(e: dict) -> None:
    try:
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    except OSError:
        pass


def log_chat(lang, question, answer, sources, ctx_site, mode, latency_ms) -> int:
    with _lock:
        cid = len(_chats) + 1
        e = {"t": "chat", "id": cid, "lang": lang, "question": question, "answer": answer,
             "sources": sources, "ctx_site": ctx_site, "mode": mode, "latency_ms": latency_ms}
        _chats.append(e)
        _append(e)
        return cid


def log_feedback(chat_id: int, value: int) -> None:
    with _lock:
        e = {"t": "fb", "chat_id": chat_id, "value": value}
        _fb.append(e)
        _append(e)


def stats() -> dict:
    by_lang: dict[str, int] = {}
    cnt: dict[str, int] = {}
    for c in _chats:
        by_lang[c["lang"]] = by_lang.get(c["lang"], 0) + 1
        for s in c.get("sources") or []:
            cnt[s] = cnt.get(s, 0) + 1
    up = sum(1 for f in _fb if f["value"] == 1)
    down = sum(1 for f in _fb if f["value"] == -1)
    avg = int(sum(c["latency_ms"] for c in _chats) / len(_chats)) if _chats else 0
    return {
        "questions": len(_chats),
        "by_lang": by_lang,
        "top_sites": [{"site": s, "count": n} for s, n in sorted(cnt.items(), key=lambda x: -x[1])[:10]],
        "feedback": {"up": up, "down": down, "satisfaction": round(up / (up + down), 2) if up + down else None},
        "avg_latency_ms": avg,
    }
