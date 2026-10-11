import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import rag
from . import lite
from .db import LITE, connect, init_db
from .ingest import ingest


@asynccontextmanager
async def lifespan(_: FastAPI):
    if LITE:
        print(f"[rihla] mode léger (sans base de données) : {lite.load()} chunks en mémoire")
        yield
        return
    init_db()
    with connect() as conn:
        empty = conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0
    if empty:
        print(f"[rihla] première ingestion : {ingest()} chunks")
    yield


app = FastAPI(title="RihlaAI API", version="0.1.0", lifespan=lifespan)


class ChatReq(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    lang: str = "fr"
    site_id: str | None = None
    history: list[dict] = []


class FeedbackReq(BaseModel):
    chat_id: int
    value: int = Field(ge=-1, le=1)


class TripReq(BaseModel):
    duration: str = "one"  # half | one | two
    interest: str = "mix"  # mix | culture | nature
    lang: str = "fr"


@app.get("/api/health")
def health():
    return {"ok": True, "llm": rag.llm_enabled(), "provider": rag.PROVIDER, "model": rag.current_model()}


@app.get("/api/llm-test")
def llm_test():
    """Ouvrez http://localhost:8000/api/llm-test pour diagnostiquer la connexion au LLM."""
    if not rag.llm_enabled():
        return {"ok": False, "provider": rag.PROVIDER,
                "error": "LLM non configuré : clé absente. Vérifiez le fichier .env (nom exact, emplacement) puis redémarrez."}
    t0 = time.time()
    try:
        reply = rag.complete(None, [{"role": "user", "content": "Réponds uniquement par le mot OK."}], 20)
        return {"ok": True, "provider": rag.PROVIDER, "model": rag.current_model(), "reply": reply.strip()[:100],
                "ms": int((time.time() - t0) * 1000)}
    except rag.RateLimited:
        return {"ok": False, "provider": rag.PROVIDER, "error": "HTTP 429 : quota gratuit dépassé, réessayez dans une minute"}
    except Exception as e:
        return {"ok": False, "provider": rag.PROVIDER, "model": rag.current_model(), "error": f"{type(e).__name__}: {e}"[:500]}


@app.get("/api/sites")
def sites():
    return rag.all_sites()


@app.post("/api/chat")
def chat(req: ChatReq):
    t0 = time.time()
    q = req.question.strip()
    if not q:
        raise HTTPException(400, "empty question")
    lang = req.lang if req.lang in ("fr", "en", "ar") else "fr"
    found = rag.load_sites(rag.retrieve(q, req.site_id))
    ctx_name = next((s["name"]["fr"] for s in found if s["id"] == req.site_id), None)
    mode = "llm"
    llm_error = None
    try:
        if not rag.llm_enabled():
            raise RuntimeError("LLM non configuré : renseignez la clé (GEMINI_API_KEY) dans le fichier .env puis relancez")
        answer, srcs = rag.generate(q, lang, found, ctx_name, req.history)
        if not answer:
            raise RuntimeError("empty answer")
    except rag.RateLimited:
        raise HTTPException(429, "rate limited")
    except Exception as e:  # clé absente, réseau, quota... -> repli extractif
        llm_error = f"{type(e).__name__}: {e}"[:300]
        print(f"[rihla] repli extractif : {llm_error}")
        answer, srcs = rag.extractive(found, lang)
        mode = "extractive"
    latency = int((time.time() - t0) * 1000)
    if LITE:
        chat_id = lite.log_chat(lang, q, answer, srcs, req.site_id, mode, latency)
    else:
        with connect() as conn:
            chat_id = conn.execute(
                """INSERT INTO chat_logs (lang, question, answer, sources, ctx_site, mode, latency_ms)
                   VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                (lang, q, answer, srcs, req.site_id, mode, latency),
            ).fetchone()[0]
    return {"answer": answer, "sources": srcs, "chat_id": chat_id, "mode": mode, "latency_ms": latency, "llm_error": llm_error}


@app.post("/api/itinerary")
def itinerary(req: TripReq):
    sites = rag.all_sites()
    try:
        if not rag.llm_enabled():
            raise RuntimeError("LLM non configuré")
        steps = rag.llm_trip(sites, req.duration, req.interest, req.lang)
        mode = "llm"
    except Exception as e:
        print(f"[rihla] parcours local : {type(e).__name__}: {e}")
        steps = rag.local_trip(sites, req.duration, req.interest)
        mode = "local"
    return {"steps": steps, "mode": mode}


@app.post("/api/feedback")
def feedback(req: FeedbackReq):
    if LITE:
        lite.log_feedback(req.chat_id, req.value)
        return {"ok": True}
    with connect() as conn:
        conn.execute("INSERT INTO feedback (chat_id, value) VALUES (%s,%s)", (req.chat_id, req.value))
    return {"ok": True}


@app.get("/api/stats")
def stats():
    if LITE:
        return lite.stats()
    with connect() as conn:
        total = conn.execute("SELECT count(*) FROM chat_logs").fetchone()[0]
        by_lang = dict(conn.execute("SELECT lang, count(*) FROM chat_logs GROUP BY lang").fetchall())
        top = conn.execute(
            "SELECT s, count(*) FROM chat_logs, unnest(sources) s GROUP BY s ORDER BY 2 DESC LIMIT 10"
        ).fetchall()
        fb = dict(conn.execute("SELECT value, count(*) FROM feedback GROUP BY value").fetchall())
        avg = conn.execute("SELECT coalesce(avg(latency_ms),0)::int FROM chat_logs").fetchone()[0]
    up, down = fb.get(1, 0), fb.get(-1, 0)
    return {
        "questions": total,
        "by_lang": by_lang,
        "top_sites": [{"site": s, "count": c} for s, c in top],
        "feedback": {"up": up, "down": down, "satisfaction": round(up / (up + down), 2) if up + down else None},
        "avg_latency_ms": avg,
    }


# le front statique est servi à la racine (doit rester en dernier)
app.mount("/", StaticFiles(directory=Path(__file__).resolve().parent.parent / "static", html=True), name="static")
