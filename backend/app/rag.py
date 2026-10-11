"""Retrieval (pgvector) + génération (Claude) + mode extractif de secours."""
import json
import os
import re

import httpx

from . import lite
from .db import LITE, connect
from .embeddings import embed_query

# Fournisseur LLM :
#   "gemini"    = Google Gemini (clé gratuite via Google AI Studio) : GEMINI_API_KEY, GEMINI_MODEL
#   "openai"    = toute API compatible OpenAI (Ollama local, Groq, OpenRouter...) : LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
#   "anthropic" = API Claude (payante)
PROVIDER = os.getenv("LLM_PROVIDER", "openai")
if PROVIDER == "gemini":
    # API native Gemini (generateContent, en-tête x-goog-api-key) : la voie documentée par Google
    BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
    API_KEY = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip().strip("\"'")
    MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest").strip()
else:
    BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1").rstrip("/")  # Ollama par défaut
    API_KEY = os.getenv("LLM_API_KEY", "ollama")
    MODEL = os.getenv("LLM_MODEL") or ("claude-sonnet-5-5" if PROVIDER == "anthropic" else "qwen2.5:3b")
REASONING = os.getenv("LLM_REASONING_EFFORT")  # optionnel : "none" / "low"... pour les modèles « thinking »
TIMEOUT = float(os.getenv("LLM_TIMEOUT", "180"))  # un petit modèle sur CPU peut être lent


class RateLimited(Exception):
    pass


def llm_enabled() -> bool:
    if PROVIDER == "anthropic":
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    if PROVIDER == "gemini":
        return bool(API_KEY)
    return True


_gm_model: str | None = None  # modèle Gemini validé (ou découvert) pendant l'exécution


def current_model() -> str:
    return _gm_model or MODEL


def _gm_error(r: httpx.Response) -> str:
    try:
        return r.json()["error"]["message"][:300]
    except Exception:
        return r.text[:300]


def _gm_pick_model() -> str:
    """Le modèle demandé n'existe pas (ou plus) : on choisit un modèle Flash disponible via ListModels."""
    r = httpx.get(f"{BASE_URL}/models", params={"pageSize": 200}, headers={"x-goog-api-key": API_KEY}, timeout=30)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code} (liste des modèles) : {_gm_error(r)}")
    names = [
        m["name"].removeprefix("models/")
        for m in r.json().get("models", [])
        if "generateContent" in (m.get("supportedGenerationMethods") or [])
        and m["name"].removeprefix("models/").startswith("gemini")
        and not re.search(r"image|tts|live|audio|embed|robotics|computer|learnlm|thinking|exp|vision", m["name"])
    ]
    if not names:
        raise RuntimeError("aucun modèle Gemini compatible generateContent n'est disponible avec cette clé")
    names.sort(reverse=True)
    for pool in (
        [n for n in names if "flash" in n and "lite" not in n and "preview" not in n],
        [n for n in names if "flash-lite" in n and "preview" not in n],
        [n for n in names if "flash" in n],
        names,
    ):
        if pool:
            print(f"[rihla] modèle Gemini '{MODEL}' introuvable, utilisation de '{pool[0]}'")
            return pool[0]
    raise RuntimeError("aucun modèle Gemini utilisable")


def _gm_call(model, system, messages, max_tokens, thinking):
    body = {
        "contents": [
            {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
            for m in messages
        ],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.2},
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    if thinking is not None:
        body["generationConfig"]["thinkingConfig"] = thinking
    return httpx.post(
        f"{BASE_URL}/models/{model}:generateContent",
        headers={"x-goog-api-key": API_KEY, "Content-Type": "application/json"},
        json=body,
        timeout=TIMEOUT,
    )


def _gemini_complete(system: str | None, messages: list[dict], max_tokens: int) -> str:
    global _gm_model
    model = current_model()
    think = None
    if REASONING:
        think = {"thinkingBudget": 0} if REASONING in ("none", "0") else {"thinkingLevel": REASONING}
    budget = max_tokens + 1500  # les jetons de réflexion comptent dans maxOutputTokens : on garde de la marge
    discovered = False
    for _ in range(3):
        r = _gm_call(model, system, messages, budget, think)
        if r.status_code == 404 and not discovered:
            discovered = True
            model = _gm_pick_model()
            continue
        if r.status_code == 400 and think is not None and "think" in r.text.lower():
            think = None  # ce modèle n'accepte pas ce réglage de réflexion : on réessaie sans
            continue
        break
    if r.status_code == 429:
        raise RateLimited()
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code} (modèle {model}) : {_gm_error(r)}")
    data = r.json()
    cands = data.get("candidates") or []
    if not cands:
        raise RuntimeError(f"aucune réponse du modèle {model} (contenu bloqué ?) : {data.get('promptFeedback')}")
    c = cands[0]
    parts = (c.get("content") or {}).get("parts") or []
    out = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    if not out.strip():
        raise RuntimeError(
            f"réponse vide du modèle {model} (finishReason={c.get('finishReason')}) : "
            "essayez LLM_REASONING_EFFORT=none dans .env"
        )
    _gm_model = model
    return out


def complete(system: str | None, messages: list[dict], max_tokens: int = 500) -> str:
    if PROVIDER == "gemini":
        return _gemini_complete(system, messages, max_tokens)
    if PROVIDER == "anthropic":
        import anthropic

        try:
            msg = anthropic.Anthropic().messages.create(
                model=MODEL, max_tokens=max_tokens, system=system or "", messages=messages
            )
        except anthropic.RateLimitError as e:
            raise RateLimited() from e
        return "".join(b.text for b in msg.content if b.type == "text")
    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    payload = {"model": MODEL, "messages": msgs, "max_tokens": max_tokens, "temperature": 0.2}
    if REASONING:
        payload["reasoning_effort"] = REASONING
    r = httpx.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json=payload,
        timeout=TIMEOUT,
    )
    if r.status_code == 429:
        raise RateLimited()
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code} : {r.text[:250]}")
    ch = r.json()["choices"][0]
    out = (ch.get("message") or {}).get("content") or ""
    if not out.strip():
        if ch.get("finish_reason") == "length":
            raise RuntimeError("réponse vide (jetons épuisés par la réflexion du modèle) : ajoutez LLM_REASONING_EFFORT=none dans .env")
        raise RuntimeError("réponse vide du modèle")
    return out
LANG_NAMES = {"fr": "français", "en": "English", "ar": "العربية"}

SYSTEM = """Tu es RihlaAI, un guide touristique chaleureux et bavard (sans excès) de la province de Taza (Maroc).
Tu discutes naturellement, comme un assistant conversationnel :
- Réponds aux salutations, aux questions générales de voyage (météo typique, tenue, transport, sécurité, culture marocaine, idées de visite) avec tes connaissances générales, en restant prudent.
- Pour les FAITS PRÉCIS sur Taza (histoire, dates, horaires, prix, distances, conditions d'accès), appuie-toi UNIQUEMENT sur la BASE fournie. Si la BASE ne contient pas l'information, dis-le franchement et conseille de vérifier sur place ou auprès de la délégation du tourisme : n'invente jamais un horaire, un prix ou une distance.
- Si la question n'a aucun rapport avec le voyage, réponds brièvement puis ramène gentiment la conversation vers Taza.
- Réponds dans la langue du visiteur (arabe, français ou anglais), en 150 mots maximum, avec un ton vivant (une courte suggestion ou question de relance est bienvenue).
- Dernière ligne EXACTE : SOURCES: id1,id2 (ids de la BASE réellement utilisés), ou SOURCES: none."""


def retrieve(question: str, site_id: str | None = None, k: int = 8, top_sites: int = 3) -> list[str]:
    if LITE:
        return lite.retrieve(question, site_id, k, top_sites)
    qv = embed_query(question)
    with connect() as conn:
        rows = conn.execute(
            "SELECT site_id, embedding <=> %s AS dist FROM chunks ORDER BY dist LIMIT %s", (qv, k)
        ).fetchall()
    order: list[str] = []
    if site_id:
        order.append(site_id)  # le site de la fiche consultée passe en premier
    for sid, _ in rows:
        if sid not in order:
            order.append(sid)
    return order[:top_sites]


def load_sites(ids: list[str]) -> list[dict]:
    if not ids:
        return []
    if LITE:
        return lite.load_sites(ids)
    with connect() as conn:
        rows = conn.execute("SELECT id, data FROM sites WHERE id = ANY(%s)", (ids,)).fetchall()
    by_id = {r[0]: r[1] for r in rows}
    return [by_id[i] for i in ids if i in by_id]


def base_text(sites: list[dict]) -> str:
    return "\n".join(
        f"[{s['id']}] {s['name']['fr']} ({s['cat']}) — {s['d']['fr']} Conseil : {s['tip']['fr']} || EN: {s['d']['en']}"
        for s in sites
    )


def parse_answer(raw: str, ids: list[str]) -> tuple[str, list[str]]:
    raw = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()  # certains modèles exposent leur raisonnement
    m = re.search(r"SOURCES?\s*:\s*([^\n]*)\s*$", raw, re.I)
    if m:
        srcs = [x for x in re.split(r"[,\s;\[\]]+", m.group(1).strip()) if x in ids]
    else:
        srcs = ids[:1]  # petit modèle qui oublie la ligne SOURCES : on cite le site le plus proche
    text = re.sub(r"\n?\s*SOURCES?\s*:[\s\S]*$", "", raw, flags=re.I).strip()
    return text, list(dict.fromkeys(srcs))


def clean_history(history: list[dict]) -> list[dict]:
    """Garde les 6 derniers tours, commence par 'user', alterne les rôles."""
    out: list[dict] = []
    for h in history[-6:]:
        role, content = h.get("role"), (h.get("content") or "").strip()[:1500]
        if role not in ("user", "assistant") or not content:
            continue
        if not out and role != "user":
            continue
        if out and out[-1]["role"] == role:
            out[-1] = {"role": role, "content": content}
        else:
            out.append({"role": role, "content": content})
    if out and out[-1]["role"] == "user":
        out.pop()  # la question courante est ajoutée séparément
    return out


def generate(question: str, lang: str, sites: list[dict], ctx_site: str | None, history: list[dict]):
    ctx = f"Le visiteur consulte la fiche : {ctx_site}." if ctx_site else "Aucune fiche consultée."
    user = (
        f"CONTEXTE : {ctx}\nLangue par défaut de l'interface : {LANG_NAMES.get(lang, lang)}\n\n"
        f"BASE :\n{base_text(sites)}\n\nQUESTION DU VISITEUR : {question}"
    )
    raw = complete(SYSTEM, [*clean_history(history), {"role": "user", "content": user}], 700)
    return parse_answer(raw, [s["id"] for s in sites])


def extractive(sites: list[dict], lang: str) -> tuple[str, list[str]]:
    top = sites[:2]
    text = "\n\n".join(f"**{s['name'][lang]}**\n{s['d'][lang]}\n💡 {s['tip'][lang]}" for s in top)
    return text, [s["id"] for s in top]


# ---------------- itinéraires ----------------
SLOTS = ["09:00", "10:30", "12:00", "14:30", "16:00"]


def local_trip(sites: list[dict], duration: str, interest: str) -> list[dict]:
    n = {"half": 4, "one": 5, "two": 8}.get(duration, 5)
    cult = [s for s in sites if s["cat"] == "culture"]
    nat = [s for s in sites if s["cat"] == "nature"]
    if interest == "culture" or (interest == "mix" and duration == "half"):
        pick = cult[:n]
    elif interest == "nature":
        pick = nat[:n]
    else:
        pick = cult[: (n + 1) // 2] + nat[: n // 2]
    per_day = -(-len(pick) // 2) if duration == "two" else len(pick)
    per_day = max(per_day, 1)
    return [
        {"day": i // per_day + 1, "time": SLOTS[i % per_day] if i % per_day < len(SLOTS) else "17:00", "site": s["id"], "note": ""}
        for i, s in enumerate(pick)
    ]


def llm_trip(sites: list[dict], duration: str, interest: str, lang: str) -> list[dict]:
    days = 2 if duration == "two" else 1
    intr = {"mix": "culture et nature", "culture": "culture", "nature": "nature"}.get(interest, "culture et nature")
    half = " (une demi-journée : 3 à 4 étapes, le matin)" if duration == "half" else ""
    prompt = (
        f"Compose un parcours touristique de {days} jour(s){half}, centre d'intérêt : {intr}.\n"
        "Utilise UNIQUEMENT les ids de la BASE. Regroupe les sites proches : la ville de Taza d'un côté, "
        "le parc de Tazekka de l'autre (30 à 60 minutes de route), sans les mélanger dans la même demi-journée.\n"
        f'Réponds UNIQUEMENT par un tableau JSON : [{{"day":1,"time":"09:00","site":"id","note":"conseil de 10 mots max en {LANG_NAMES.get(lang, lang)}"}}]\n\n'
        f"BASE :\n{base_text(sites)}"
    )
    raw = complete(None, [{"role": "user", "content": prompt}], 900)
    data = json.loads(raw[raw.index("[") : raw.rindex("]") + 1])
    ids = {s["id"] for s in sites}
    steps = [
        {"day": int(x.get("day", 1)), "time": str(x.get("time", "")), "site": x["site"], "note": str(x.get("note", ""))}
        for x in data
        if isinstance(x, dict) and x.get("site") in ids
    ][:10]
    if not steps:
        raise ValueError("empty trip")
    return steps


def all_sites() -> list[dict]:
    if LITE:
        return lite.all_sites()
    with connect() as conn:
        rows = conn.execute("SELECT data FROM sites ORDER BY id").fetchall()
    order = ["medina", "grande-mosquee", "mosquee-andalous", "remparts", "bastion", "ras-el-ma", "tazekka", "friouato", "dayet-chiker"]
    data = [r[0] for r in rows]
    return sorted(data, key=lambda s: order.index(s["id"]) if s["id"] in order else 99)
