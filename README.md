# RihlaAI : MVP Taza Smart Tourism

Assistant touristique multilingue (FR / EN / AR) pour la province de Taza.
Front mobile-first + API FastAPI + RAG (pgvector, ou en mémoire) + LLM Gemini (clé gratuite) ou Ollama (local).

## LLM : Gemini (clé gratuite) ou Ollama (local)

**Gemini** (configuré dans `.env.example`) :
1. Créez une clé sur https://aistudio.google.com/apikey (pas de carte bancaire pour le palier gratuit).
2. Copiez `.env.example` en `.env` (à la racine du projet ou dans `backend/`) et collez la clé dans `GEMINI_API_KEY`.
   Le fichier est chargé automatiquement, sans rien exporter.
3. Le modèle par défaut est l'alias `gemini-flash-latest`. Vous pouvez imposer un modèle précis avec `GEMINI_MODEL`
   (liste actuelle dans AI Studio).

Points d'attention Gemini :
- Le palier gratuit est limité (quelques requêtes par minute selon le modèle) : en cas de dépassement l'API
  renvoie une erreur 429 et le front affiche « Trop de requêtes ».
- Sur le palier gratuit, Google peut utiliser les entrées et sorties pour améliorer ses modèles :
  n'envoyez pas de données personnelles.
- Avec un modèle « thinking » (ex. `gemini-2.5-flash`), la réflexion consomme des jetons : si les réponses sont
  vides ou coupées, ajoutez `LLM_REASONING_EFFORT=none`.

**Ollama** (100 % local, sans quota) : installez https://ollama.com, puis `ollama pull qwen2.5:3b`
(ou `qwen2.5:7b`, meilleur en arabe), et utilisez `LLM_PROVIDER=openai` (voir `.env.example`).

Le fournisseur `openai` couvre toute API compatible (Ollama, Groq, OpenRouter...) via `LLM_BASE_URL`,
`LLM_API_KEY`, `LLM_MODEL`. Claude reste possible avec `LLM_PROVIDER=anthropic`.
Si le LLM est injoignable ou non configuré, l'API retombe sur le mode extractif (réponses tirées des fiches).

## Tester la connexion à Gemini en 10 secondes

Lancez le serveur puis ouvrez **http://localhost:8000/api/llm-test**. Réponse `"ok": true` = Gemini fonctionne.
Sinon, le champ `error` donne la cause exacte (clé invalide, quota, modèle introuvable...).
L'app appelle l'API native de Gemini (`generateContent`, en-tête `x-goog-api-key`), ignore les guillemets et espaces
parasites dans la clé, et, si le modèle de `GEMINI_MODEL` n'existe plus, en choisit un autre disponible avec votre clé.

## Dépannage : « l'assistant répond comme un moteur de recherche, pas comme un LLM »

Quand l'appel au LLM échoue, l'API retombe sur le mode extractif (texte des fiches) et le front affiche,
sous la réponse, « Mode sans IA » suivi de la **cause exacte**. Causes fréquentes :
- `LLM non configuré` : le fichier `.env` n'est pas trouvé ou `GEMINI_API_KEY` est vide. Vérifiez `http://localhost:8000/api/health`
  (`"llm": true`, bon `provider` et bon `model`), puis **redémarrez le serveur** après toute modification de `.env`.
- `HTTP 400` / `403` : clé invalide ou nom de modèle inexistant (changez `GEMINI_MODEL`).
- `HTTP 429` : quota gratuit dépassé, patientez une minute.
- `réponse vide` : modèle « thinking » : ajoutez `LLM_REASONING_EFFORT=none` dans `.env`.

## Lancer sur Windows sans Docker (le plus simple)

Mode léger : pas de PostgreSQL, les embeddings restent en mémoire et les journaux
vont dans `rihla_logs.jsonl`. Il est actif tant que la variable `DATABASE_URL` n'est pas définie.
Prérequis : Python 3.10 à 3.12, et un fichier `.env` avec votre clé Gemini (`copy ..\.env.example ..\.env`).

```powershell
cd rihlaai-mvp\backend
py -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
# la clé Gemini est lue dans le fichier .env (voir ci-dessus) ; sinon :
# $env:LLM_PROVIDER = "gemini"; $env:GEMINI_API_KEY = "votre-clé"
.\.venv\Scripts\python -m uvicorn app.main:app --port 8000
# -> http://localhost:8000
```

Si `py` n'est pas reconnu, essayez `python` à la place. Le premier démarrage télécharge
le modèle d'embeddings (~120 Mo) et peut prendre une minute.

## Lancer avec Docker (PostgreSQL + pgvector)

```bash
cp .env.example .env        # puis coller la clé Gemini dans .env
docker compose up --build
# -> http://localhost:8000
```

Au premier démarrage : création du schéma, téléchargement du modèle d'embeddings
(~120 Mo, mis en cache dans le volume `models`) et indexation des 9 fiches de `backend/data/sites.json`.

Si le LLM est injoignable ou non configuré, l'assistant fonctionne en **mode extractif** (réponses tirées des fiches, sans LLM).
Le front affiche alors « Mode sans IA ».

## Lancer sans Docker

```bash
# PostgreSQL 16 avec l'extension pgvector, puis :
createdb rihla
export DATABASE_URL=postgresql://USER:PASS@localhost:5432/rihla
# la clé Gemini est lue dans .env
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Architecture

```
static/index.html --fetch--> FastAPI (app/main.py)
                               |- /api/chat       embed(question) -> pgvector top-k -> LLM -> réponse + sources
                               |- /api/itinerary  parcours (LLM, repli déterministe)
                               |- /api/sites      fiches
                               |- /api/feedback   pouce haut / bas
                               '- /api/stats      usage, langues, sites cités, satisfaction, latence
PostgreSQL + pgvector : sites, chunks (embeddings), chat_logs, feedback
```

| Fichier | Rôle |
|---|---|
| `app/embeddings.py` | embeddings multilingues via fastembed (modèle configurable : `EMBED_MODEL`, `EMBED_DIM`) |
| `app/ingest.py` | `sites.json` -> chunks (1 par site et par langue) -> pgvector. `python -m app.ingest` pour ré-indexer |
| `app/rag.py` | retrieval, appel LLM (compatible OpenAI), parsing des sources, repli extractif, parcours |
| `app/main.py` | endpoints + service du front |
| `data/sites.json` | **source de vérité du contenu** |

## Mettre à jour le contenu

1. Éditer `backend/data/sites.json` (ajouter un site = même structure : `name`, `d`, `tip`, `kw` en fr/en/ar).
2. `docker compose exec api python -m app.ingest`.
3. Le front embarque encore sa propre copie des fiches (`SITES` dans `static/index.html`) :
   pour un site ajouté, la mettre à jour aussi, ou la remplacer par un `fetch('/api/sites')` (étape suivante naturelle).

## Points d'attention

- **Contenu à faire valider** (fiches marquées « à valider », coordonnées approximatives).
- Ajouter une **limitation de débit** (ex. slowapi) avant toute exposition publique : chaque question appelle le LLM.
- Les petits modèles gratuits se trompent plus souvent : gardez l'instruction « réponds uniquement depuis la base » et testez vos questions types (FR, AR, EN) avant le pilote.
- Pour changer de modèle d'embeddings (ex. `intfloat/multilingual-e5-large`), ajuster `EMBED_MODEL` et
  `EMBED_DIM` **puis recréer la base** (la dimension du vecteur est fixée à la création de la table).
- Les identifiants de la base dans `docker-compose.yml` sont des valeurs de développement.

## Idées pour la suite

amazigh, reconnaissance de monuments par photo, PWA hors-ligne, back-office de saisie des fiches,
tableau de bord Power BI branché sur `chat_logs`.
