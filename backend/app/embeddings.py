"""Embeddings multilingues (FR / EN / AR) via fastembed (ONNX, sans PyTorch).

EMBED_FAKE=1 active un embedder lexical bidon, utile uniquement pour les tests
sans téléchargement de modèle.
"""
import hashlib
import math
import os
import re
import threading

import numpy as np

MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
DIM = int(os.getenv("EMBED_DIM", "384"))
FAKE = os.getenv("EMBED_FAKE") == "1"

_model = None
_lock = threading.Lock()


def _get_model():
    global _model
    with _lock:
        if _model is None:
            from fastembed import TextEmbedding

            _model = TextEmbedding(model_name=MODEL, cache_dir=os.getenv("FASTEMBED_CACHE_PATH") or None)
        return _model


def _fake(text: str) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    for tok in re.findall(r"\w+", text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        v[h % DIM] += 1.0
    n = math.sqrt(float((v * v).sum())) or 1.0
    return v / n


def embed_passages(texts: list[str]) -> list[np.ndarray]:
    if FAKE:
        return [_fake(t) for t in texts]
    return [np.asarray(v, dtype=np.float32) for v in _get_model().passage_embed(texts)]


def embed_query(text: str) -> np.ndarray:
    if FAKE:
        return _fake(text)
    return np.asarray(next(iter(_get_model().query_embed([text]))), dtype=np.float32)
