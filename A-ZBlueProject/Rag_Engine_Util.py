# File: rag_engine_and_utils.py
# Contains two modules in one file for easy copy-paste:
# 1) rag_engine (embedding/index/search/generation)
# 2) utils_rag (upgraded utils with authentication, file extractors, wrappers)


# ------------------------
# rag_engine module
# ------------------------
import os
import tempfile
import numpy as np
from sentence_transformers import SentenceTransformer


# Optional: faiss
try:
import faiss
except Exception:
faiss = None


# Optional OpenAI
try:
import openai
except Exception:
openai = None


EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
EMBED_CACHE_DIR = os.environ.get("EMBED_CACHE_DIR", ".emb_cache")
INDEX_FILE = os.path.join(EMBED_CACHE_DIR, "faiss.index")
META_FILE = os.path.join(EMBED_CACHE_DIR, "meta.npy")


os.makedirs(EMBED_CACHE_DIR, exist_ok=True)


# Load embedding model lazily
_embed_model = None


def get_embed_model():
global _embed_model
if _embed_model is None:
_embed_model = SentenceTransformer(EMBED_MODEL_NAME)
return _embed_model


# ------------------------
# Index building / loading
# ------------------------
def build_index_from_documents(docs, rebuild=False):
"""
docs: list of dicts: {"text": ..., "source": ..., "meta": {...}}
If rebuild=True, overwrite existing index.
"""
if faiss is None:
raise RuntimeError("faiss not installed. Please install faiss-cpu or switch to an alternative.")


model = get_embed_model()
texts = [d["text"] for d in docs]
embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


dim = embeddings.shape[1]
faiss.normalize_L2(embeddings)


if rebuild or not os.path.exists(INDEX_FILE):
index = faiss.IndexFlatIP(dim)
index.add(embeddings)
faiss.write_index(index, INDEX_FILE)
# answer = synthesize_answer_local(hits, user_query)
