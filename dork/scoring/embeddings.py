from __future__ import annotations

import logging
import math

import httpx

log = logging.getLogger(__name__)

S2_API_URL = "https://api.semanticscholar.org/graph/v1/paper"


def fetch_embedding(arxiv_id: str) -> list[float] | None:
    """Fetch SPECTER v2 embedding for an arXiv paper from Semantic Scholar."""
    url = f"{S2_API_URL}/ARXIV:{arxiv_id}"
    try:
        resp = httpx.get(url, params={"fields": "embedding.specter_v2"}, timeout=15)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        log.debug("s2 api error", extra={"arxiv_id": arxiv_id, "error": str(e)})
        return None

    data = resp.json()
    embedding = data.get("embedding")
    if not embedding:
        return None

    vector = embedding.get("vector")
    if not vector or not isinstance(vector, list):
        return None

    return vector


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def max_similarity(candidate_embedding: list[float], reference_embeddings: list[list[float]]) -> float:
    """Compute max cosine similarity between a candidate and all reference embeddings."""
    if not reference_embeddings:
        return 1.0  # No references → don't filter
    return max(cosine_similarity(candidate_embedding, ref) for ref in reference_embeddings)
