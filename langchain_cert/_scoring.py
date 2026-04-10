"""
CERT Geometric Grounding — standalone scoring implementation.

SGI (Semantic Grounding Index) — arXiv:2512.13771
    SGI = dist(response, question) / dist(response, context)
    Threshold: < 0.95 flags for review, >= 1.20 is strong pass.

DGI (Directional Grounding Index) — arXiv:2602.13224
    DGI = dot(normalize(phi(r) - phi(q)), mu_hat)
    Threshold: < 0.30 flags for review.

Both methods use the same embedding model (all-MiniLM-L6-v2 by default).
The embedding model is loaded once per process and cached.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Thresholds (from arXiv:2512.13771) ───────────────────────────────────────

SGI_STRONG_PASS = 1.20   # Response clearly moved toward context
SGI_REVIEW      = 0.95   # Below this: flag for review
DGI_PASS        = 0.30   # Displacement aligns with grounded patterns


# ── Embedding model (module-level singleton) ──────────────────────────────────

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_encoder = None
_encoder_model_name: Optional[str] = None


def _get_encoder(model_name: str = _DEFAULT_MODEL) -> Any:
    """Load embedding model once, cache for process lifetime."""
    global _encoder, _encoder_model_name
    if _encoder is None or _encoder_model_name != model_name:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for local scoring. "
                "Install with: pip install langchain-cert[local]"
            ) from e
        logger.info("Loading embedding model: %s", model_name)
        _encoder = SentenceTransformer(model_name)
        _encoder_model_name = model_name
        logger.info("Embedding model loaded.")
    return _encoder


# ── DGI reference direction ───────────────────────────────────────────────────

def _load_reference_pairs(
    reference_csv: Optional[str] = None,
) -> list[tuple[str, str]]:
    """
    Load grounded (question, response) pairs for DGI calibration.

    Two sources:
    - Bundled dataset (default): loaded from langchain_cert/data/reference_pairs.csv
      using importlib.resources. Contains finance and medical domain pairs from
      Claude and Gemini responses.
    - User-provided CSV (reference_csv parameter): simple two-column format.
      See format requirements below.

    User CSV format:
        - Comma OR semicolon delimited (auto-detected)
        - Required columns: "question" and one of "response", "answer", or "output"
        - Header row required
        - Encoding: UTF-8
        - Each row should be a verified grounded (question, response) pair.
          Do NOT include hallucinated responses — they will degrade calibration.

    Example user CSV:
        question,response
        What is our refund policy?,Refunds are processed within 5 business days.
        How do I reset my password?,Navigate to Settings > Security > Reset Password.

    Args:
        reference_csv: Path to a user-provided CSV file. If None, loads the
                       bundled dataset.

    Returns:
        List of (question, response) string tuples.

    Raises:
        FileNotFoundError: If reference_csv path does not exist.
        ValueError: If the CSV is missing required columns or contains no valid rows.
    """
    if reference_csv is not None:
        return _load_user_csv(reference_csv)
    return _load_bundled_csv()


def _load_bundled_csv() -> list[tuple[str, str]]:
    """Load the bundled reference dataset from package data."""
    import csv
    from importlib import resources

    pairs: list[tuple[str, str]] = []
    ref = resources.files("langchain_cert.data").joinpath("reference_pairs.csv")
    raw = ref.read_text(encoding="utf-8-sig")
    reader = csv.DictReader(raw.splitlines(), delimiter=";")
    for row in reader:
        q = row.get("question", "").strip()
        ans = row.get("grounded_response", "").strip()
        if q and ans:
            pairs.append((q, ans))

    if not pairs:
        raise ValueError(
            "Bundled reference dataset loaded 0 pairs. "
            "The package installation may be corrupted."
        )
    return pairs


def _load_user_csv(path: str) -> list[tuple[str, str]]:
    """
    Load user-provided reference CSV.

    Auto-detects delimiter. Accepts 'response', 'answer', or 'output'
    as the response column name.
    """
    import csv
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Reference CSV not found: {path}\n"
            "Provide a path to a CSV file with columns: question, response"
        )

    # Auto-detect delimiter from first line
    with p.open(encoding="utf-8") as f:
        sample = f.read(1024)

    delimiter = ";" if sample.count(";") > sample.count(",") else ","

    pairs: list[tuple[str, str]] = []
    with p.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file appears empty: {path}")

        # Find the response column — accept multiple names
        response_col = None
        for candidate in ("response", "answer", "output"):
            if candidate in (reader.fieldnames or []):
                response_col = candidate
                break

        if "question" not in (reader.fieldnames or []):
            raise ValueError(
                f"CSV missing required 'question' column. "
                f"Found columns: {list(reader.fieldnames or [])}"
            )
        if response_col is None:
            raise ValueError(
                f"CSV missing response column. "
                f"Expected one of: 'response', 'answer', 'output'. "
                f"Found columns: {list(reader.fieldnames or [])}"
            )

        for row in reader:
            q = row.get("question", "").strip()
            ans = row.get(response_col, "").strip()
            if q and ans:
                pairs.append((q, ans))

    if not pairs:
        raise ValueError(
            f"No valid pairs loaded from {path}. "
            "Check that question and response columns contain data."
        )

    logger.info("Loaded %d reference pairs from %s.", len(pairs), path)
    return pairs


def _compute_reference_direction_from_pairs(
    pairs: list[tuple[str, str]],
    model_name: str = _DEFAULT_MODEL,
) -> np.ndarray:
    """Compute DGI reference direction (mu_hat) from grounded pairs."""
    encoder = _get_encoder(model_name)

    texts: list[str] = []
    for q, r in pairs:
        texts.extend([q, r])

    embs = encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=False)

    displacements = []
    for i in range(len(pairs)):
        q_emb = embs[i * 2]
        r_emb = embs[i * 2 + 1]
        delta = r_emb - q_emb
        norm = float(np.linalg.norm(delta))
        if norm > 1e-8:
            displacements.append(delta / norm)

    mu: np.ndarray = np.mean(displacements, axis=0)
    mu_norm = float(np.linalg.norm(mu))
    result: np.ndarray = mu / mu_norm if mu_norm > 1e-8 else mu
    return result


_mu_hat: dict[tuple[str, str], np.ndarray] = {}


def _get_mu_hat(
    model_name: str = _DEFAULT_MODEL,
    reference_csv: Optional[str] = None,
) -> np.ndarray:
    """
    Get DGI reference direction.

    Computes once per (model_name, reference_csv) pair and caches.
    Using different reference_csv paths in the same process produces
    independent reference directions — no cache collisions.
    """
    cache_key = (model_name, reference_csv or "bundled")
    if cache_key not in _mu_hat:
        logger.info(
            "Computing DGI reference direction (model=%s, data=%s)...",
            model_name,
            reference_csv or "bundled",
        )
        pairs = _load_reference_pairs(reference_csv)
        _mu_hat[cache_key] = _compute_reference_direction_from_pairs(
            pairs, model_name
        )
        logger.info(
            "DGI reference direction ready (dims=%d, pairs=%d).",
            _mu_hat[cache_key].shape[0],
            len(pairs),
        )
    return _mu_hat[cache_key]


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class SGIResult:
    """Result of Semantic Grounding Index computation."""
    raw_score: float          # dist(response, question) / dist(response, context)
    normalized: float         # 0.0 – 1.0 for LangSmith compatibility
    flag: bool                # True if below SGI_REVIEW threshold
    q_dist: float             # dist(response, question)
    ctx_dist: float           # dist(response, context)
    method: str = "sgi"


@dataclass
class DGIResult:
    """Result of Directional Grounding Index computation."""
    raw_score: float          # cosine similarity to reference direction
    normalized: float         # 0.0 – 1.0 for LangSmith compatibility
    flag: bool                # True if below DGI_PASS threshold
    method: str = "dgi"


# ── Scoring functions ─────────────────────────────────────────────────────────

def compute_sgi(
    question: str,
    context: str,
    response: str,
    model_name: str = _DEFAULT_MODEL,
) -> SGIResult:
    """
    Compute Semantic Grounding Index.

    Measures whether the response engaged with the provided context or
    stayed anchored to the question (semantic laziness / confabulation).

    Higher SGI = stronger context engagement = grounded.

    Args:
        question: The input query.
        context:  Source document or retrieved chunks.
        response: The LLM's response to evaluate.
        model_name: Sentence transformer model (default: all-MiniLM-L6-v2).

    Returns:
        SGIResult with raw score, normalized score, and flag.
    """
    encoder = _get_encoder(model_name)
    embs = encoder.encode(
        [question, context, response],
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    q_emb, ctx_emb, resp_emb = embs

    q_dist   = float(np.linalg.norm(resp_emb - q_emb))
    ctx_dist = float(np.linalg.norm(resp_emb - ctx_emb))

    if ctx_dist < 1e-8:
        # Response identical to context — degenerate, treat as strongly grounded
        return SGIResult(
            raw_score=10.0, normalized=1.0, flag=False,
            q_dist=q_dist, ctx_dist=ctx_dist,
        )
    if q_dist < 1e-8:
        # Response identical to question — degenerate, treat as ungrounded
        return SGIResult(
            raw_score=0.0, normalized=0.0, flag=True,
            q_dist=q_dist, ctx_dist=ctx_dist,
        )

    raw = q_dist / ctx_dist

    # Normalize to [0, 1] for LangSmith.
    # tanh maps raw SGI to a smooth 0-1 curve.
    # SGI 0.95 (review threshold) → ~0.46; SGI 1.20 (strong pass) → ~0.60.
    normalized = float(math.tanh(max(0.0, raw - 0.3)))
    normalized = min(1.0, max(0.0, normalized))

    return SGIResult(
        raw_score=round(raw, 4),
        normalized=round(normalized, 4),
        flag=raw < SGI_REVIEW,
        q_dist=round(q_dist, 4),
        ctx_dist=round(ctx_dist, 4),
    )


def compute_dgi(
    question: str,
    response: str,
    model_name: str = _DEFAULT_MODEL,
    reference_csv: Optional[str] = None,
) -> DGIResult:
    """
    Compute Directional Grounding Index.

    Measures whether the query-to-response displacement vector aligns with
    the mean displacement of verified grounded (question, response) pairs.
    Detects confabulation (Type II hallucinations) without source context.

    Higher DGI = displacement matches grounded patterns = likely grounded.

    Args:
        question:      The input query.
        response:      The LLM's response to evaluate.
        model_name:    Sentence transformer model.
        reference_csv: Path to a user-provided CSV for DGI calibration.
                       If None, uses the bundled domain-specific dataset.
                       See _load_reference_pairs() for CSV format.

    Returns:
        DGIResult with raw score, normalized score, and flag.
    """
    encoder = _get_encoder(model_name)
    mu_hat = _get_mu_hat(model_name, reference_csv)

    embs = encoder.encode(
        [question, response],
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    q_emb, r_emb = embs

    delta = r_emb - q_emb
    magnitude = float(np.linalg.norm(delta))

    if magnitude < 1e-8:
        return DGIResult(raw_score=0.0, normalized=0.0, flag=True)

    delta_hat = delta / magnitude
    gamma = float(np.dot(delta_hat, mu_hat))

    if math.isnan(gamma):
        logger.warning("DGI produced NaN — check embedding dimensions.")
        return DGIResult(raw_score=0.0, normalized=0.0, flag=True)

    # Normalize [-1, 1] → [0, 1]
    normalized = round((gamma + 1.0) / 2.0, 4)
    normalized = min(1.0, max(0.0, normalized))

    return DGIResult(
        raw_score=round(gamma, 4),
        normalized=normalized,
        flag=gamma < DGI_PASS,
    )
