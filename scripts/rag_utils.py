"""
Utilities for preparing retrieval-augmented generation (RAG) assets.

This module currently provides helpers to convert the Gearspace JSON corpus
into text chunks that can later be embedded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence


def _normalize_text_segments(segments: Iterable[str | None]) -> str:
    """Join text segments while dropping falsy values and normalizing whitespace."""
    normalized_segments: list[str] = []
    for segment in segments:
        if not segment:
            continue
        text = " ".join(segment.strip().split())
        if text:
            normalized_segments.append(text)
    return "\n\n".join(normalized_segments)


def _chunk_tokens(tokens: list[str], chunk_size: int, chunk_overlap: int) -> Iterable[list[str]]:
    """Yield token windows with the requested chunk_size and chunk_overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be in the interval [0, chunk_size).")

    if not tokens:
        return

    step = chunk_size - chunk_overlap
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_size]
        if window:
            yield window


def create_corpus_chunks(
    json_path: str | Path,
    *,
    chunk_size: int = 200,
    chunk_overlap: int = 40,
    fields: Sequence[str] = ("title", "summary"),
) -> list[dict[str, Any]]:
    """
    Load a Gearspace JSON corpus file and slice its entries into text chunks.

    Parameters
    ----------
    json_path:
        Path to the JSON file produced by the scraper.
    chunk_size:
        Maximum number of word tokens to include in a chunk.
    chunk_overlap:
        Number of tokens to overlap between consecutive chunks.
    fields:
        Ordered sequence of keys to pull from each corpus entry to construct the
        text content. Missing or falsy fields are skipped automatically.

    Returns
    -------
    list of dict
        A list of chunk dictionaries. Each chunk contains:
        - `id`: deterministic identifier combining the source id and chunk index.
        - `text`: the chunked text ready for embedding.
        - `metadata`: original entry metadata for downstream retrieval.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON corpus not found at {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        corpus = json.load(f)

    if not isinstance(corpus, list):
        raise ValueError("Expected the corpus JSON to be a list of entries.")

    chunks: list[dict[str, Any]] = []

    for entry in corpus:
        if not isinstance(entry, dict):
            continue  # Skip malformed rows.

        text_segments = [str(entry.get(field, "")) for field in fields]
        text = _normalize_text_segments(text_segments)
        if not text:
            # Fallback to anything resembling textual content.
            fallback_fields = [
                entry.get("summary_generated"),
                entry.get("details"),
                entry.get("description"),
            ]
            text = _normalize_text_segments([str(value) for value in fallback_fields if value])

        if not text:
            # No text content available; skip this entry.
            continue

        tokens = text.split()
        for chunk_index, token_window in enumerate(_chunk_tokens(tokens, chunk_size, chunk_overlap)):
            chunk_text = " ".join(token_window)
            chunk_id = f"{entry.get('id', str(json_path))}::chunk-{chunk_index:03d}"
            chunk_metadata = {
                "source_id": entry.get("id"),
                "source_title": entry.get("title"),
                "source_link": entry.get("link"),
                "published": entry.get("published"),
                "chunk_index": chunk_index,
                "chunk_size": len(token_window),
            }

            chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": chunk_metadata,
                }
            )

    return chunks


