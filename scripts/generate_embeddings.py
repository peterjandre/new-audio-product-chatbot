"""
Generate and cache embeddings for the Gearspace corpus using OpenAI's embedding API.

This script generates embeddings only for new or changed entries, avoiding
unnecessary API calls and costs. It uses the original_hash field to detect
content changes, similar to how the scraper tracks updates.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from rag_utils import _normalize_text_segments, _chunk_tokens


def load_embeddings_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    """Load existing embeddings cache, returning empty dict if file doesn't exist."""
    if not cache_path.exists():
        return {}

    with cache_path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            # Convert list to dict keyed by entry_id for faster lookup
            if isinstance(data, list):
                return {item["entry_id"]: item for item in data}
            return data
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"Warning: Could not parse embeddings cache: {exc}", file=sys.stderr)
            return {}


def save_embeddings_cache(cache_path: Path, embeddings: dict[str, dict[str, Any]]) -> None:
    """Save embeddings cache to JSON file."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert dict to list for JSON serialization
    embeddings_list = list(embeddings.values())
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(embeddings_list, f, ensure_ascii=False, indent=2)


def get_text_for_embedding(entry: dict[str, Any], fields: tuple[str, ...] = ("title", "summary")) -> str:
    """Extract and normalize text from corpus entry for embedding."""
    text_segments = [str(entry.get(field, "")) for field in fields]
    return _normalize_text_segments(text_segments)


def generate_embeddings_batch(
    texts: list[str],
    client: OpenAI,
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
) -> list[list[float]]:
    """
    Generate embeddings for a batch of texts using OpenAI API.
    
    OpenAI's API supports batching, but we'll process in chunks to handle
    large corpora efficiently and respect rate limits.
    """
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            response = client.embeddings.create(
                model=model,
                input=batch,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        except Exception as exc:
            print(f"Error generating embeddings for batch {i//batch_size + 1}: {exc}", file=sys.stderr)
            raise
    
    return all_embeddings


def process_corpus_without_chunking(
    corpus: list[dict[str, Any]],
    cache: dict[str, dict[str, Any]],
    client: OpenAI,
    model: str,
    fields: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    """Process corpus entries without chunking (one embedding per entry)."""
    entries_to_embed = []
    entry_indices = []
    
    for idx, entry in enumerate(corpus):
        entry_id = entry.get("id")
        original_hash = entry.get("original_hash")
        
        if not entry_id:
            continue
        
        # Check if we need to embed this entry
        cached = cache.get(entry_id)
        needs_embedding = (
            cached is None  # Not in cache
            or cached.get("original_hash") != original_hash  # Content changed
            or cached.get("embedding_model") != model  # Model changed
        )
        
        if needs_embedding:
            text = get_text_for_embedding(entry, fields)
            if text:
                entries_to_embed.append((entry_id, entry, text))
                entry_indices.append(idx)
    
    if not entries_to_embed:
        print("All entries already have up-to-date embeddings.")
        return cache
    
    print(f"Generating embeddings for {len(entries_to_embed)} entries...")
    
    # Extract texts for batch API call
    texts = [text for _, _, text in entries_to_embed]
    embeddings = generate_embeddings_batch(texts, client, model)
    
    # Update cache with new embeddings
    updated_cache = dict(cache)
    for (entry_id, entry, _), embedding in zip(entries_to_embed, embeddings):
        updated_cache[entry_id] = {
            "entry_id": entry_id,
            "embedding": embedding,
            "embedding_model": model,
            "original_hash": entry.get("original_hash"),
            "metadata": {
                "title": entry.get("title"),
                "link": entry.get("link"),
                "published": entry.get("published"),
            },
        }
    
    return updated_cache


def process_corpus_with_chunking(
    corpus: list[dict[str, Any]],
    cache: dict[str, dict[str, Any]],
    client: OpenAI,
    model: str,
    chunk_size: int,
    chunk_overlap: int,
    fields: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    """Process corpus entries with chunking (multiple embeddings per entry)."""
    chunks_to_embed = []
    
    for entry in corpus:
        entry_id = entry.get("id")
        original_hash = entry.get("original_hash")
        
        if not entry_id:
            continue
        
        # Extract text using same logic as rag_utils
        text_segments = [str(entry.get(field, "")) for field in fields]
        text = _normalize_text_segments(text_segments)
        if not text:
            continue
        
        # Tokenize and chunk using rag_utils utilities
        tokens = text.split()
        chunk_index = 0
        for token_window in _chunk_tokens(tokens, chunk_size, chunk_overlap):
            chunk_text = " ".join(token_window)
            chunk_id = f"{entry_id}::chunk-{chunk_index:03d}"
            
            # Check if we need to embed this chunk
            cached = cache.get(chunk_id)
            needs_embedding = (
                cached is None
                or cached.get("original_hash") != original_hash
                or cached.get("embedding_model") != model
            )
            
            if needs_embedding:
                chunks_to_embed.append((chunk_id, entry, chunk_text, original_hash, chunk_index))
            
            chunk_index += 1
    
    if not chunks_to_embed:
        print("All chunks already have up-to-date embeddings.")
        return cache
    
    print(f"Generating embeddings for {len(chunks_to_embed)} chunks...")
    
    # Extract texts for batch API call
    texts = [text for _, _, text, _, _ in chunks_to_embed]
    embeddings = generate_embeddings_batch(texts, client, model)
    
    # Update cache with new embeddings
    updated_cache = dict(cache)
    for (chunk_id, entry, _, original_hash, chunk_index), embedding in zip(chunks_to_embed, embeddings):
        updated_cache[chunk_id] = {
            "entry_id": chunk_id,
            "embedding": embedding,
            "embedding_model": model,
            "original_hash": original_hash,
            "metadata": {
                "source_id": entry.get("id"),
                "source_title": entry.get("title"),
                "source_link": entry.get("link"),
                "published": entry.get("published"),
                "chunk_index": chunk_index,
            },
        }
    
    return updated_cache


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and cache embeddings for Gearspace corpus using OpenAI."
    )
    parser.add_argument(
        "--corpus",
        default="data/gearspace_corpus.json",
        help="Path to the corpus JSON file.",
    )
    parser.add_argument(
        "--cache",
        default="data/embeddings_cache.json",
        help="Path to the embeddings cache JSON file.",
    )
    parser.add_argument(
        "--model",
        default="text-embedding-3-small",
        help="OpenAI embedding model to use (e.g., text-embedding-3-small, text-embedding-3-large).",
    )
    parser.add_argument(
        "--chunk",
        action="store_true",
        help="Enable chunking (multiple embeddings per entry).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Maximum number of word tokens per chunk (only used with --chunk).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=40,
        help="Number of tokens to overlap between chunks (only used with --chunk).",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["title", "summary"],
        help="Fields to include in embedding text (default: title summary).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of texts to embed per API batch call.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if OpenAI is None:
        print("Error: openai package not installed. Install it with: pip install openai", file=sys.stderr)
        return 1
    
    args = parse_args(argv)
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        return 1
    
    client = OpenAI(api_key=api_key)
    
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"Error: Corpus file not found at {corpus_path}", file=sys.stderr)
        return 1
    
    cache_path = Path(args.cache)
    cache = load_embeddings_cache(cache_path)
    
    print(f"Loaded {len(cache)} existing embeddings from cache.")
    
    with corpus_path.open("r", encoding="utf-8") as f:
        corpus = json.load(f)
    
    if not isinstance(corpus, list):
        print("Error: Corpus must be a list of entries.", file=sys.stderr)
        return 1
    
    print(f"Processing {len(corpus)} corpus entries...")
    
    fields_tuple = tuple(args.fields)
    
    if args.chunk:
        updated_cache = process_corpus_with_chunking(
            corpus=corpus,
            cache=cache,
            client=client,
            model=args.model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            fields=fields_tuple,
        )
    else:
        updated_cache = process_corpus_without_chunking(
            corpus=corpus,
            cache=cache,
            client=client,
            model=args.model,
            fields=fields_tuple,
        )
    
    save_embeddings_cache(cache_path, updated_cache)
    print(f"Saved {len(updated_cache)} embeddings to {cache_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

