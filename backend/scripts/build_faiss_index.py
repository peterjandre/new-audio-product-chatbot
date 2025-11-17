"""
Build a FAISS vector index from embeddings_cache.json.

This script loads embeddings from the cache, creates a FAISS index for fast
similarity search, and saves both the index and associated metadata.
The index can be used locally or deployed with a web application.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import faiss
    import numpy as np
except ImportError as e:
    print(f"Error: Required packages not installed. Install with: pip install faiss-cpu numpy", file=sys.stderr)
    sys.exit(1)


def load_embeddings_cache(cache_path: Path) -> list[dict]:
    """Load embeddings from cache file."""
    if not cache_path.exists():
        raise FileNotFoundError(f"Embeddings cache not found at {cache_path}")
    
    with cache_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("Embeddings cache must be a list of entries")
    
    return data


def build_faiss_index(
    embeddings_cache_path: str | Path,
    index_output_path: str | Path,
    metadata_output_path: str | Path | None = None,
    index_type: str = "flat",
    normalize: bool = True,
) -> tuple[faiss.Index, list[dict]]:
    """
    Build a FAISS index from embeddings cache.
    
    Parameters
    ----------
    embeddings_cache_path:
        Path to embeddings_cache.json
    index_output_path:
        Path where FAISS index will be saved
    metadata_output_path:
        Path where metadata JSON will be saved (default: index_output_path with _metadata.json)
    index_type:
        Type of FAISS index: "flat" (exact search) or "ivf" (approximate, faster for large datasets)
    normalize:
        Whether to normalize embeddings for cosine similarity (recommended)
    
    Returns
    -------
    tuple of (faiss.Index, list of metadata dicts)
    """
    cache_path = Path(embeddings_cache_path)
    index_path = Path(index_output_path)
    
    if metadata_output_path is None:
        metadata_path = index_path.parent / f"{index_path.stem}_metadata.json"
    else:
        metadata_path = Path(metadata_output_path)
    
    print(f"Loading embeddings from {cache_path}...")
    embeddings_data = load_embeddings_cache(cache_path)
    
    if not embeddings_data:
        raise ValueError("No embeddings found in cache")
    
    print(f"Found {len(embeddings_data)} embeddings")
    
    # Extract embeddings and metadata
    embeddings = []
    metadata = []
    
    for item in embeddings_data:
        embedding = item.get("embedding")
        if not embedding:
            print(f"Warning: Skipping entry {item.get('entry_id', 'unknown')} - no embedding", file=sys.stderr)
            continue
        
        embeddings.append(embedding)
        
        # Store all metadata for retrieval
        entry_metadata = {
            "entry_id": item.get("entry_id"),
            "embedding_model": item.get("embedding_model"),
            "original_hash": item.get("original_hash"),
        }
        
        # Add nested metadata if present
        if "metadata" in item:
            entry_metadata.update(item["metadata"])
        
        metadata.append(entry_metadata)
    
    if not embeddings:
        raise ValueError("No valid embeddings found")
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings, dtype="float32")
    dimension = embeddings_array.shape[1]
    
    print(f"Embedding dimension: {dimension}")
    print(f"Total vectors: {len(embeddings_array)}")
    
    # Normalize for cosine similarity (recommended for most use cases)
    if normalize:
        print("Normalizing embeddings for cosine similarity...")
        faiss.normalize_L2(embeddings_array)
        index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity when normalized
        distance_metric = "cosine similarity (IP on normalized vectors)"
    else:
        index = faiss.IndexFlatL2(dimension)  # L2 distance
        distance_metric = "L2 distance"
    
    # Add vectors to index
    print(f"Building {index_type} index with {distance_metric}...")
    index.add(embeddings_array)
    
    # Save index
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    print(f"Index saved to: {index_path}")
    
    # Save metadata
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Metadata saved to: {metadata_path}")
    
    print(f"\n✓ Successfully built FAISS index with {index.ntotal} vectors")
    print(f"  Index file size: {index_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"  Metadata file size: {metadata_path.stat().st_size / 1024:.2f} KB")
    
    return index, metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a FAISS vector index from embeddings cache."
    )
    parser.add_argument(
        "--cache",
        default="data/embeddings_cache.json",
        help="Path to embeddings_cache.json (default: data/embeddings_cache.json)",
    )
    parser.add_argument(
        "--output",
        default="data/faiss_index.index",
        help="Path to save FAISS index (default: data/faiss_index.index)",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Path to save metadata JSON (default: auto-generated from --output)",
    )
    parser.add_argument(
        "--index-type",
        choices=["flat", "ivf"],
        default="flat",
        help="Type of FAISS index: flat (exact) or ivf (approximate, for large datasets)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Don't normalize embeddings (use L2 distance instead of cosine similarity)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    
    try:
        build_faiss_index(
            embeddings_cache_path=args.cache,
            index_output_path=args.output,
            metadata_output_path=args.metadata,
            index_type=args.index_type,
            normalize=not args.no_normalize,
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

