"""
Query a FAISS index for similar embeddings.

This script demonstrates how to load and query a FAISS index for similarity search.
You can use this as a reference when building your web application.
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


class FAISSIndex:
    """Wrapper class for FAISS index with metadata."""
    
    def __init__(self, index_path: str | Path, metadata_path: str | Path | None = None):
        """
        Load FAISS index and metadata.
        
        Parameters
        ----------
        index_path:
            Path to FAISS index file
        metadata_path:
            Path to metadata JSON file (default: index_path with _metadata.json)
        """
        index_path = Path(index_path)
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        
        if metadata_path is None:
            metadata_path = index_path.parent / f"{index_path.stem}_metadata.json"
        else:
            metadata_path = Path(metadata_path)
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
        
        # Load index
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata
        with metadata_path.open("r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        
        if len(self.metadata) != self.index.ntotal:
            print(
                f"Warning: Metadata count ({len(self.metadata)}) doesn't match index size ({self.index.ntotal})",
                file=sys.stderr
            )
        
        print(f"Loaded FAISS index with {self.index.ntotal} vectors")
        print(f"Index dimension: {self.index.d}")
    
    def search(
        self,
        query_embedding: list[float] | np.ndarray,
        k: int = 5,
        normalize: bool = True,
    ) -> list[dict]:
        """
        Search for similar embeddings.
        
        Parameters
        ----------
        query_embedding:
            Query vector (list or numpy array)
        k:
            Number of results to return
        normalize:
            Whether to normalize the query vector (should match how index was built)
        
        Returns
        -------
        List of result dicts with keys: entry_id, score, metadata
        """
        # Convert to numpy array
        if isinstance(query_embedding, list):
            query_vector = np.array([query_embedding], dtype="float32")
        else:
            query_vector = query_embedding.reshape(1, -1).astype("float32")
        
        # Normalize if needed (for cosine similarity)
        if normalize:
            faiss.normalize_L2(query_vector)
        
        # Search
        distances, indices = self.index.search(query_vector, min(k, self.index.ntotal))
        
        # Build results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
            
            result = {
                "rank": i + 1,
                "score": float(distance),
                "entry_id": self.metadata[idx]["entry_id"],
                "metadata": self.metadata[idx],
            }
            results.append(result)
        
        return results
    
    def search_by_text(
        self,
        text: str,
        openai_client,
        model: str = "text-embedding-3-small",
        k: int = 5,
    ) -> list[dict]:
        """
        Search using text query (embeds text first, then searches).
        
        Parameters
        ----------
        text:
            Text to search for
        openai_client:
            OpenAI client instance
        model:
            Embedding model to use
        k:
            Number of results to return
        
        Returns
        -------
        List of result dicts
        """
        # Generate embedding for query text
        response = openai_client.embeddings.create(
            model=model,
            input=[text],
        )
        query_embedding = response.data[0].embedding
        
        # Search using embedding
        return self.search(query_embedding, k=k)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query FAISS index for similar embeddings."
    )
    parser.add_argument(
        "--index",
        default="data/faiss_index.index",
        help="Path to FAISS index file (default: data/faiss_index.index)",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Path to metadata JSON file (default: auto-detected)",
    )
    parser.add_argument(
        "--query-embedding",
        help="Query embedding as comma-separated values (for testing)",
    )
    parser.add_argument(
        "--query-text",
        help="Query text (requires OPENAI_API_KEY to generate embedding)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    
    try:
        # Load index
        faiss_index = FAISSIndex(args.index, args.metadata)
        
        # Perform search
        if args.query_text:
            # Text-based search
            import os
            from openai import OpenAI
            
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print("Error: OPENAI_API_KEY required for text queries", file=sys.stderr)
                return 1
            
            client = OpenAI(api_key=api_key)
            results = faiss_index.search_by_text(args.query_text, client, k=args.k)
            print(f"\nSearch results for: '{args.query_text}'\n")
        
        elif args.query_embedding:
            # Embedding-based search
            embedding = [float(x) for x in args.query_embedding.split(",")]
            results = faiss_index.search(embedding, k=args.k)
            print(f"\nSearch results for provided embedding\n")
        
        else:
            print("Error: Must provide either --query-text or --query-embedding", file=sys.stderr)
            return 1
        
        # Display results
        for result in results:
            print(f"Rank {result['rank']}: Score = {result['score']:.4f}")
            print(f"  Entry ID: {result['entry_id']}")
            meta = result['metadata']
            if 'title' in meta:
                print(f"  Title: {meta['title']}")
            if 'link' in meta:
                print(f"  Link: {meta['link']}")
            print()
        
        return 0
    
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

