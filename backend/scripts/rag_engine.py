"""
RAG (Retrieval-Augmented Generation) engine.

This module provides a complete RAG implementation that:
1. Retrieves relevant context from a FAISS index
2. Generates responses using OpenAI
3. Returns answers with source citations
"""

import os
import sys
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Import FAISSIndex - handle both relative and absolute imports
try:
    from query_faiss_index import FAISSIndex
except ImportError:
    # If running from project root, try scripts module
    try:
        from scripts.query_faiss_index import FAISSIndex
    except ImportError:
        # Last resort: add scripts to path
        import sys
        scripts_dir = Path(__file__).parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from query_faiss_index import FAISSIndex


class RAGEngine:
    """
    RAG engine that combines FAISS retrieval with LLM generation.
    Uses OpenAI for both embeddings and generation.
    """
    
    def __init__(
        self,
        index_path: str | Path,
        metadata_path: str | Path | None = None,
        corpus_path: str | Path | None = None,
        openai_api_key: str | None = None,
        embedding_model: str = "text-embedding-3-small",
        openai_model: str = "gpt-3.5-turbo",
    ):
        """
        Initialize the RAG engine.
        
        Parameters
        ----------
        index_path:
            Path to FAISS index file
        metadata_path:
            Path to metadata JSON file (default: auto-detected)
        corpus_path:
            Path to original corpus JSON file (optional, for full text content)
        openai_api_key:
            OpenAI API key (default: from OPENAI_API_KEY env var)
        embedding_model:
            OpenAI embedding model to use
        openai_model:
            OpenAI model to use for generation (default: "gpt-3.5-turbo")
        """
        # Load FAISS index
        self.faiss_index = FAISSIndex(index_path, metadata_path)
        
        # Optionally load corpus for full text content
        self.corpus = None
        if corpus_path:
            corpus_path = Path(corpus_path)
            if corpus_path.exists():
                import json
                with corpus_path.open("r", encoding="utf-8") as f:
                    corpus_data = json.load(f)
                    # Create a lookup dict by id
                    if isinstance(corpus_data, list):
                        self.corpus = {entry.get("id"): entry for entry in corpus_data if entry.get("id")}
                        print(f"Loaded corpus with {len(self.corpus)} entries")
                    else:
                        print("Warning: Corpus file is not a list, skipping corpus loading", file=sys.stderr)
        
        # Configure OpenAI for embeddings and generation
        if OpenAI is None:
            raise RuntimeError(
                "openai package not available. Install with: pip install openai"
            )
        
        openai_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
        self.openai_client = OpenAI(api_key=openai_key)
        self.embedding_model = embedding_model
        self.openai_model = openai_model
        self.generation_model_name = openai_model
        
        print(f"RAG Engine initialized")
        print(f"  - FAISS index: {self.faiss_index.index.ntotal} vectors")
        print(f"  - Embedding model: {embedding_model}")
        print(f"  - Generation model: {self.generation_model_name}")
    
    def _format_context(self, retrieved_chunks: list[dict]) -> str:
        """
        Format retrieved chunks into context string for the LLM.
        
        Parameters
        ----------
        retrieved_chunks:
            List of retrieved chunk dictionaries from FAISS search
        
        Returns
        -------
        Formatted context string
        """
        if not retrieved_chunks:
            return "No relevant context found."
        
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            metadata = chunk.get("metadata", {})
            entry_id = chunk.get("entry_id", "unknown")
            
            # Determine source information
            source_title = metadata.get("source_title") or metadata.get("title", "")
            source_link = metadata.get("source_link") or metadata.get("link", "")
            source_id = metadata.get("source_id") or entry_id.split("::")[0] if "::" in entry_id else entry_id
            
            # Try to get full text content
            text = metadata.get("text", "")
            
            # If no text in metadata, try to get from corpus
            if not text and self.corpus:
                corpus_entry = self.corpus.get(source_id)
                if corpus_entry:
                    # Try to reconstruct text from corpus entry
                    summary = corpus_entry.get("summary") or corpus_entry.get("summary_generated", "")
                    title = corpus_entry.get("title", "")
                    if summary:
                        text = f"{title}\n\n{summary}" if title else summary
                    elif title:
                        text = title
            
            # Final fallback to title
            if not text:
                text = source_title or "No content available"
            
            # Build context entry
            context_entry = f"[Source {i}]"
            if source_title:
                context_entry += f" {source_title}"
            if source_link:
                context_entry += f" ({source_link})"
            context_entry += f"\n{text}\n"
            
            context_parts.append(context_entry)
        
        return "\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """
        Build the prompt with query and retrieved context.
        
        Parameters
        ----------
        query:
            User's question
        context:
            Formatted context from retrieved chunks
        
        Returns
        -------
        Complete prompt string
        """
        prompt = f"""You are a helpful assistant that answers questions about audio production gear and equipment based on information from Gearspace.com.

Use the following context to answer the user's question. If the context doesn't contain enough information to answer the question, say so. Be concise and accurate.

Context from Gearspace:
{context}

User Question: {query}

Answer:"""
        return prompt
    
    def query(
        self,
        question: str,
        k: int = 5,
        temperature: float = 0.3,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """
        Answer a question using RAG (retrieve + generate).
        
        Parameters
        ----------
        question:
            User's question
        k:
            Number of chunks to retrieve (default: 5)
        temperature:
            Temperature for generation (default: 0.3)
        max_retries:
            Maximum number of retries for API calls (default: 3)
        
        Returns
        -------
        Dictionary with keys:
            - answer: Generated answer string
            - sources: List of source dictionaries with metadata
            - retrieved_chunks: Raw retrieved chunks
        """
        # Step 1: Generate embedding for the question
        try:
            embedding_response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=[question],
            )
            query_embedding = embedding_response.data[0].embedding
        except Exception as exc:
            raise RuntimeError(f"Failed to generate embedding: {exc}") from exc
        
        # Step 2: Retrieve relevant chunks from FAISS
        retrieved_chunks = self.faiss_index.search(query_embedding, k=k)
        
        if not retrieved_chunks:
            return {
                "answer": "I couldn't find any relevant information in the knowledge base to answer your question.",
                "sources": [],
                "retrieved_chunks": [],
            }
        
        # Step 3: Format context
        context = self._format_context(retrieved_chunks)
        
        # Step 4: Build prompt
        prompt = self._build_prompt(question, context)
        
        # Step 5: Generate answer using OpenAI
        attempt = 0
        while attempt < max_retries:
            attempt += 1
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions about audio production gear and equipment based on information from Gearspace.com."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                )
                answer = response.choices[0].message.content.strip()
                
                if not answer:
                    raise RuntimeError("Empty response from OpenAI")
                
                # Extract sources from retrieved chunks
                sources = []
                seen_links = set()
                for chunk in retrieved_chunks:
                    metadata = chunk.get("metadata", {})
                    link = metadata.get("source_link") or metadata.get("link")
                    title = metadata.get("source_title") or metadata.get("title")
                    
                    # Use score from FAISS (higher is better for cosine similarity)
                    # Convert distance to similarity score if needed
                    score = chunk.get("score", 0.0)
                    
                    if link and link not in seen_links:
                        sources.append({
                            "title": title or "Untitled",
                            "link": link,
                            "score": float(score),
                        })
                        seen_links.add(link)
                
                return {
                    "answer": answer,
                    "sources": sources,
                    "retrieved_chunks": retrieved_chunks,
                }
            
            except Exception as exc:
                if attempt >= max_retries:
                    raise RuntimeError(
                        f"Failed to generate answer after {max_retries} attempts: {exc}"
                    ) from exc
                # Exponential backoff
                import time
                time.sleep(2 ** attempt)
        
        # Should never reach here, but just in case
        raise RuntimeError("Failed to generate answer")


def main():
    """CLI interface for testing the RAG engine."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Query the RAG engine with a question."
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Question to ask",
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
        "--corpus",
        default="data/gearspace_corpus.json",
        help="Path to corpus JSON file for full text content (default: data/gearspace_corpus.json)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Temperature for generation (default: 0.3)",
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-3.5-turbo",
        help="OpenAI model to use (default: gpt-3.5-turbo)",
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize RAG engine
        engine = RAGEngine(
            index_path=args.index,
            metadata_path=args.metadata,
            corpus_path=args.corpus,
            openai_model=args.openai_model,
        )
        
        # Query
        print(f"\nQuestion: {args.question}\n")
        print("Thinking...\n")
        
        result = engine.query(
            question=args.question,
            k=args.k,
            temperature=args.temperature,
        )
        
        # Display answer
        print("Answer:")
        print("-" * 80)
        print(result["answer"])
        print("-" * 80)
        
        # Display sources
        if result["sources"]:
            print(f"\nSources ({len(result['sources'])}):")
            for i, source in enumerate(result["sources"], start=1):
                print(f"  {i}. {source['title']}")
                print(f"     {source['link']}")
                print(f"     (Relevance score: {source['score']:.4f})")
        
        return 0
    
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

