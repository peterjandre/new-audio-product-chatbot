"""
FastAPI web application for the RAG-powered audio products chatbot.

This API provides endpoints for querying the RAG engine and retrieving
information about audio production gear from Gearspace.com.
"""

import os
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
import urllib.parse

import requests

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add scripts directory to path for imports
scripts_dir = Path(__file__).parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from rag_engine import RAGEngine


# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Request model for querying the RAG engine."""
    question: str = Field(..., description="The question to ask about audio products")
    k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve (1-20)")
    temperature: float = Field(0.3, ge=0.0, le=2.0, description="Temperature for generation (0.0-2.0)")


class Source(BaseModel):
    """Source citation model."""
    title: str
    link: str
    score: float


class QueryResponse(BaseModel):
    """Response model for query results."""
    answer: str = Field(..., description="Generated answer to the question")
    sources: list[Source] = Field(..., description="List of source citations")
    question: str = Field(..., description="The original question")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    index_size: Optional[int] = None
    generation_provider: Optional[str] = None


# Global RAG engine instance (initialized on startup)
rag_engine: Optional[RAGEngine] = None

# Temporary directory for downloaded files (cleaned up on shutdown)
_temp_dir: Optional[tempfile.TemporaryDirectory] = None


def _is_url(path: str) -> bool:
    """Check if a path is a URL."""
    parsed = urllib.parse.urlparse(path)
    return parsed.scheme in ('http', 'https')


def _download_file(url: str, dest_path: Path) -> Path:
    """Download a file from a URL to a destination path."""
    print(f"Downloading {url} to {dest_path}...")
    response = requests.get(url, timeout=300)  # 5 minute timeout for large files
    response.raise_for_status()
    
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with dest_path.open('wb') as f:
        f.write(response.content)
    
    print(f"Downloaded {dest_path.name} ({len(response.content) / 1024 / 1024:.2f} MB)")
    return dest_path


def _resolve_path(path_str: str, temp_dir: Path) -> Path:
    """
    Resolve a path string that can be either a URL or local path.
    If it's a URL, download it to temp_dir. Otherwise, return as Path.
    """
    if not path_str:
        return None
    
    if _is_url(path_str):
        # It's a URL - download it
        filename = Path(urllib.parse.urlparse(path_str).path).name
        dest_path = temp_dir / filename
        if not dest_path.exists():
            _download_file(path_str, dest_path)
        return dest_path
    else:
        # It's a local path
        return Path(path_str)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    global rag_engine, _temp_dir
    
    # Startup
    try:
        # Create temporary directory for downloaded files (if using Supabase URLs)
        _temp_dir = tempfile.TemporaryDirectory()
        temp_dir_path = Path(_temp_dir.name)
        print(f"Created temporary directory: {temp_dir_path}")
        
        # Get configuration from environment variables
        # Support both local paths and Supabase URLs
        base_dir = Path(__file__).parent
        
        # Check for Supabase URLs first, then fall back to local paths
        index_path_env = os.getenv("FAISS_INDEX_URL") or os.getenv("FAISS_INDEX_PATH")
        if not index_path_env:
            index_path_env = str(base_dir / "data" / "faiss_index.index")
        
        metadata_path_env = os.getenv("FAISS_METADATA_URL") or os.getenv("FAISS_METADATA_PATH")
        if not metadata_path_env:
            metadata_path_env = str(base_dir / "data" / "faiss_index_metadata.json")
        
        corpus_path_env = os.getenv("CORPUS_URL") or os.getenv("CORPUS_PATH")
        if not corpus_path_env:
            corpus_path_env = str(base_dir / "data" / "gearspace_corpus.json")
        
        # Resolve paths (download if URLs, use local if paths)
        index_path = _resolve_path(index_path_env, temp_dir_path)
        
        if index_path is None or not index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found at {index_path_env}")
        
        # Resolve metadata path (auto-detect if not provided and using URL)
        metadata_path = None
        if metadata_path_env:
            metadata_path = _resolve_path(metadata_path_env, temp_dir_path)
        else:
            # Auto-detect metadata from index path
            auto_metadata = index_path.parent / f"{index_path.stem}_metadata.json"
            if auto_metadata.exists():
                metadata_path = auto_metadata
            else:
                # If index was downloaded from URL, try downloading metadata with similar name
                if _is_url(index_path_env):
                    # Try constructing metadata URL
                    metadata_url = index_path_env.replace('.index', '_metadata.json')
                    try:
                        metadata_path = _resolve_path(metadata_url, temp_dir_path)
                    except Exception as e:
                        print(f"Warning: Could not auto-detect metadata file: {e}")
        
        # Resolve corpus path
        corpus_path = None
        if corpus_path_env:
            corpus_path = _resolve_path(corpus_path_env, temp_dir_path)
        
        openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        
        print("Initializing RAG engine...")
        print(f"  Index path: {index_path}")
        print(f"  Metadata path: {metadata_path}")
        print(f"  Corpus path: {corpus_path}")
        
        rag_engine = RAGEngine(
            index_path=str(index_path),
            metadata_path=str(metadata_path) if metadata_path else None,
            corpus_path=str(corpus_path) if corpus_path else None,
            openai_model=openai_model,
        )
        print("RAG engine initialized successfully!")
        
    except Exception as e:
        print(f"Error initializing RAG engine: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise
    
    yield
    
    # Shutdown - clean up temporary directory
    if _temp_dir:
        print("Cleaning up temporary files...")
        _temp_dir.cleanup()
    print("Shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title="Audio Products Chatbot API",
    description="RAG-powered API for querying information about audio production gear",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def api_info():
    """API information endpoint."""
    return {
        "name": "Audio Products Chatbot API",
        "version": "1.0.0",
        "description": "RAG-powered API for querying information about audio production gear",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    global rag_engine
    
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    return HealthResponse(
        status="healthy",
        index_size=rag_engine.faiss_index.index.ntotal if rag_engine else None,
        generation_provider="openai",  # RAG engine uses OpenAI
    )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(request: QueryRequest):
    """
    Query the RAG engine with a question about audio products.
    
    This endpoint:
    1. Generates an embedding for the question
    2. Retrieves relevant chunks from the FAISS index
    3. Generates an answer using the selected LLM provider
    4. Returns the answer with source citations
    """
    global rag_engine
    
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        # Query the RAG engine
        result = rag_engine.query(
            question=request.question,
            k=request.k,
            temperature=request.temperature,
        )
        
        # Convert sources to response model
        sources = [
            Source(title=src["title"], link=src["link"], score=src["score"])
            for src in result["sources"]
        ]
        
        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            question=request.question,
        )
    
    except Exception as e:
        print(f"Error processing query: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=True,  # Enable auto-reload for development
    )

