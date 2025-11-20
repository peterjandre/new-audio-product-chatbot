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
    try:
        print(f"Downloading {url} to {dest_path}...", flush=True)
        print(f"Dest path parent: {dest_path.parent}", flush=True)
        print(f"Dest path parent exists: {dest_path.parent.exists()}", flush=True)
        
        response = requests.get(url, timeout=300, stream=True)  # Stream for large files
        response.raise_for_status()
        
        # Ensure parent directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Created parent directory: {dest_path.parent}", flush=True)
        
        # Write file in chunks to handle large files
        total_size = 0
        with dest_path.open('wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)
        
        file_size_mb = total_size / 1024 / 1024
        print(f"Downloaded {dest_path.name} ({file_size_mb:.2f} MB)", flush=True)
        print(f"File exists after download: {dest_path.exists()}", flush=True)
        print(f"File size: {dest_path.stat().st_size} bytes", flush=True)
        
        return dest_path
    except Exception as e:
        error_msg = f"Error downloading {url}: {e}"
        print(error_msg, file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise RuntimeError(error_msg) from e


def _resolve_path(path_str: str, temp_dir: Path) -> Path:
    """
    Resolve a path string that can be either a URL or local path.
    If it's a URL, download it to temp_dir. Otherwise, return as Path.
    """
    if not path_str:
        print(f"Warning: Empty path string provided", flush=True)
        return None
    
    try:
        if _is_url(path_str):
            # It's a URL - download it
            print(f"Resolving URL: {path_str}", flush=True)
            filename = Path(urllib.parse.urlparse(path_str).path).name
            if not filename:
                raise ValueError(f"Could not extract filename from URL: {path_str}")
            
            dest_path = temp_dir / filename
            print(f"Destination path: {dest_path}", flush=True)
            
            if not dest_path.exists():
                print(f"File doesn't exist, downloading...", flush=True)
                _download_file(path_str, dest_path)
            else:
                file_size = dest_path.stat().st_size
                print(f"File already exists: {dest_path} ({file_size} bytes)", flush=True)
            
            if not dest_path.exists():
                raise FileNotFoundError(f"Downloaded file not found at {dest_path}")
            
            return dest_path
        else:
            # It's a local path
            print(f"Resolving local path: {path_str}", flush=True)
            local_path = Path(path_str)
            if not local_path.exists():
                raise FileNotFoundError(f"Local file not found at {path_str}")
            return local_path
    except Exception as e:
        error_msg = f"Error resolving path '{path_str}': {e}"
        print(error_msg, file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise RuntimeError(error_msg) from e


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    global rag_engine, _temp_dir
    
    # Startup
    try:
        print("=" * 80, flush=True)
        print("Starting RAG engine initialization...", flush=True)
        print("=" * 80, flush=True)
        
        # Create temporary directory for downloaded files (if using Supabase URLs)
        _temp_dir = tempfile.TemporaryDirectory()
        temp_dir_path = Path(_temp_dir.name)
        print(f"Created temporary directory: {temp_dir_path}", flush=True)
        print(f"Temp directory exists: {temp_dir_path.exists()}", flush=True)
        print(f"Temp directory writable: {os.access(temp_dir_path, os.W_OK)}", flush=True)
        
        # Get configuration from environment variables
        # Support both local paths and Supabase URLs
        base_dir = Path(__file__).parent
        print(f"Base directory: {base_dir}", flush=True)
        
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
        
        print(f"Environment variables:", flush=True)
        print(f"  FAISS_INDEX_URL: {os.getenv('FAISS_INDEX_URL', 'not set')}", flush=True)
        print(f"  FAISS_INDEX_PATH: {os.getenv('FAISS_INDEX_PATH', 'not set')}", flush=True)
        print(f"  Index path to use: {index_path_env}", flush=True)
        print(f"  Metadata path to use: {metadata_path_env}", flush=True)
        print(f"  Corpus path to use: {corpus_path_env}", flush=True)
        
        # Resolve paths (download if URLs, use local if paths)
        print("\nStep 1: Resolving index path...", flush=True)
        index_path = _resolve_path(index_path_env, temp_dir_path)
        
        if index_path is None:
            raise FileNotFoundError(f"FAISS index path is None (from: {index_path_env})")
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found at {index_path} (resolved from: {index_path_env})")
        
        print(f"✓ Index path resolved: {index_path} ({index_path.stat().st_size} bytes)", flush=True)
        
        # Resolve metadata path (auto-detect if not provided and using URL)
        print("\nStep 2: Resolving metadata path...", flush=True)
        metadata_path = None
        if metadata_path_env:
            metadata_path = _resolve_path(metadata_path_env, temp_dir_path)
            if metadata_path and metadata_path.exists():
                print(f"✓ Metadata path resolved: {metadata_path} ({metadata_path.stat().st_size} bytes)", flush=True)
        else:
            # Auto-detect metadata from index path
            auto_metadata = index_path.parent / f"{index_path.stem}_metadata.json"
            if auto_metadata.exists():
                metadata_path = auto_metadata
                print(f"✓ Auto-detected metadata: {metadata_path}", flush=True)
            else:
                # If index was downloaded from URL, try downloading metadata with similar name
                if _is_url(index_path_env):
                    # Try constructing metadata URL
                    metadata_url = index_path_env.replace('.index', '_metadata.json')
                    try:
                        metadata_path = _resolve_path(metadata_url, temp_dir_path)
                        if metadata_path and metadata_path.exists():
                            print(f"✓ Auto-downloaded metadata: {metadata_path}", flush=True)
                    except Exception as e:
                        print(f"⚠ Warning: Could not auto-detect metadata file: {e}", flush=True)
        
        # Resolve corpus path
        print("\nStep 3: Resolving corpus path...", flush=True)
        corpus_path = None
        if corpus_path_env:
            corpus_path = _resolve_path(corpus_path_env, temp_dir_path)
            if corpus_path and corpus_path.exists():
                print(f"✓ Corpus path resolved: {corpus_path} ({corpus_path.stat().st_size} bytes)", flush=True)
        
        # Check OpenAI API key
        print("\nStep 4: Checking OpenAI API key...", flush=True)
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set")
        print(f"✓ OpenAI API key found (length: {len(openai_key)})", flush=True)
        
        openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        print(f"  Using model: {openai_model}", flush=True)
        
        # Initialize RAG engine
        print("\nStep 5: Initializing RAG engine...", flush=True)
        print(f"  Index path: {index_path}", flush=True)
        print(f"  Metadata path: {metadata_path}", flush=True)
        print(f"  Corpus path: {corpus_path}", flush=True)
        
        rag_engine = RAGEngine(
            index_path=str(index_path),
            metadata_path=str(metadata_path) if metadata_path else None,
            corpus_path=str(corpus_path) if corpus_path else None,
            openai_model=openai_model,
        )
        
        print("=" * 80, flush=True)
        print("✓ RAG engine initialized successfully!", flush=True)
        print("=" * 80, flush=True)
        
    except Exception as e:
        error_msg = f"Error initializing RAG engine: {e}"
        print("\n" + "=" * 80, file=sys.stderr, flush=True)
        print("ERROR:", file=sys.stderr, flush=True)
        print(error_msg, file=sys.stderr, flush=True)
        print("=" * 80, file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        # Don't re-raise, let the error propagate naturally
        raise RuntimeError(error_msg) from e
    
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

