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


# Global RAG engine instance (initialized lazily on first request)
rag_engine: Optional[RAGEngine] = None
_rag_engine_lock = False  # Simple lock to prevent concurrent initialization
_rag_engine_error: Optional[str] = None  # Store initialization error

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


def _test_file_loading(file_type: str, file_path: Path) -> dict:
    """
    Test loading a single file with detailed error reporting.
    
    Returns dict with 'success', 'error', 'details' keys.
    """
    result = {
        "success": False,
        "error": None,
        "details": {}
    }
    
    try:
        print(f"\n{'='*80}", file=sys.stderr, flush=True)
        print(f"TESTING {file_type.upper()}: {file_path}", file=sys.stderr, flush=True)
        print(f"{'='*80}", file=sys.stderr, flush=True)
        
        # Step 1: Check if file exists
        print(f"Step 1: Checking if file exists...", file=sys.stderr, flush=True)
        if not file_path.exists():
            result["error"] = f"File does not exist: {file_path}"
            result["details"]["exists"] = False
            return result
        result["details"]["exists"] = True
        print(f"✓ File exists", file=sys.stderr, flush=True)
        
        # Step 2: Check file size
        print(f"Step 2: Checking file size...", file=sys.stderr, flush=True)
        file_size = file_path.stat().st_size
        result["details"]["size_bytes"] = file_size
        result["details"]["size_mb"] = file_size / (1024 * 1024)
        print(f"✓ File size: {file_size} bytes ({result['details']['size_mb']:.2f} MB)", file=sys.stderr, flush=True)
        
        # Step 3: Try to read file based on type
        if file_type == "index":
            print(f"Step 3: Loading FAISS index...", file=sys.stderr, flush=True)
            import faiss
            index = faiss.read_index(str(file_path))
            result["details"]["index_ntotal"] = index.ntotal
            result["details"]["index_dimension"] = index.d
            print(f"✓ FAISS index loaded: {index.ntotal} vectors, dimension {index.d}", file=sys.stderr, flush=True)
            
        elif file_type == "metadata":
            print(f"Step 3: Loading JSON metadata...", file=sys.stderr, flush=True)
            import json
            with file_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
            result["details"]["metadata_count"] = len(metadata) if isinstance(metadata, list) else "not a list"
            result["details"]["metadata_type"] = type(metadata).__name__
            print(f"✓ Metadata loaded: {result['details']['metadata_count']} entries", file=sys.stderr, flush=True)
            
        elif file_type == "corpus":
            print(f"Step 3: Loading JSON corpus...", file=sys.stderr, flush=True)
            import json
            with file_path.open("r", encoding="utf-8") as f:
                corpus = json.load(f)
            result["details"]["corpus_count"] = len(corpus) if isinstance(corpus, list) else "not a list"
            result["details"]["corpus_type"] = type(corpus).__name__
            if isinstance(corpus, list) and len(corpus) > 0:
                result["details"]["first_entry_keys"] = list(corpus[0].keys()) if isinstance(corpus[0], dict) else "not a dict"
            print(f"✓ Corpus loaded: {result['details']['corpus_count']} entries", file=sys.stderr, flush=True)
        
        result["success"] = True
        print(f"✓ {file_type.upper()} file test PASSED", file=sys.stderr, flush=True)
        
    except Exception as e:
        result["error"] = str(e)
        print(f"✗ {file_type.upper()} file test FAILED: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
    
    return result


def _initialize_rag_engine():
    """Initialize the RAG engine lazily (called on first request)."""
    global rag_engine, _rag_engine_lock, _rag_engine_error, _temp_dir
    
    # If already initialized, return
    if rag_engine is not None:
        return
    
    # If initialization failed before, raise error
    if _rag_engine_error:
        raise RuntimeError(f"RAG engine initialization failed: {_rag_engine_error}")
    
    # If currently initializing, wait a bit (simple lock)
    if _rag_engine_lock:
        import time
        time.sleep(0.1)
        if rag_engine is not None:
            return
        if _rag_engine_error:
            raise RuntimeError(f"RAG engine initialization failed: {_rag_engine_error}")
    
    # Set lock
    _rag_engine_lock = True
    
    try:
        print("\n" + "="*80, file=sys.stderr, flush=True)
        print("Initializing RAG engine (lazy load)...", file=sys.stderr, flush=True)
        print("="*80, file=sys.stderr, flush=True)
        
        # Create temporary directory for downloaded files (if using Supabase URLs)
        if _temp_dir is None:
            _temp_dir = tempfile.TemporaryDirectory()
        temp_dir_path = Path(_temp_dir.name)
        
        # Get configuration from environment variables
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
        
        # TEST EACH FILE INDIVIDUALLY
        print("\n" + "="*80, file=sys.stderr, flush=True)
        print("TESTING DATA FILES INDIVIDUALLY", file=sys.stderr, flush=True)
        print("="*80, file=sys.stderr, flush=True)
        
        # Test 1: Resolve and test index file
        print("\n[TEST 1] Testing FAISS index file...", file=sys.stderr, flush=True)
        index_path = None
        try:
            index_path = _resolve_path(index_path_env, temp_dir_path)
            if index_path is None or not index_path.exists():
                raise FileNotFoundError(f"FAISS index file not found at {index_path_env}")
            index_test = _test_file_loading("index", index_path)
            if not index_test["success"]:
                raise RuntimeError(f"Index file test failed: {index_test['error']}")
        except Exception as e:
            error_msg = f"Failed to load index file: {e}"
            print(f"✗ {error_msg}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)
            _rag_engine_error = error_msg
            raise RuntimeError(error_msg) from e
        
        # Test 2: Resolve and test metadata file
        print("\n[TEST 2] Testing metadata file...", file=sys.stderr, flush=True)
        metadata_path = None
        try:
            if metadata_path_env:
                metadata_path = _resolve_path(metadata_path_env, temp_dir_path)
            else:
                # Auto-detect metadata from index path
                auto_metadata = index_path.parent / f"{index_path.stem}_metadata.json"
                if auto_metadata.exists():
                    metadata_path = auto_metadata
                elif _is_url(index_path_env):
                    # Try constructing metadata URL
                    metadata_url = index_path_env.replace('.index', '_metadata.json')
                    try:
                        metadata_path = _resolve_path(metadata_url, temp_dir_path)
                    except Exception as e:
                        print(f"Warning: Could not auto-detect metadata: {e}", file=sys.stderr, flush=True)
            
            if metadata_path:
                metadata_test = _test_file_loading("metadata", metadata_path)
                if not metadata_test["success"]:
                    raise RuntimeError(f"Metadata file test failed: {metadata_test['error']}")
        except Exception as e:
            error_msg = f"Failed to load metadata file: {e}"
            print(f"✗ {error_msg}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)
            _rag_engine_error = error_msg
            raise RuntimeError(error_msg) from e
        
        # Test 3: Resolve and test corpus file (optional)
        print("\n[TEST 3] Testing corpus file (optional)...", file=sys.stderr, flush=True)
        corpus_path = None
        try:
            if corpus_path_env:
                corpus_path = _resolve_path(corpus_path_env, temp_dir_path)
                if corpus_path:
                    corpus_test = _test_file_loading("corpus", corpus_path)
                    if not corpus_test["success"]:
                        print(f"Warning: Corpus file test failed (continuing anyway): {corpus_test['error']}", file=sys.stderr, flush=True)
                        corpus_path = None  # Make it optional
        except Exception as e:
            print(f"Warning: Failed to load corpus file (continuing anyway): {e}", file=sys.stderr, flush=True)
            corpus_path = None  # Make it optional
        
        # Check OpenAI API key
        print("\n[TEST 4] Checking OpenAI API key...", file=sys.stderr, flush=True)
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            error_msg = "OPENAI_API_KEY environment variable is not set"
            print(f"✗ {error_msg}", file=sys.stderr, flush=True)
            _rag_engine_error = error_msg
            raise RuntimeError(error_msg)
        print(f"✓ OpenAI API key found", file=sys.stderr, flush=True)
        
        openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        
        # Initialize RAG engine (all files tested successfully)
        print("\n" + "="*80, file=sys.stderr, flush=True)
        print("All file tests passed. Initializing RAG engine...", file=sys.stderr, flush=True)
        print("="*80, file=sys.stderr, flush=True)
        
        rag_engine = RAGEngine(
            index_path=str(index_path),
            metadata_path=str(metadata_path) if metadata_path else None,
            corpus_path=str(corpus_path) if corpus_path else None,
            openai_model=openai_model,
        )
        
        print("✓ RAG engine initialized successfully!", file=sys.stderr, flush=True)
        
    except Exception as e:
        error_msg = f"Error initializing RAG engine: {e}"
        print("\n" + "="*80, file=sys.stderr, flush=True)
        print("ERROR:", file=sys.stderr, flush=True)
        print(error_msg, file=sys.stderr, flush=True)
        print("="*80, file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        _rag_engine_error = str(e)
        raise RuntimeError(error_msg) from e
    finally:
        _rag_engine_lock = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    global _temp_dir
    
    # Startup - minimal initialization (RAG engine will be initialized lazily on first request)
    print("FastAPI app starting...", file=sys.stderr, flush=True)
    
    yield
    
    # Shutdown - clean up temporary directory
    if _temp_dir:
        print("Cleaning up temporary files...", file=sys.stderr, flush=True)
        _temp_dir.cleanup()
    print("Shutting down...", file=sys.stderr, flush=True)


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


@app.get("/test-files", tags=["Debug"])
async def test_files():
    """
    Test endpoint to check each data file individually.
    Useful for debugging which file is causing issues.
    """
    results = {
        "index": None,
        "metadata": None,
        "corpus": None,
        "openai_key": None,
    }
    
    base_dir = Path(__file__).parent
    
    # Create temp directory
    if _temp_dir is None:
        global _temp_dir
        _temp_dir = tempfile.TemporaryDirectory()
    temp_dir_path = Path(_temp_dir.name)
    
    # Test index file
    try:
        index_path_env = os.getenv("FAISS_INDEX_URL") or os.getenv("FAISS_INDEX_PATH")
        if not index_path_env:
            index_path_env = str(base_dir / "data" / "faiss_index.index")
        index_path = _resolve_path(index_path_env, temp_dir_path)
        if index_path:
            results["index"] = _test_file_loading("index", index_path)
    except Exception as e:
        results["index"] = {"success": False, "error": str(e), "details": {}}
    
    # Test metadata file
    try:
        metadata_path_env = os.getenv("FAISS_METADATA_URL") or os.getenv("FAISS_METADATA_PATH")
        if not metadata_path_env:
            metadata_path_env = str(base_dir / "data" / "faiss_index_metadata.json")
        metadata_path = _resolve_path(metadata_path_env, temp_dir_path)
        if metadata_path:
            results["metadata"] = _test_file_loading("metadata", metadata_path)
    except Exception as e:
        results["metadata"] = {"success": False, "error": str(e), "details": {}}
    
    # Test corpus file
    try:
        corpus_path_env = os.getenv("CORPUS_URL") or os.getenv("CORPUS_PATH")
        if not corpus_path_env:
            corpus_path_env = str(base_dir / "data" / "gearspace_corpus.json")
        corpus_path = _resolve_path(corpus_path_env, temp_dir_path)
        if corpus_path:
            results["corpus"] = _test_file_loading("corpus", corpus_path)
    except Exception as e:
        results["corpus"] = {"success": False, "error": str(e), "details": {}}
    
    # Test OpenAI key
    openai_key = os.getenv("OPENAI_API_KEY")
    results["openai_key"] = {
        "success": openai_key is not None,
        "error": None if openai_key else "OPENAI_API_KEY not set",
        "details": {"key_length": len(openai_key) if openai_key else 0}
    }
    
    return results


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    global rag_engine, _rag_engine_error
    
    # Try to initialize if not already done
    if rag_engine is None:
        try:
            _initialize_rag_engine()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"RAG engine not available: {str(e)}"
            )
    
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
    
    # Initialize RAG engine if not already done (lazy initialization)
    if rag_engine is None:
        try:
            _initialize_rag_engine()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"RAG engine not available: {str(e)}"
            )
    
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

