"""
FastAPI web application for the RAG-powered audio products chatbot.

This API provides endpoints for querying the RAG engine and retrieving
information about audio production gear from Gearspace.com.
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    global rag_engine
    
    # Startup
    try:
        # Get configuration from environment variables or use defaults
        # For Vercel, data files should be in the backend/data directory
        base_dir = Path(__file__).parent
        index_path = os.getenv("FAISS_INDEX_PATH", str(base_dir / "data" / "faiss_index.index"))
        metadata_path = os.getenv("FAISS_METADATA_PATH", None)
        if metadata_path is None:
            metadata_path = str(base_dir / "data" / "faiss_index_metadata.json")
            if not Path(metadata_path).exists():
                metadata_path = None
        corpus_path = os.getenv("CORPUS_PATH", str(base_dir / "data" / "gearspace_corpus.json"))
        openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        
        print("Initializing RAG engine...")
        rag_engine = RAGEngine(
            index_path=index_path,
            metadata_path=metadata_path,
            corpus_path=corpus_path,
            openai_model=openai_model,
        )
        print("RAG engine initialized successfully!")
        
    except Exception as e:
        print(f"Error initializing RAG engine: {e}", file=sys.stderr)
        raise
    
    yield
    
    # Shutdown (if needed, add cleanup code here)
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
        generation_provider=rag_engine.generation_provider if rag_engine else None,
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

