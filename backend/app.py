"""
Minimal FastAPI app for testing basic functionality.
"""
from fastapi import FastAPI

# Initialize FastAPI app (no lifespan, no middleware)
app = FastAPI(
    title="Audio Products Chatbot API",
    description="Minimal test version",
    version="1.0.0",
)


@app.get("/")
async def root():
    """Simple root endpoint."""
    return {
        "message": "Hello from minimal FastAPI app",
        "status": "working"
    }


@app.get("/health")
async def health():
    """Simple health check."""
    return {"status": "healthy"}

