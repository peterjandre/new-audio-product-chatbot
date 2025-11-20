"""
Minimal FastAPI app for testing.
"""

from fastapi import FastAPI

app = FastAPI(
    title="Audio Products Chatbot API",
    description="Minimal test API",
    version="1.0.0",
)


@app.get("/")
async def root():
    """Simple root endpoint."""
    return {"message": "FastAPI is working! This is a minimal test."}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "message": "FastAPI is running successfully"}
