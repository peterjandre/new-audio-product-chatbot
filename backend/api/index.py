"""
Absolute minimal FastAPI handler for Vercel - everything in one file.
"""

from fastapi import FastAPI
from mangum import Mangum

# Create FastAPI app directly in this file
app = FastAPI(
    title="Test API",
    description="Minimal test",
    version="1.0.0",
)

@app.get("/")
async def root():
    return {"message": "FastAPI is working! This is a minimal test."}

@app.get("/health")
async def health():
    return {"status": "healthy", "message": "FastAPI is running successfully"}

# Create Mangum handler for Vercel
handler = Mangum(app)
