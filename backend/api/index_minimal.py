"""
Minimal Vercel serverless function for testing deployment.
"""
from mangum import Mangum
from fastapi import FastAPI

# Create minimal FastAPI app
app = FastAPI(title="Minimal Test API")

@app.get("/")
async def root():
    return {"message": "Hello from Vercel!", "status": "ok"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

# Create Mangum handler for Vercel
handler = Mangum(app, lifespan="off")

