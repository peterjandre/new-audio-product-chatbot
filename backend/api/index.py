"""
Absolute minimal FastAPI handler for Vercel - everything in one file.
"""

import sys

# Log to stderr (visible in Vercel logs)
def log(msg):
    print(msg, file=sys.stderr, flush=True)
    print(msg, file=sys.stdout, flush=True)

log("=" * 80)
log("Starting module import...")

try:
    log("Importing FastAPI...")
    from fastapi import FastAPI
    log("✓ FastAPI imported")
    
    log("Importing Mangum...")
    from mangum import Mangum
    log("✓ Mangum imported")
    
    log("Creating FastAPI app...")
    app = FastAPI(
        title="Test API",
        description="Minimal test",
        version="1.0.0",
    )
    log("✓ FastAPI app created")
    
    log("Defining routes...")
    @app.get("/")
    async def root():
        return {"message": "FastAPI is working! This is a minimal test."}
    
    @app.get("/health")
    async def health():
        return {"status": "healthy", "message": "FastAPI is running successfully"}
    log("✓ Routes defined")
    
    log("Creating Mangum handler...")
    handler = Mangum(app)
    log("✓ Mangum handler created")
    log(f"Handler type: {type(handler)}")
    log(f"Handler is callable: {callable(handler)}")
    
    log("=" * 80)
    log("Module loaded successfully!")
    log("=" * 80)
    
except Exception as e:
    log("=" * 80)
    log("ERROR during module load:")
    log(f"Error type: {type(e).__name__}")
    log(f"Error message: {str(e)}")
    import traceback
    traceback.print_exc(file=sys.stderr)
    traceback.print_exc(file=sys.stdout)
    log("=" * 80)
    raise

# Ensure handler exists
if 'handler' not in globals():
    log("CRITICAL: handler not defined!")
    raise RuntimeError("Handler not defined")
