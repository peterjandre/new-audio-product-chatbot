# Test if file is being executed at all
print("FILE IS BEING EXECUTED", flush=True)
import sys
sys.stderr.write("STDERR: FILE IS BEING EXECUTED\n")
sys.stderr.flush()

"""
Absolute minimal FastAPI handler for Vercel - no Mangum.
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
    
    # Vercel's @vercel/python should handle FastAPI apps automatically
    # If it needs a handler, we'll add one, but try without first
    log("✓ App ready for Vercel")
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
