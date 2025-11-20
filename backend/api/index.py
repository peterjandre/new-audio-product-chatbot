"""
Minimal Vercel serverless function wrapper for FastAPI app.
"""

import sys
import traceback
from pathlib import Path

def log(msg):
    """Log to stderr (captured by Vercel function logs)"""
    print(msg, file=sys.stderr, flush=True)
    print(msg, file=sys.stdout, flush=True)

log("=" * 80)
log(">>> MODULE LOADING STARTED <<<")
log("=" * 80)

handler = None

try:
    log(f"Python version: {sys.version}")
    log(f"Python path: {sys.path[:5]}")
    
    # Check dependencies
    log("\nChecking dependencies...")
    try:
        import fastapi
        log(f"✓ fastapi - version: {fastapi.__version__}")
    except ImportError as e:
        log(f"✗ fastapi - MISSING: {e}")
        raise
    
    try:
        import mangum
        version = getattr(mangum, '__version__', 'unknown')
        log(f"✓ mangum - version: {version}")
    except ImportError as e:
        log(f"✗ mangum - MISSING: {e}")
        raise
    
    # Add backend directory to path
    log("\nSetting up paths...")
    current_file = Path(__file__).resolve()
    backend_dir = current_file.parent.parent.resolve()
    log(f"Current file: {current_file}")
    log(f"Backend directory: {backend_dir}")
    log(f"Backend directory exists: {backend_dir.exists()}")
    
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
        log(f"Added {backend_dir} to Python path")
    
    # Import the app
    log("\nImporting app...")
    try:
        from app import app
        log("✓ Successfully imported app")
        log(f"App type: {type(app)}")
    except Exception as e:
        log(f"✗ Failed to import app: {e}")
        traceback.print_exc(file=sys.stderr)
        raise
    
    # Create Mangum handler
    log("\nCreating Mangum handler...")
    try:
        from mangum import Mangum
        handler = Mangum(app)
        log("✓ Mangum handler created successfully")
        log(f"Handler type: {type(handler)}")
    except Exception as e:
        log(f"✗ Failed to create Mangum handler: {e}")
        traceback.print_exc(file=sys.stderr)
        raise
    
    log("=" * 80)
    log("Module initialization complete!")
    log("=" * 80)

except Exception as e:
    log("\n" + "=" * 80)
    log("ERROR during module initialization:")
    log("=" * 80)
    log(f"Error type: {type(e).__name__}")
    log(f"Error message: {str(e)}")
    log("\nFull traceback:")
    traceback.print_exc(file=sys.stderr)
    traceback.print_exc(file=sys.stdout)
    log("=" * 80)
    
    # Try to create a fallback handler
    try:
        from fastapi import FastAPI
        from mangum import Mangum
        
        error_app = FastAPI()
        
        @error_app.get("/{path:path}")
        @error_app.post("/{path:path}")
        async def error_handler():
            return {
                "error": "Module initialization failed",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "message": "Check Vercel logs for details"
            }
        
        handler = Mangum(error_app)
        log("Created fallback error handler")
    except Exception as handler_error:
        log(f"Could not create fallback handler: {handler_error}")
        traceback.print_exc(file=sys.stderr)
        raise

# Ensure handler is defined
if handler is None:
    log("CRITICAL: handler is None - attempting emergency handler")
    try:
        from fastapi import FastAPI
        from mangum import Mangum
        emergency_app = FastAPI()
        @emergency_app.get("/{path:path}")
        @emergency_app.post("/{path:path}")
        async def emergency_handler():
            return {"error": "Handler initialization failed", "message": "Check logs"}
        handler = Mangum(emergency_app)
        log("Created emergency fallback handler")
    except Exception as e:
        log(f"Exception creating emergency handler: {e}")
        traceback.print_exc(file=sys.stderr)
        raise RuntimeError(f"CRITICAL: Handler is None. Last error: {e}")

log(f"SUCCESS: handler is defined: {type(handler)}")
if callable(handler):
    log("✓ Handler is callable")
else:
    log("✗ WARNING: Handler is not callable!")
