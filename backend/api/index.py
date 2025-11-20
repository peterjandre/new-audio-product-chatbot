"""
Vercel serverless function wrapper for FastAPI app.
"""
# CRITICAL: Create a minimal handler FIRST before any imports that might fail
# This ensures Vercel can always import the module and find a handler
handler = None

# Try to create fallback handler immediately - wrap in try/except to ensure module loads
try:
    from fastapi import FastAPI
    from mangum import Mangum
    _fallback_app = FastAPI()
    @_fallback_app.get("/{path:path}")
    @_fallback_app.post("/{path:path}")
    async def _fallback_handler():
        return {"status": "initializing", "message": "Check logs for initialization status"}
    handler = Mangum(_fallback_app)
except Exception as e:
    # If we can't even create fallback, handler stays None but module still loads
    import sys
    print(f"WARNING: Could not create fallback handler: {e}", file=sys.stderr, flush=True)
    handler = None

# Now do imports - if these fail, handler stays None but module still loads
import sys
import traceback

def log(msg):
    """Log to stderr (captured by Vercel function logs)"""
    print(msg, file=sys.stderr, flush=True)
    print(msg, file=sys.stdout, flush=True)

# Log immediately
log("=" * 80)
log(">>> MODULE LOADING STARTED <<<")
log("=" * 80)

try:
    log(f"Python version: {sys.version}")
    
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
    from pathlib import Path
    import os
    
    # Get the backend directory - try multiple approaches for Vercel compatibility
    current_file = Path(__file__).resolve()
    backend_dir = current_file.parent.parent.resolve()
    
    # Also try getting from current working directory
    cwd = Path(os.getcwd()).resolve()
    log(f"\nCurrent working directory: {cwd}")
    log(f"Current file: {current_file}")
    log(f"Backend directory (from __file__): {backend_dir}")
    log(f"Backend directory exists: {backend_dir.exists()}")
    
    # Add both possible paths
    paths_to_add = [str(backend_dir)]
    if str(cwd) not in sys.path and cwd.exists():
        paths_to_add.append(str(cwd))
    
    for path_str in paths_to_add:
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
            log(f"Added {path_str} to Python path")
    
    log(f"Python path: {sys.path[:5]}")  # Log first 5 entries
    
    # Try importing app
    log("\n" + "=" * 80)
    log("Importing app...")
    log("=" * 80)
    
    try:
        from app import app
        log("✓ Successfully imported app")
        log(f"App type: {type(app)}")
    except Exception as e:
        log(f"✗ Failed to import app: {e}")
        traceback.print_exc(file=sys.stderr)
        raise
    
    # Create Mangum handler
    log("\n" + "=" * 80)
    log("Creating Mangum handler...")
    log("=" * 80)
    
    try:
        from mangum import Mangum
        handler = Mangum(app)
        log("✓ Mangum handler created successfully")
        log(f"Handler type: {type(handler)}")
        log("=" * 80)
        log("Module initialization complete!")
        log("=" * 80)
    except Exception as e:
        log(f"✗ Failed to create Mangum handler: {e}")
        traceback.print_exc(file=sys.stderr)
        raise

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
    
    # Create a fallback handler so module still works
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
                "error_message": str(e)
            }
        
        handler = Mangum(error_app)
        log("Created fallback error handler")
    except Exception as handler_error:
        log(f"Could not create fallback handler: {handler_error}")
        # handler stays None - Vercel will fail but we have logs
    
    log("=" * 80)

# Ensure handler is always valid
if handler is None:
    log("CRITICAL: handler is None - trying to create emergency handler")
    try:
        from fastapi import FastAPI
        from mangum import Mangum
        emergency_app = FastAPI()
        @emergency_app.get("/{path:path}")
        @emergency_app.post("/{path:path}")
        async def emergency_handler():
            return {"error": "Handler initialization failed", "message": "Check logs"}
        handler = Mangum(emergency_app)
        if handler:
            log("Created emergency fallback handler")
        else:
            log("FAILED to create emergency handler")
    except Exception as e:
        log(f"Exception creating emergency handler: {e}")

if handler:
    log(f"SUCCESS: handler is defined: {type(handler)}")
    # Verify handler is callable (required by Vercel)
    if callable(handler):
        log("✓ Handler is callable")
    else:
        log("✗ WARNING: Handler is not callable!")
else:
    log("ERROR: handler is still None - Vercel deployment will fail")

# CRITICAL: Ensure handler is exported and accessible to Vercel
# Vercel looks for a 'handler' variable at module level
# This must be the last line to ensure handler is always defined
if handler is None:
    raise RuntimeError("CRITICAL: Handler is None - Vercel deployment will fail. Check logs above for initialization errors.")
