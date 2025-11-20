"""
Vercel serverless function wrapper for FastAPI app.
"""
import sys
import os
from pathlib import Path

# Initialize handler to None - will be set below
handler = None

try:
    # Add parent directory to path to import app
    backend_dir = Path(__file__).parent.parent
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    # Add debug output before importing app
    print("=" * 80, flush=True)
    print("Vercel handler initialization starting...", flush=True)
    print(f"Backend directory: {backend_dir}", flush=True)
    print(f"Python path: {sys.path}", flush=True)
    print(f"Working directory: {os.getcwd()}", flush=True)
    print(f"Python version: {sys.version}", flush=True)
    print("=" * 80, flush=True)
except Exception as e:
    print(f"ERROR in initial setup: {e}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc(file=sys.stderr)

    # Try importing dependencies first
    print("\n[STEP 1] Testing basic imports...", flush=True)
    try:
        import json
        print("✓ json imported", flush=True)
    except Exception as e:
        print(f"✗ Error importing json: {e}", file=sys.stderr, flush=True)
        raise

    try:
        import tempfile
        print("✓ tempfile imported", flush=True)
    except Exception as e:
        print(f"✗ Error importing tempfile: {e}", file=sys.stderr, flush=True)
        raise

    try:
        from fastapi import FastAPI
        print("✓ fastapi imported", flush=True)
    except Exception as e:
        print(f"✗ Error importing fastapi: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise

    try:
        from mangum import Mangum
        print("✓ mangum imported", flush=True)
    except Exception as e:
        print(f"✗ Error importing mangum: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise

    print("\n[STEP 2] Importing app module...", flush=True)
    app = None
    try:
        from app import app
        print("✓ Successfully imported app", flush=True)
    except Exception as e:
        print(f"✗ Error importing app: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        # Don't raise here - let it fail when handler is called
        app = None

    print("\n[STEP 3] Creating Mangum handler...", flush=True)
    # Create Mangum handler for Vercel
    init_error = None

    try:
        if app is None:
            raise RuntimeError("App is None - cannot create handler")
        handler = Mangum(app, lifespan="off")  # Disable lifespan events for serverless
        print("✓ Successfully created Mangum handler", flush=True)
    except Exception as e:
        init_error = str(e)
        print(f"✗ Error creating Mangum handler: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        
        # Create a minimal error handler that matches Mangum's signature
        async def error_handler(event, context):
            import json
            return {
                "statusCode": 500,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({
                    "error": f"Handler initialization failed: {init_error}",
                    "message": "The server failed to initialize. Check logs for details."
                })
            }
        handler = error_handler

    print("\n" + "=" * 80, flush=True)
    print("Vercel handler initialization complete", flush=True)
    print(f"Handler type: {type(handler)}", flush=True)
    print("=" * 80 + "\n", flush=True)

except Exception as e:
    # Catch any unhandled exceptions during initialization
    error_msg = f"Fatal error during initialization: {e}"
    print(error_msg, file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc(file=sys.stderr)
    
    # Create a fallback error handler
    async def fallback_handler(event, context):
        import json
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "error": error_msg,
                "message": "The server failed to initialize. Check logs for details."
            })
        }
    handler = fallback_handler

# Ensure handler is always defined
if handler is None:
    async def final_fallback_handler(event, context):
        import json
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "error": "Handler was not properly initialized",
                "message": "Check server logs for details."
            })
        }
    handler = final_fallback_handler

