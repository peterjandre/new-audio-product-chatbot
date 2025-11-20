"""
Vercel serverless function wrapper for FastAPI app.
"""
from mangum import Mangum
import sys
import os
from pathlib import Path

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
print("=" * 80, flush=True)

try:
    from app import app
    print("✓ Successfully imported app", flush=True)
except Exception as e:
    print(f"✗ Error importing app: {e}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc(file=sys.stderr)
    raise

# Create Mangum handler for Vercel
try:
    handler = Mangum(app)
    print("✓ Successfully created Mangum handler", flush=True)
except Exception as e:
    print(f"✗ Error creating Mangum handler: {e}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc(file=sys.stderr)
    raise

