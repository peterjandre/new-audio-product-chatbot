"""
Vercel serverless function wrapper for FastAPI app.
Minimal version for testing.
"""
import sys
import traceback
from pathlib import Path

def log(msg):
    """Log to stderr (captured by Vercel function logs)"""
    print(msg, file=sys.stderr, flush=True)

log("=" * 80)
log("Starting api/index.py - Minimal Test Version")
log("=" * 80)

# Check Python version
log(f"Python version: {sys.version}")

# Check only essential dependencies for minimal app
log("\nChecking essential dependencies...")
try:
    import fastapi
    log(f"✓ fastapi - version: {fastapi.__version__}")
except ImportError as e:
    log(f"✗ fastapi - MISSING: {e}")
    sys.exit(1)

try:
    import mangum
    log(f"✓ mangum - version: {mangum.__version__}")
except ImportError as e:
    log(f"✗ mangum - MISSING: {e}")
    sys.exit(1)

log("=" * 80)

# Add parent directory to path to import app
backend_dir = Path(__file__).parent.parent
log(f"Backend directory: {backend_dir}")
log(f"Backend directory exists: {backend_dir.exists()}")

if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
    log(f"Added {backend_dir} to Python path")

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
    sys.exit(1)

# Create Mangum handler
log("\n" + "=" * 80)
log("Creating Mangum handler...")
log("=" * 80)

try:
    from mangum import Mangum
    handler = Mangum(app)
    log("✓ Mangum handler created successfully")
    log("=" * 80)
except Exception as e:
    log(f"✗ Failed to create Mangum handler: {e}")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
