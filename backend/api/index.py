"""
Vercel serverless function wrapper for FastAPI app.
"""
import sys
import traceback
from pathlib import Path

def log(msg):
    """Log to stderr (captured by Vercel function logs)"""
    print(msg, file=sys.stderr, flush=True)

log("=" * 80)
log("Starting api/index.py - Dependency Check")
log("=" * 80)

# Check Python version
log(f"Python version: {sys.version}")
log(f"Python executable: {sys.executable}")

# Check if key dependencies are installed
dependencies_to_check = [
    ("requests", "requests"),
    ("openai", "openai"),
    ("faiss", "faiss-cpu"),
    ("numpy", "numpy"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("mangum", "mangum"),
]

log("\nChecking dependencies...")
missing_deps = []
installed_deps = []

for module_name, package_name in dependencies_to_check:
    try:
        __import__(module_name)
        version = getattr(__import__(module_name), "__version__", "unknown")
        log(f"✓ {package_name} ({module_name}) - version: {version}")
        installed_deps.append(package_name)
    except ImportError as e:
        log(f"✗ {package_name} ({module_name}) - MISSING: {e}")
        missing_deps.append(package_name)
    except Exception as e:
        log(f"⚠ {package_name} ({module_name}) - ERROR: {e}")
        missing_deps.append(package_name)

log("\n" + "=" * 80)
if missing_deps:
    log(f"ERROR: Missing {len(missing_deps)} dependencies: {', '.join(missing_deps)}")
    log("=" * 80)
    sys.exit(1)
else:
    log(f"SUCCESS: All {len(installed_deps)} dependencies installed")
    log("=" * 80)

# Check Python path
log(f"\nPython path: {sys.path}")

# Add parent directory to path to import app
backend_dir = Path(__file__).parent.parent
log(f"Backend directory: {backend_dir}")
log(f"Backend directory exists: {backend_dir.exists()}")

if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
    log(f"Added {backend_dir} to Python path")

# Check if scripts directory exists
scripts_dir = backend_dir / "scripts"
log(f"Scripts directory: {scripts_dir}")
log(f"Scripts directory exists: {scripts_dir.exists()}")

# Try importing app
log("\n" + "=" * 80)
log("Importing app...")
log("=" * 80)

try:
    from app import app
    log("✓ Successfully imported app")
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
