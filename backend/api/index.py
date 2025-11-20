"""
Vercel serverless function wrapper for FastAPI app.
"""
from mangum import Mangum
import sys
from pathlib import Path

# Add parent directory to path to import app
backend_dir = Path(__file__).parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app import app

# Create Mangum handler for Vercel
handler = Mangum(app)
