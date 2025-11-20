"""
Vercel serverless function wrapper for FastAPI app.
"""

import sys
from pathlib import Path

# Add backend directory to path
current_file = Path(__file__).resolve()
backend_dir = current_file.parent.parent.resolve()
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Import the app
from app import app
