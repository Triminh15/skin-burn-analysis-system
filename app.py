"""
BurnScan Clinical Analysis System - Main Server Entrypoint
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from frontend.app import app

if __name__ == "__main__":
    import uvicorn
    print("[INFO] Starting BurnScan Clinical System on http://127.0.0.1:8000 ...")
    uvicorn.run("frontend.app:app", host="127.0.0.1", port=8000, reload=True)

