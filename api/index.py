import os
import sys

# Ensure project root and backend are on sys.path for Vercel Serverless environment
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.app import app
