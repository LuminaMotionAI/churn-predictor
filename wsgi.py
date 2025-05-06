"""
Render.com deployment entry point
"""
import sys
import os

# Add app directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))

# Import app from app.py
from app import app as application

# This allows Render to find the application
app = application

if __name__ == "__main__":
    # Get PORT from environment variable or use 8080 as default
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=False) 