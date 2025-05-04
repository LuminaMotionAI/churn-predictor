#!/usr/bin/env bash
# exit on error
set -o errexit

echo "Starting Flask application..."

# Set environment variables
export PYTHONUNBUFFERED=true
export FLASK_ENV=production

# Print current directory and files
echo "Current directory: $(pwd)"
ls -la

# Change to app directory and run Flask app
cd app
echo "App directory content:"
ls -la
python app.py 