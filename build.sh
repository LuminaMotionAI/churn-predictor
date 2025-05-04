#!/usr/bin/env bash
# exit on error
set -o errexit

# Install dependencies
pip install -r requirements.txt

# Create necessary directories if they don't exist
mkdir -p models
mkdir -p data/external
mkdir -p app/static/images

# Make the script executable
chmod +x build.sh 