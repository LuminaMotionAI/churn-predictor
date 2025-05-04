#!/usr/bin/env bash
# exit on error
set -o errexit

# Run the Flask app directly
cd app && python app.py 