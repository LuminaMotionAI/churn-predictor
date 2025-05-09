#!/bin/bash
# exit on error
set -o errexit

echo "Starting churn-predictor app..."
echo "App directory content:"
ls -la

# Render.com에서 제공하는 PORT 환경변수 사용
gunicorn --bind 0.0.0.0:$PORT app.app:app 