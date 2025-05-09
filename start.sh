#!/bin/bash
# exit on error
set -o errexit

echo "Starting churn-predictor app..."
echo "App directory content:"
ls -la

# WSGI 모듈과 application 변수 사용
gunicorn --bind 0.0.0.0:$PORT wsgi:application 