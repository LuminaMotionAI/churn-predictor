"""
Render.com deployment entry point
"""
import sys
import os

# 프로젝트 경로 추가
project_home = os.path.expanduser('~/churn-predictor')
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# 앱 초기화
from app.app import app

# Render.com에서는 PORT 환경변수를 제공합니다
port = int(os.environ.get("PORT", 8080))

# WSGI 애플리케이션
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port) 