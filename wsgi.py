"""
WSGI 진입점 - Render.com 배포용
"""
import sys
import os

# 앱 초기화
from app.app import app

# 표준 WSGI application 변수 설정 (gunicorn이 이 이름을 찾음)
application = app

# 개발 환경을 위한 테스트 실행 코드
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port) 