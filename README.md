# 넷플릭스 이탈률 예측 시스템 (Netflix Churn Prediction System)

이 프로젝트는 머신러닝과 딥러닝을 활용하여 넷플릭스 사용자의 이탈 가능성을 예측하고, 이탈 방지를 위한 개인화된 추천 시스템을 구현합니다.

## 프로젝트 개요

스트리밍 서비스의 지속적인 성장과 경쟁 심화로 인해 고객 유지(Customer Retention)는 넷플릭스와 같은 OTT 플랫폼에서 매우 중요한 과제가 되었습니다. 고객 이탈(Churn)은 구독 서비스 취소를 의미하며, 새로운 고객을 유치하는 비용이 기존 고객을 유지하는 비용보다 훨씬 높기 때문에 이탈률을 낮추는 것이 비즈니스에 중요합니다.

이 프로젝트는 사용자 행동 데이터를 분석하여 이탈 가능성이 높은 고객을 식별하고, 개인화된 콘텐츠 추천과 타겟 마케팅 전략을 통해 이탈을 방지하는 시스템을 구현합니다.

## 주요 기능

1. **사용자 이탈 예측 모델**
   - 머신러닝 분류 모델을 활용한 이탈 위험 사용자 식별
   - 딥러닝 기반 고급 이탈 예측 모델링

2. **이탈 요인 분석**
   - 특성 중요도 분석을 통한 이탈 주요 요인 식별
   - 사용자 세그먼트별 이탈 패턴 분석

3. **개인화된 콘텐츠 추천 시스템**
   - 협업 필터링과 콘텐츠 기반 필터링을 결합한 하이브리드 추천
   - 실시간 사용자 행동 분석 기반 동적 추천

4. **이탈 방지 전략 시뮬레이션**
   - 다양한 이탈 방지 전략의 효과 시뮬레이션
   - 비용-효과 분석을 통한 최적의 전략 도출

5. **대시보드 및 모니터링**
   - 이탈 위험 실시간 모니터링 대시보드
   - 추천 효과성 측정 및 시각화

## 프로젝트 구조

```
churn-predictor/
├── data/                      # 데이터 폴더
│   ├── raw/                   # 원본 데이터
│   ├── processed/             # 전처리된 데이터
│   └── external/              # 외부 데이터소스
├── notebooks/                 # Jupyter 노트북
│   ├── exploratory/           # 탐색적 데이터 분석
│   ├── modeling/              # 모델링 노트북
│   └── evaluation/            # 모델 평가 및 검증
├── src/                       # 소스 코드
│   ├── data/                  # 데이터 처리 관련 코드
│   ├── features/              # 특성 엔지니어링 코드
│   ├── models/                # 머신러닝 및 딥러닝 모델
│   ├── visualization/         # 시각화 코드
│   └── utils/                 # 유틸리티 함수
├── models/                    # 저장된 모델 파일
├── reports/                   # 결과 리포트 및 시각화
├── app/                       # 웹 애플리케이션 (Flask)
│   ├── static/                # CSS, JavaScript 등
│   ├── templates/             # HTML 템플릿
│   └── app.py                 # 메인 애플리케이션 파일
├── tests/                     # 테스트 코드
├── requirements.txt           # 필요한 패키지 목록
├── setup.py                   # 설치 스크립트
└── README.md                  # 프로젝트 설명
```

## 설치 및 실행 방법

### 요구사항
- Python 3.8 이상
- pip (Python 패키지 매니저)

### 설치
```bash
# 저장소 클론
git clone https://github.com/yourusername/churn-predictor.git
cd churn-predictor

# 가상 환경 생성 및 활성화 (선택 사항)
python -m venv venv
source venv/bin/activate  # Linux, macOS
venv\Scripts\activate     # Windows

# 필요한 패키지 설치
pip install -r requirements.txt
```

### 데이터 준비
```bash
# 데이터 다운로드 및 전처리 스크립트 실행
python src/data/make_dataset.py
```

### 모델 학습
```bash
# 모델 학습 실행
python src/models/train_model.py
```

### 웹 애플리케이션 실행
```bash
# Flask 앱 실행
python app/app.py
```
이후 웹 브라우저에서 `http://localhost:5000`으로 접속하여 대시보드를 확인할 수 있습니다.

## 기술 스택

- **언어**: Python
- **머신러닝**: Scikit-learn, XGBoost, LightGBM
- **딥러닝**: TensorFlow, Keras
- **데이터 처리**: Pandas, NumPy
- **시각화**: Matplotlib, Seaborn, Plotly
- **웹 애플리케이션**: Flask
- **모델 설명**: SHAP, Lime

## 성능 및 벤치마크

- 이탈 예측 정확도: 85% 이상 (목표)
- 추천 시스템 정확도: 80% 이상 (목표)
- 웹 애플리케이션 응답 시간: 1초 이내

## 기여 방법

1. 이 저장소를 Fork하세요
2. 새로운 Branch를 생성하세요 (`git checkout -b feature/amazing-feature`)
3. 변경사항을 Commit하세요 (`git commit -m 'Add some amazing feature'`)
4. Branch에 Push하세요 (`git push origin feature/amazing-feature`)
5. Pull Request를 생성하세요

## 라이센스

MIT 라이센스에 따라 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 연락처

프로젝트 관리자 - [이메일 주소](mailto:your.email@example.com)

프로젝트 링크: [https://github.com/yourusername/churn-predictor](https://github.com/yourusername/churn-predictor) 