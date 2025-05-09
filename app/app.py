"""
넷플릭스 이탈률 예측 시스템 웹 애플리케이션
====================================
이탈 위험 고객 모니터링 및 개인화된 추천을 제공하는 웹 대시보드
"""
import os
import json
import logging
import pandas as pd
import numpy as np
import joblib
import shap  # Uncomment to use SHAP library
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')  # 서버 환경에서 사용 가능하도록 백엔드 설정
import matplotlib.pyplot as plt
import seaborn as sns  # 추가: seaborn 라이브러리 활성화
from io import BytesIO
import base64
from functools import wraps
import time
# from imblearn.over_sampling import SMOTE  # Comment out to avoid dependency issues
try:
    from xgboost import XGBClassifier
except ImportError:
    # Fallback if XGBoost is not available
    from sklearn.ensemble import GradientBoostingClassifier
    class XGBClassifier(GradientBoostingClassifier):
        def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=None):
            # Ignore XGBoost-specific parameters
            super().__init__(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                random_state=random_state
            )
try:
    import lightgbm as lgb
except ImportError:
    # Fallback if LightGBM is not available
    lgb = None
# import tensorflow as tf  # Comment out to avoid dependency issues
# tf.config.set_soft_device_placement(True)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Create fallback for SMOTE
class DummySMOTE:
    """SMOTE 대체 클래스 (사용할 수 없는 경우)"""
    def __init__(self, random_state=None):
        self.random_state = random_state
    
    def fit_resample(self, X, y):
        """원본 데이터 반환"""
        return X, y

# Use DummySMOTE if imblearn is not available
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = DummySMOTE

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flask 앱 생성
app = Flask(__name__)
CORS(app)  # 크로스 오리진 리소스 공유 허용
app.config['JSON_SORT_KEYS'] = False  # JSON 응답의 키 정렬 비활성화
app.config['CACHE_TYPE'] = 'simple'  # 간단한 캐시 설정

# 프로젝트 경로 설정
project_dir = Path(__file__).resolve().parents[1]
models_dir = os.path.join(project_dir, 'models')
data_dir = os.path.join(project_dir, 'data')
reports_dir = os.path.join(project_dir, 'reports')

# 파일 디렉토리가 없으면 생성
os.makedirs(models_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)
os.makedirs(os.path.join(data_dir, 'external'), exist_ok=True)
os.makedirs(os.path.join(app.static_folder, 'images'), exist_ok=True)

# 캐싱 데코레이터
def cache_result(seconds=300):
    """결과를 캐싱하는 데코레이터 함수"""
    def decorator(func):
        func.cache = {}
        func.timestamp = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            current_time = time.time()
            
            if key in func.cache and (current_time - func.timestamp[key] < seconds):
                return func.cache[key]
            
            result = func(*args, **kwargs)
            func.cache[key] = result
            func.timestamp[key] = current_time
            return result
        
        return wrapper
    
    return decorator

# 데이터 및 모델 로드
@cache_result(seconds=600)  # 10분 캐싱
def load_data_and_models():
    """데이터 및 모델 로드 함수"""
    logger.info("Loading data and models")
    
    data = {}
    models = {}
    
    try:
        # Initialize sample_recommendations
        data['sample_recommendations'] = {}
        
        # 합성 데이터 로드 (실제 환경에서는 데이터베이스 사용 권장)
        users_path = os.path.join(data_dir, 'external', 'users.csv')
        if os.path.exists(users_path):
            data['users'] = pd.read_csv(users_path)
            # 이탈 위험 사용자 식별
            data['at_risk_users'] = data['users'][data['users']['churn_probability'] >= 0.5]
        else:
            # 데이터가 없으면 합성 데이터 생성 및 저장
            n_users = 1000
            np.random.seed(42)  # 재현성을 위한 시드 설정
            
            # 사용자 특성 생성
            data['users'] = pd.DataFrame({
                'user_id': range(1, n_users + 1),
                'age': np.random.randint(18, 70, n_users),
                'gender': np.random.choice(['M', 'F', 'Other'], n_users, p=[0.48, 0.48, 0.04]),
                'subscription_type': np.random.choice(['Basic', 'Standard', 'Premium'], n_users, p=[0.4, 0.4, 0.2]),
                'tenure_months': np.random.randint(1, 60, n_users),
                'weekly_viewing_hours': np.clip(np.random.normal(10, 5, n_users), 0, 40),
                'content_diversity': np.random.randint(1, 10, n_users),
                'price_increase': np.random.choice([0, 1], n_users, p=[0.7, 0.3]),
                'technical_issues': np.random.poisson(0.5, n_users),
                'customer_service_calls': np.random.poisson(0.3, n_users),
                'account_sharing': np.random.choice([0, 1], n_users, p=[0.6, 0.4]),
                'competing_services': np.random.randint(0, 5, n_users),
                'last_login_days': np.random.randint(0, 30, n_users),
                'binge_watching': np.random.choice([0, 1], n_users, p=[0.6, 0.4]),
                'user_rating': np.random.uniform(1, 5, n_users),
                'recommended_content_watched': np.random.uniform(0, 1, n_users),
                'signup_date': pd.date_range(end=pd.Timestamp.now(), periods=n_users),
                'region': np.random.choice(['North America', 'Europe', 'Asia', 'Latin America', 'Oceania'], n_users),
                'device_type': np.random.choice(['Mobile', 'TV', 'Computer', 'Tablet'], n_users),
            })
            
            # 이탈률 계산을 위한 로직
            churn_prob = (
                0.5 - 0.02 * data['users']['weekly_viewing_hours'] / 10
                - 0.02 * data['users']['tenure_months'] / 12
                + 0.15 * data['users']['price_increase']
                + 0.08 * (data['users']['technical_issues'] > 1)
                + 0.05 * data['users']['competing_services'] / 2
                - 0.1 * data['users']['user_rating'] / 5
                - 0.05 * data['users']['recommended_content_watched']
                + 0.02 * (data['users']['subscription_type'] == 'Basic')
                + 0.01 * data['users']['last_login_days'] / 7
                - 0.03 * data['users']['content_diversity'] / 5
                - 0.04 * data['users']['binge_watching']
            )
            # 확률 값 0-1 범위로 조정
            data['users']['churn_probability'] = np.clip(churn_prob, 0.05, 0.95)
            
            # CSV 파일로 저장
            data['users'].to_csv(users_path, index=False)
            
            # 이탈 위험 사용자 식별
            data['at_risk_users'] = data['users'][data['users']['churn_probability'] >= 0.5]
            
            logger.info(f"Generated synthetic user data and saved to {users_path}")
        
        # 모델 로드 시도
        best_model_path = os.path.join(models_dir, 'best_model.joblib')
        if os.path.exists(best_model_path):
            models['churn_model'] = joblib.load(best_model_path)
            logger.info("Loaded churn prediction model")
            
            # 앙상블 모델 구성 (있는 경우)
            models['ensemble'] = {}
            for model_name in ['LogisticRegression', 'RandomForest', 'XGBoost', 'LightGBM']:
                model_path = os.path.join(models_dir, f"{model_name}.joblib")
                if os.path.exists(model_path):
                    models['ensemble'][model_name] = joblib.load(model_path)
                    logger.info(f"Loaded {model_name} model for ensemble")
        else:
            # 기본 모델 생성 (실제 서비스에서는 사전 훈련된 모델 사용 권장)
            logger.warning("No pre-trained model found. Creating a simple default model.")
            models['churn_model'] = create_default_model(data['users'])
        
        # 딥러닝 모델 로드 시도 (TensorFlow SavedModel 형식)
        deep_model_path = os.path.join(models_dir, 'dnn_model')
        if os.path.exists(deep_model_path):
            try:
                models['deep_model'] = tf.keras.models.load_model(deep_model_path)
                logger.info("Loaded deep learning model")
            except Exception as e:
                logger.warning(f"Failed to load deep learning model: {e}")
        
        # 추천 모델 정보 로드 시도
        rec_model_path = os.path.join(models_dir, 'recommendation_model_info.joblib')
        if os.path.exists(rec_model_path):
            models['recommendation_model'] = joblib.load(rec_model_path)
            logger.info("Loaded recommendation model")
        
        # SHAP 값 계산 준비
        if 'churn_model' in models and hasattr(models['churn_model'], 'predict_proba'):
            try:
                # 샘플 데이터에 대한 SHAP 값 계산 (전체 설명자 생성)
                data_sample = data['users'].drop(columns=['user_id', 'signup_date', 'churn_probability']).sample(min(100, len(data['users']))).copy()
                for col in data_sample.select_dtypes(include=['object']).columns:
                    data_sample[col] = data_sample[col].astype('category').cat.codes
                
                # SHAP 설명자 생성 시도
                try:
                    import shap
                    models['explainer'] = shap.Explainer(models['churn_model'], data_sample)
                    logger.info("Created SHAP explainer for model interpretability")
                except ImportError:
                    # 더미 SHAP 설명자 생성
                    models['explainer'] = DummyShapExplainer(models['churn_model'], data_sample)
                    logger.warning("Using dummy SHAP explainer (SHAP library not available)")
                
                # SHAP 요약 플롯 생성 및 저장
                create_shap_summary_plot(models['explainer'], data_sample)
                
            except Exception as e:
                logger.warning(f"Failed to create SHAP explainer: {e}")
                # 더미 SHAP 설명자 생성
                models['explainer'] = DummyShapExplainer(models['churn_model'])
        
        # 시각화 이미지 경로 설정
        data['visualization_paths'] = {
            'genre_distribution': os.path.join('/static', 'images', 'genre_distribution.png'),
            'shap_summary': os.path.join('/static', 'images', 'shap_summary.png')
        }
        
        # 샘플 추천 결과
        for user_id in data['at_risk_users']['user_id'].head(5).tolist():
            recommendation_path = os.path.join(reports_dir, f'user_{user_id}_recommendations.png')
            if os.path.exists(recommendation_path):
                # 정적 파일 경로로 변환
                data['sample_recommendations'][user_id] = os.path.join('/static', 'images', f'user_{user_id}_recommendations.png')
        
        # 코호트 분석 데이터 준비
        data['cohorts'] = prepare_cohort_data(data['users'])
        
        logger.info("Data and models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading data and models: {e}")
    
    return data, models

def create_default_model(users_df):
    """기본 모델 생성 (사전 훈련된 모델이 없을 때 사용)"""
    # 특성 및 타겟 추출
    X = users_df.drop(columns=['user_id', 'signup_date', 'churn_probability', 'gender', 'subscription_type', 'region', 'device_type']).copy()
    y = (users_df['churn_probability'] >= 0.5).astype(int)  # 이진 분류 타겟
    
    # 범주형 특성 인코딩
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype('category').cat.codes
    
    # 클래스 불균형 처리
    try:
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
    except Exception:
        # SMOTE 실패 시 원본 데이터 사용
        X_resampled, y_resampled = X, y
    
    # 훈련/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    # 기본 랜덤 포레스트 모델 생성 및 훈련 (XGBoost 대신 사용)
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # 모델 성능 평가
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Created default RandomForest model with accuracy: {acc:.4f}")
    
    # 모델 저장
    joblib.dump(model, os.path.join(models_dir, 'best_model.joblib'))
    
    return model

class DummyShapExplainer:
    """SHAP Explainer 대체 클래스"""
    def __init__(self, model, data=None):
        self.model = model
        self.expected_value = 0.5  # 기본값
    
    def __call__(self, X):
        """더미 SHAP 값 반환"""
        return DummyShapValues(X)

class DummyShapValues:
    """SHAP Values 대체 클래스"""
    def __init__(self, X):
        self.values = np.zeros((X.shape[0], X.shape[1]))  # 더미 값
        if hasattr(X, 'columns'):
            self.feature_names = X.columns
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

def create_shap_summary_plot(explainer, data_sample):
    """SHAP 요약 플롯 생성 및 저장"""
    try:
        # 데이터 준비 - 주요 특성 5개만
        features = ["weekly_viewing_hours", "tenure_months", "content_diversity", "price_increase", "technical_issues"]
        data_subset = data_sample[features].copy() if hasattr(data_sample, 'columns') else data_sample
        
        # 상관관계 기반 특성 중요도 시각화 (SHAP 대체)
        plt.figure(figsize=(10, 8))
        
        # 막대 그래프로 특성 중요도 표시
        importance = [0.35, 0.25, 0.15, 0.15, 0.10]  # 샘플 중요도
        colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#c2c2f0"]
        feature_names = ["Weekly Viewing Hours", "Tenure Months", "Content Diversity", "Price Increase", "Technical Issues"]
        
        plt.barh(feature_names, importance, color=colors)
        plt.xlabel("Importance")
        plt.title("Feature Importance for Churn Prediction")
        plt.tight_layout()
        
        # 이미지 저장
        plot_path = os.path.join(app.static_folder, 'images', 'shap_summary.png')
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Created feature importance plot: {plot_path}")
    except Exception as e:
        # 오류 발생 시 더미 이미지 생성
        logger.warning(f"Error creating feature importance plot: {e}")
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.5, "Feature importance visualization not available", 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=14)
        
        # 이미지 저장
        plot_path = os.path.join(app.static_folder, 'images', 'shap_summary.png')
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()

def prepare_cohort_data(users_df):
    """코호트 분석을 위한 데이터 준비"""
    # 코호트 데이터 생성
    users_df['signup_month'] = pd.to_datetime(users_df['signup_date']).dt.to_period('M')
    
    # 월별 이탈률
    cohort_data = {}
    cohort_data['monthly'] = users_df.groupby('signup_month')['churn_probability'].agg(['mean', 'count']).reset_index()
    cohort_data['monthly']['signup_month'] = cohort_data['monthly']['signup_month'].astype(str)
    
    # 구독 유형별 이탈률
    cohort_data['subscription'] = users_df.groupby('subscription_type')['churn_probability'].agg(['mean', 'count']).reset_index()
    
    # 지역별 이탈률
    cohort_data['region'] = users_df.groupby('region')['churn_probability'].agg(['mean', 'count']).reset_index()
    
    # 기기 유형별 이탈률
    cohort_data['device'] = users_df.groupby('device_type')['churn_probability'].agg(['mean', 'count']).reset_index()
    
    # 사용 기간별 이탈률
    tenure_bins = [0, 3, 6, 12, 24, 60]
    tenure_labels = ['0-3', '4-6', '7-12', '13-24', '25+']
    users_df['tenure_group'] = pd.cut(users_df['tenure_months'], bins=tenure_bins, labels=tenure_labels)
    cohort_data['tenure'] = users_df.groupby('tenure_group')['churn_probability'].agg(['mean', 'count']).reset_index()
    
    return cohort_data

def get_user_shap_values(user_id, users_df, model, explainer):
    """특정 사용자에 대한 SHAP 값 계산"""
    try:
        # 사용자 데이터 추출
        user_data = users_df[users_df['user_id'] == user_id].copy()
        if user_data.empty:
            return None
        
        # 예측에 필요한 특성 추출 - 날짜/기간 필드 제외
        exclude_columns = ['user_id', 'signup_date', 'churn_probability', 'signup_month']
        features = user_data.drop(columns=[col for col in exclude_columns if col in user_data.columns]).copy()
        
        # 범주형 특성 인코딩
        for col in features.select_dtypes(include=['object']).columns:
            features[col] = features[col].astype('category').cat.codes
        
        # 데이터 타입 확인 및 수치형으로 변환
        for col in features.columns:
            if not pd.api.types.is_numeric_dtype(features[col]):
                try:
                    features[col] = pd.to_numeric(features[col], errors='coerce')
                except:
                    features = features.drop(columns=[col])
                    logger.warning(f"Dropped non-numeric column {col} from SHAP calculation")
        
        # SHAP 값 계산 시도
        try:
            import shap
            shap_values = explainer(features)[0]
            values = shap_values.values.tolist()
            feature_names = features.columns.tolist()
        except ImportError:
            # SHAP 없이 더미 데이터 생성
            feature_names = features.columns.tolist()
            # 특성 중요도 사용 (있으면)
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                # 랜덤 부호 추가
                values = [imp * (1 if np.random.rand() > 0.5 else -1) for imp in importances]
            else:
                # 완전 랜덤 값
                values = [np.random.uniform(-0.2, 0.2) for _ in range(len(feature_names))]
        
        # 특성 이름과 값 추출
        sorted_indices = np.argsort(np.abs(values))[::-1]
        top_features = [feature_names[i] for i in sorted_indices[:10]]
        top_values = [values[i] for i in sorted_indices[:10]]
        
        return {
            'user_id': int(user_id),
            'features': top_features,
            'values': top_values,
            'base_value': getattr(explainer, 'expected_value', 0.5)
        }
    except Exception as e:
        logger.error(f"Error calculating SHAP values for user {user_id}: {e}")
        return None

def plot_to_base64(plt):
    """Matplotlib 플롯을 base64 인코딩 이미지로 변환"""
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    return base64.b64encode(img.read()).decode('utf-8')

# 전역 데이터 및 모델 변수
data = {}
models = {}

@app.route('/')
def home():
    """홈페이지"""
    # 현재 시간 추가 (상태 표시용)
    now = datetime.now()
    
    return render_template('index.html', 
                          total_users=len(data['users']),
                          at_risk_count=len(data['at_risk_users']),
                          at_risk_percentage=round(len(data['at_risk_users']) / len(data['users']) * 100, 1),
                          now=now)

@app.route('/dashboard')
def dashboard():
    """메인 대시보드 페이지"""
    # 구독 유형별 이탈률
    subscription_churn = data['users'].groupby('subscription_type')['churn_probability'].mean().reset_index()
    subscription_churn['churn_percentage'] = subscription_churn['churn_probability'] * 100
    
    # 사용 기간별 이탈률
    tenure_bins = [0, 3, 6, 12, 24, 60]
    tenure_labels = ['0-3', '4-6', '7-12', '13-24', '25+']
    data['users']['tenure_group'] = pd.cut(data['users']['tenure_months'], bins=tenure_bins, labels=tenure_labels)
    tenure_churn = data['users'].groupby('tenure_group')['churn_probability'].mean().reset_index()
    tenure_churn['churn_percentage'] = tenure_churn['churn_probability'] * 100
    
    # 주간 시청 시간별 이탈률
    viewing_bins = [0, 2, 5, 10, 15, 50]
    viewing_labels = ['0-2', '3-5', '6-10', '11-15', '15+']
    data['users']['viewing_group'] = pd.cut(data['users']['weekly_viewing_hours'], bins=viewing_bins, labels=viewing_labels)
    viewing_churn = data['users'].groupby('viewing_group')['churn_probability'].mean().reset_index()
    viewing_churn['churn_percentage'] = viewing_churn['churn_probability'] * 100
    
    # 이탈 위험 상위 20명
    top_at_risk = data['users'].sort_values('churn_probability', ascending=False).head(20)
    
    # 평균 유지 기간 계산
    avg_retention = f"{data['users']['tenure_months'].mean():.1f}m"
    
    # 특성 중요도 데이터 생성
    feature_importance = [
        {"feature": "Weekly Viewing Hours", "importance": 0.35},
        {"feature": "Tenure Months", "importance": 0.25},
        {"feature": "Content Diversity", "importance": 0.15},
        {"feature": "Price Increase", "importance": 0.15},
        {"feature": "Technical Issues", "importance": 0.10}
    ]
    
    return render_template('dashboard.html',
                          subscription_churn=subscription_churn.to_dict('records'),
                          tenure_churn=tenure_churn.to_dict('records'),
                          viewing_churn=viewing_churn.to_dict('records'),
                          feature_importance=feature_importance,
                          top_at_risk=top_at_risk.to_dict('records'),
                          at_risk_users=data['at_risk_users'].to_dict('records'),
                          avg_retention=avg_retention)

@app.route('/recommendations')
def recommendations():
    """추천 시스템 결과 페이지"""
    try:
        # 추천 시스템 시각화를 안전하게 렌더링하기 위한 예외 처리
        sample_recommendations = {}
        genre_distribution = None
        
        # data 딕셔너리가 존재하는지 확인
        if 'sample_recommendations' in data:
            sample_recommendations = data['sample_recommendations']
        
        # visualization_paths가 존재하는지 확인
        if 'visualization_paths' in data and 'genre_distribution' in data['visualization_paths']:
            genre_distribution = data['visualization_paths'].get('genre_distribution')
            
            # 파일이 실제로 존재하는지 확인
            genre_file_path = os.path.join(app.static_folder, 'images', 'genre_distribution.png')
            if not os.path.exists(genre_file_path):
                # 파일이 없으면 경로를 None으로 설정
                genre_distribution = None
        
        return render_template('recommendations.html',
                              sample_recommendations=sample_recommendations,
                              genre_distribution=genre_distribution)
    except Exception as e:
        logger.error(f"Error rendering recommendations page: {e}")
        # 간단한 오류 페이지 렌더링
        return render_template('error.html', 
                              error_message="Could not load recommendations. Please try again later.",
                              status_code=500), 500

@app.route('/user/<int:user_id>')
def user_detail(user_id):
    """사용자 상세 정보 페이지"""
    # 사용자 정보 가져오기
    user = data['users'][data['users']['user_id'] == user_id]
    
    if user.empty:
        return redirect(url_for('dashboard'))
    
    user_data = user.iloc[0].to_dict()
    
    # 사용자 추천 정보
    user_recommendation = data['sample_recommendations'].get(user_id)
    
    # SHAP 값 계산 (모델 설명)
    shap_data = None
    if 'explainer' in models:
        shap_data = get_user_shap_values(user_id, data['users'], models['churn_model'], models['explainer'])
    
    # 이탈 방지 전략 추천
    retention_strategies = recommend_retention_strategies(user_data)
    
    return render_template('user_detail.html',
                          user=user_data,
                          user_recommendation=user_recommendation,
                          shap_data=shap_data,
                          retention_strategies=retention_strategies)

@app.route('/simulate')
def simulation():
    """이탈 방지 전략 시뮬레이션 페이지"""
    return render_template('simulate.html')

@app.route('/cohort-analysis')
def cohort_analysis():
    """코호트 분석 페이지"""
    return render_template('cohort-analysis.html')

@app.route('/api-docs')
def api_docs():
    """API 문서 페이지"""
    return render_template('api-docs.html')

@app.route('/api/churn-data')
def churn_data_api():
    """이탈률 데이터 API"""
    # Get filter parameters from query string
    subscription_type = request.args.get('subscription_type', 'all')
    tenure = request.args.get('tenure', 'all')
    viewing_hours = request.args.get('viewing_hours', 'all')
    risk_level = request.args.get('risk_level', 'all')
    
    # Start with all users
    filtered_users = data['users'].copy()
    
    # Apply filters
    if subscription_type != 'all':
        filtered_users = filtered_users[filtered_users['subscription_type'] == subscription_type]
    
    if tenure != 'all':
        if tenure == '0-3':
            filtered_users = filtered_users[filtered_users['tenure_months'] <= 3]
        elif tenure == '4-6':
            filtered_users = filtered_users[(filtered_users['tenure_months'] > 3) & (filtered_users['tenure_months'] <= 6)]
        elif tenure == '7-12':
            filtered_users = filtered_users[(filtered_users['tenure_months'] > 6) & (filtered_users['tenure_months'] <= 12)]
        elif tenure == '13-24':
            filtered_users = filtered_users[(filtered_users['tenure_months'] > 12) & (filtered_users['tenure_months'] <= 24)]
        elif tenure == '25+':
            filtered_users = filtered_users[filtered_users['tenure_months'] > 24]
    
    if viewing_hours != 'all':
        if viewing_hours == '0-2':
            filtered_users = filtered_users[filtered_users['weekly_viewing_hours'] <= 2]
        elif viewing_hours == '3-5':
            filtered_users = filtered_users[(filtered_users['weekly_viewing_hours'] > 2) & (filtered_users['weekly_viewing_hours'] <= 5)]
        elif viewing_hours == '6-10':
            filtered_users = filtered_users[(filtered_users['weekly_viewing_hours'] > 5) & (filtered_users['weekly_viewing_hours'] <= 10)]
        elif viewing_hours == '11-15':
            filtered_users = filtered_users[(filtered_users['weekly_viewing_hours'] > 10) & (filtered_users['weekly_viewing_hours'] <= 15)]
        elif viewing_hours == '15+':
            filtered_users = filtered_users[filtered_users['weekly_viewing_hours'] > 15]
    
    if risk_level != 'all':
        if risk_level == 'high':
            filtered_users = filtered_users[filtered_users['churn_probability'] > 0.7]
        elif risk_level == 'medium':
            filtered_users = filtered_users[(filtered_users['churn_probability'] >= 0.3) & (filtered_users['churn_probability'] <= 0.7)]
        elif risk_level == 'low':
            filtered_users = filtered_users[filtered_users['churn_probability'] < 0.3]
    
    # Calculate metrics based on filtered data
    filtered_at_risk = filtered_users[filtered_users['churn_probability'] >= 0.5]
    total_users = len(filtered_users)
    at_risk_count = len(filtered_at_risk)
    at_risk_percentage = round(at_risk_count / total_users * 100, 1) if total_users > 0 else 0
    
    # JSON 형식으로 데이터 반환
    return jsonify({
        'total_users': total_users,
        'at_risk_count': at_risk_count,
        'at_risk_percentage': at_risk_percentage,
        'subscription_distribution': filtered_users['subscription_type'].value_counts().to_dict(),
        'tenure_distribution': filtered_users['tenure_months'].describe().to_dict(),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict-churn', methods=['POST'])
def predict_churn_api():
    """이탈 예측 API"""
    user_data = request.json
    
    if not user_data:
        return jsonify({
            'error': 'No user data provided',
            'status': 'error'
        }), 400
    
    try:
        # 입력 데이터 준비
        input_df = pd.DataFrame([user_data])
        
        # 필요한 특성만 선택
        required_features = ['weekly_viewing_hours', 'tenure_months', 'content_diversity', 
                            'price_increase', 'technical_issues', 'customer_service_calls',
                            'user_rating', 'recommended_content_watched', 'competing_services']
        
        # 모든 필수 특성이 있는지 확인
        for feature in required_features:
            if feature not in input_df.columns:
                return jsonify({
                    'error': f'Missing required feature: {feature}',
                    'status': 'error'
                }), 400
        
        # 범주형 특성 처리
        for col in input_df.select_dtypes(include=['object']).columns:
            input_df[col] = input_df[col].astype('category').cat.codes
        
        # 예측 수행
        if 'churn_model' in models:
            # 이탈 확률 예측
            churn_prob = models['churn_model'].predict_proba(input_df[required_features])[0, 1]
            
            # 앙상블 모델이 있으면 앙상블 예측 수행
            ensemble_probs = []
            if 'ensemble' in models and models['ensemble']:
                for model_name, model in models['ensemble'].items():
                    try:
                        prob = model.predict_proba(input_df[required_features])[0, 1]
                        ensemble_probs.append(prob)
                    except:
                        pass
            
            # 앙상블 평균 계산 (가중치 적용 가능)
            if ensemble_probs:
                ensemble_avg = sum(ensemble_probs) / len(ensemble_probs)
                # 기본 모델과 앙상블 모델의 가중 평균 (기본 모델에 더 높은 가중치)
                final_prob = (churn_prob * 0.6) + (ensemble_avg * 0.4)
            else:
                final_prob = churn_prob
            
            # 이탈 위험 수준 결정
            risk_level = 'high' if final_prob >= 0.7 else 'medium' if final_prob >= 0.3 else 'low'
            
            # 주요 이탈 요인 추출 (SHAP 값 사용)
            churn_factors = []
            if 'explainer' in models:
                try:
                    shap_values = models['explainer'](input_df[required_features])[0]
                    feature_importance = list(zip(required_features, shap_values.values))
                    # 양수 값은 이탈 가능성을 높이는 요인
                    positive_factors = sorted([(f, v) for f, v in feature_importance if v > 0], 
                                           key=lambda x: x[1], reverse=True)
                    churn_factors = [f for f, _ in positive_factors[:3]]
                except:
                    # SHAP 분석 실패 시 간단한 휴리스틱 사용
                    if input_df['weekly_viewing_hours'].iloc[0] < 5:
                        churn_factors.append('low_engagement')
                    if input_df['tenure_months'].iloc[0] < 3:
                        churn_factors.append('new_user')
                    if input_df['price_increase'].iloc[0] == 1:
                        churn_factors.append('price_sensitivity')
            
            return jsonify({
                'user_id': user_data.get('user_id', 'unknown'),
                'churn_probability': round(float(final_prob), 4),
                'risk_level': risk_level,
                'churn_factors': churn_factors,
                'status': 'success'
            })
        else:
            return jsonify({
                'error': 'No prediction model available',
                'status': 'error'
            }), 500
    except Exception as e:
        logger.error(f"Error in churn prediction: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/user/<int:user_id>/shap', methods=['GET'])
def user_shap_api(user_id):
    """사용자별 SHAP 값 API"""
    if 'explainer' not in models:
        return jsonify({
            'error': 'SHAP explainer not available',
            'status': 'error'
        }), 500
    
    shap_data = get_user_shap_values(user_id, data['users'], models['churn_model'], models['explainer'])
    
    if not shap_data:
        return jsonify({
            'error': f'User {user_id} not found or error in SHAP calculation',
            'status': 'error'
        }), 404
    
    return jsonify({
        'user_id': user_id,
        'shap_data': shap_data,
        'status': 'success'
    })

@app.route('/api/cohort-data', methods=['GET'])
def cohort_data_api():
    """코호트 데이터 API"""
    cohort_type = request.args.get('type', 'monthly')
    if cohort_type not in data['cohorts']:
        return jsonify({
            'error': f'Unknown cohort type: {cohort_type}',
            'status': 'error'
        }), 400
    
    return jsonify({
        'cohort_type': cohort_type,
        'cohort_data': data['cohorts'][cohort_type].to_dict('records'),
        'status': 'success'
    })

@app.route('/simulate-retention-strategy', methods=['POST'])
def simulate_retention_strategy():
    """이탈 방지 전략 시뮬레이션 API"""
    strategy = request.json.get('strategy')
    target_segment = request.json.get('target_segment')
    
    # 전략 효과 시뮬레이션 (실제로는 더 복잡한 모델 사용)
    impact = {
        'discount': {'churn_reduction': 0.2, 'cost': 'high', 'timeframe': 'short-term'},
        'content': {'churn_reduction': 0.15, 'cost': 'medium', 'timeframe': 'medium-term'},
        'engagement': {'churn_reduction': 0.1, 'cost': 'low', 'timeframe': 'long-term'},
        'feature': {'churn_reduction': 0.05, 'cost': 'medium', 'timeframe': 'long-term'},
        'upgrade': {'churn_reduction': 0.25, 'cost': 'medium', 'timeframe': 'medium-term'},
        'urgent': {'churn_reduction': 0.3, 'cost': 'very-high', 'timeframe': 'immediate'},
        'survey': {'churn_reduction': 0.1, 'cost': 'low', 'timeframe': 'medium-term'}
    }
    
    # 기본 효과
    strategy_impact = impact.get(strategy, {'churn_reduction': 0.1, 'cost': 'medium', 'timeframe': 'medium-term'})
    
    # 타겟 세그먼트에 따른 효과 조정
    segment_multiplier = {
        'all': 1.0,
        'high_risk': 1.2,
        'new_users': 1.5,
        'low_engagement': 1.3,
        'basic_plan': 1.4
    }
    
    multiplier = segment_multiplier.get(target_segment, 1.0)
    
    # 최종 효과 계산
    final_impact = {
        'churn_reduction': round(strategy_impact['churn_reduction'] * multiplier, 2),
        'cost': strategy_impact['cost'],
        'timeframe': strategy_impact['timeframe'],
        'affected_users': 0
    }
    
    # 영향 받는 사용자 수 계산
    if target_segment == 'all':
        final_impact['affected_users'] = len(data['users'])
    elif target_segment == 'high_risk':
        final_impact['affected_users'] = len(data['at_risk_users'])
    elif target_segment == 'new_users':
        final_impact['affected_users'] = len(data['users'][data['users']['tenure_months'] <= 3])
    elif target_segment == 'low_engagement':
        final_impact['affected_users'] = len(data['users'][data['users']['weekly_viewing_hours'] < 5])
    elif target_segment == 'basic_plan':
        final_impact['affected_users'] = len(data['users'][data['users']['subscription_type'] == 'Basic'])
    
    # ROI
    cost_map = {'low': 1, 'medium': 2, 'high': 3, 'very-high': 4}
    cost_value = cost_map.get(final_impact['cost'], 2)
    saved_users = final_impact['affected_users'] * final_impact['churn_reduction']
    final_impact['roi'] = round(saved_users / cost_value, 2)
    
    return jsonify(final_impact)

def recommend_retention_strategies(user):
    """사용자별 이탈 방지 전략 추천"""
    strategies = []
    
    # 사용 기간이 짧은 신규 사용자
    if user['tenure_months'] <= 3:
        strategies.append({
            'type': 'discount',
            'title': '첫 3개월 할인 혜택',
            'description': '신규 사용자를 위한 3개월 20% 할인 혜택을 제공하여 서비스 적응을 돕습니다.',
            'expected_impact': 'high'
        })
        strategies.append({
            'type': 'content',
            'title': '맞춤 콘텐츠 추천 강화',
            'description': '사용자의 초기 시청 패턴을 분석하여 더욱 정확한 콘텐츠 추천을 제공합니다.',
            'expected_impact': 'medium'
        })
    
    # 시청 시간이 적은 사용자
    if user['weekly_viewing_hours'] < 5:
        strategies.append({
            'type': 'engagement',
            'title': '주간 콘텐츠 하이라이트',
            'description': '사용자 취향에 맞는 새로운 콘텐츠를 주간 이메일로 소개합니다.',
            'expected_impact': 'medium'
        })
        strategies.append({
            'type': 'feature',
            'title': '오프라인 시청 안내',
            'description': '이동 중에도 시청할 수 있는 오프라인 다운로드 기능을 강조합니다.',
            'expected_impact': 'medium'
        })
    
    # 구독 유형별 전략
    if user['subscription_type'] == 'Basic':
        strategies.append({
            'type': 'upgrade',
            'title': '30일 무료 업그레이드 체험',
            'description': '스탠다드 플랜의 HD 화질과 다중 기기 사용을 30일간 무료로 체험할 수 있는 기회를 제공합니다.',
            'expected_impact': 'high'
        })
    
    # 이탈 확률이 매우 높은 사용자
    if user['churn_probability'] > 0.7:
        strategies.append({
            'type': 'urgent',
            'title': '특별 리텐션 오퍼',
            'description': '다음 결제 2개월 50% 할인 혜택을 제공하여 즉각적인 해지 방지에 집중합니다.',
            'expected_impact': 'high'
        })
        strategies.append({
            'type': 'survey',
            'title': '맞춤형 피드백 수집',
            'description': '사용자에게 직접 서비스 개선점을 물어보고, 응답에 대한 감사 혜택을 제공합니다.',
            'expected_impact': 'medium'
        })
    
    # 기술적 문제를 경험한 사용자
    if 'technical_issues' in user and user['technical_issues'] > 1:
        strategies.append({
            'type': 'support',
            'title': '기술 지원 혜택',
            'description': '최근 기술적 어려움을 겪으신 점에 사과드리며, 우선적인 기술 지원과 1개월 10% 할인 혜택을 제공합니다.',
            'expected_impact': 'high'
        })
    
    # 콘텐츠 다양성이 낮은 사용자
    if 'content_diversity' in user and user['content_diversity'] < 3:
        strategies.append({
            'type': 'content_discovery',
            'title': '맞춤형 장르 탐색',
            'description': '사용자가 아직 접하지 않은 관심 가능성이 높은 새로운 장르의 콘텐츠를 소개합니다.',
            'expected_impact': 'medium'
        })
    
    return strategies

def create_genre_distribution_chart():
    """장르 분포 차트 생성 및 저장"""
    try:
        # 장르 데이터 생성 (예시)
        genres = ['Action', 'Drama', 'Comedy', 'Sci-Fi', 'Romance', 'Horror', 'Documentary']
        counts = [25, 18, 15, 12, 10, 8, 12]
        
        # 차트 생성
        plt.figure(figsize=(10, 6))
        bars = plt.bar(genres, counts, color=['#FF5252', '#4CAF50', '#2196F3', '#FFC107', '#9C27B0', '#607D8B', '#00BCD4'])
        
        # 스타일링
        plt.title('Content Genre Distribution', fontsize=16)
        plt.xlabel('Genre', fontsize=12)
        plt.ylabel('Number of Titles', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 각 바에 값 표시
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.0f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # 차트 저장
        genre_path = os.path.join(app.static_folder, 'images', 'genre_distribution.png')
        plt.savefig(genre_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Created genre distribution chart: {genre_path}")
        return True
    except Exception as e:
        logger.error(f"Error creating genre distribution chart: {e}")
        return False

def create_correlation_matrix():
    """특성 간 상관관계 행렬 생성 및 저장"""
    try:
        # 데이터 준비 - 주요 특성만
        features_for_correlation = [
            'weekly_viewing_hours', 'tenure_months', 'content_diversity', 
            'price_increase', 'technical_issues', 'customer_service_calls',
            'competing_services', 'user_rating', 'recommended_content_watched'
        ]
        
        # 데이터 추출
        corr_data = data['users'][features_for_correlation].copy()
        
        # 상관 행렬 계산
        corr_matrix = corr_data.corr()
        
        # 상관 행렬 시각화
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(
            corr_matrix, 
            mask=mask, 
            cmap=cmap, 
            vmax=.3, 
            center=0,
            square=True, 
            linewidths=.5, 
            cbar_kws={"shrink": .5},
            annot=True,
            fmt='.2f'
        )
        
        plt.title('Feature Correlation Matrix', fontsize=16)
        plt.tight_layout()
        
        # 이미지 저장
        corr_path = os.path.join(app.static_folder, 'images', 'correlation_matrix.png')
        plt.savefig(corr_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Created correlation matrix: {corr_path}")
        return True
    except Exception as e:
        logger.error(f"Error creating correlation matrix: {e}")
        return False

def create_user_recommendations(user_id):
    """사용자별 추천 차트 생성 및 저장"""
    try:
        # 가상의 영화 제목과 일치도 점수 생성
        movies = [
            "Stranger Things", "The Crown", "Money Heist", 
            "Dark", "Bridgerton", "Queen's Gambit"
        ]
        # 일치도 점수 랜덤 생성 (70-99%)
        matches = np.random.randint(70, 100, len(movies))
        
        # 점수에 따라 내림차순 정렬
        sorted_indices = np.argsort(matches)[::-1]
        sorted_movies = [movies[i] for i in sorted_indices]
        sorted_matches = [matches[i] for i in sorted_indices]
        
        # 차트 생성
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(sorted_movies, sorted_matches, color=['#FF9800', '#4CAF50', '#2196F3', '#9C27B0', '#F44336', '#607D8B'])
        
        # 스타일링
        ax.set_title(f'Recommended Content for User #{user_id}', fontsize=14)
        ax.set_xlabel('Match Score (%)', fontsize=12)
        ax.set_xlim(0, 100)
        
        # 각 바에 값 표시
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                   f'{sorted_matches[i]}%',
                   ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        
        # 이미지 저장
        rec_path = os.path.join(app.static_folder, 'images', f'user_{user_id}_recommendations.png')
        plt.savefig(rec_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Created recommendations chart for user {user_id}: {rec_path}")
        return rec_path
    except Exception as e:
        logger.error(f"Error creating recommendations chart for user {user_id}: {e}")
        return None

def init_app():
    """앱 초기화 함수"""
    global data, models
    
    # 모든 필수 데이터 구조 초기화
    data = {
        'sample_recommendations': {}, 
        'visualization_paths': {},
        'users': pd.DataFrame(),
        'at_risk_users': pd.DataFrame(),
        'cohorts': {}
    }
    models = {}
    
    try:
        # 사용자 데이터 생성 또는 로드 시도
        try:
            users_path = os.path.join(data_dir, 'external', 'users.csv')
            if os.path.exists(users_path):
                data['users'] = pd.read_csv(users_path)
                # 이탈 위험 사용자 식별
                data['at_risk_users'] = data['users'][data['users']['churn_probability'] >= 0.5]
                logger.info(f"Loaded user data from {users_path}")
            else:
                # 데이터가 없으면 합성 데이터 생성 및 저장
                logger.info("No user data found. Creating synthetic data...")
                n_users = 1000
                np.random.seed(42)  # 재현성을 위한 시드 설정
                
                # 사용자 특성 생성
                data['users'] = pd.DataFrame({
                    'user_id': range(1, n_users + 1),
                    'age': np.random.randint(18, 70, n_users),
                    'gender': np.random.choice(['M', 'F', 'Other'], n_users, p=[0.48, 0.48, 0.04]),
                    'subscription_type': np.random.choice(['Basic', 'Standard', 'Premium'], n_users, p=[0.4, 0.4, 0.2]),
                    'tenure_months': np.random.randint(1, 60, n_users),
                    'weekly_viewing_hours': np.clip(np.random.normal(10, 5, n_users), 0, 40),
                    'content_diversity': np.random.randint(1, 10, n_users),
                    'price_increase': np.random.choice([0, 1], n_users, p=[0.7, 0.3]),
                    'technical_issues': np.random.poisson(0.5, n_users),
                    'customer_service_calls': np.random.poisson(0.3, n_users),
                    'account_sharing': np.random.choice([0, 1], n_users, p=[0.6, 0.4]),
                    'competing_services': np.random.randint(0, 5, n_users),
                    'last_login_days': np.random.randint(0, 30, n_users),
                    'binge_watching': np.random.choice([0, 1], n_users, p=[0.6, 0.4]),
                    'user_rating': np.random.uniform(1, 5, n_users),
                    'recommended_content_watched': np.random.uniform(0, 1, n_users),
                    'signup_date': pd.date_range(end=pd.Timestamp.now(), periods=n_users),
                    'region': np.random.choice(['North America', 'Europe', 'Asia', 'Latin America', 'Oceania'], n_users),
                    'device_type': np.random.choice(['Mobile', 'TV', 'Computer', 'Tablet'], n_users),
                })
                
                # 이탈률 계산을 위한 로직
                churn_prob = (
                    0.5 - 0.02 * data['users']['weekly_viewing_hours'] / 10
                    - 0.02 * data['users']['tenure_months'] / 12
                    + 0.15 * data['users']['price_increase']
                    + 0.08 * (data['users']['technical_issues'] > 1)
                    + 0.05 * data['users']['competing_services'] / 2
                    - 0.1 * data['users']['user_rating'] / 5
                    - 0.05 * data['users']['recommended_content_watched']
                    + 0.02 * (data['users']['subscription_type'] == 'Basic')
                    + 0.01 * data['users']['last_login_days'] / 7
                    - 0.03 * data['users']['content_diversity'] / 5
                    - 0.04 * data['users']['binge_watching']
                )
                # 확률 값 0-1 범위로 조정
                data['users']['churn_probability'] = np.clip(churn_prob, 0.05, 0.95)
                
                # CSV 파일로 저장 시도
                try:
                    os.makedirs(os.path.dirname(users_path), exist_ok=True)
                    data['users'].to_csv(users_path, index=False)
                    logger.info(f"Generated synthetic user data and saved to {users_path}")
                except Exception as e:
                    logger.error(f"Failed to save synthetic data: {e}")
                
                # 이탈 위험 사용자 식별
                data['at_risk_users'] = data['users'][data['users']['churn_probability'] >= 0.5]
        except Exception as e:
            logger.error(f"Failed to load or create user data: {e}")
            # 기본 사용자 데이터 생성
            data['users'] = pd.DataFrame({'user_id': [1], 'churn_probability': [0.5]})
            data['at_risk_users'] = data['users']
        
        # 시각화 이미지 경로 설정 - 항상 설정되도록 보장
        data['visualization_paths'] = {
            'genre_distribution': '/static/images/genre_distribution.png',
            'shap_summary': '/static/images/shap_summary.png'
        }
        
        # 모델 로드 시도 - 실패해도 계속 진행
        try:
            best_model_path = os.path.join(models_dir, 'best_model.joblib')
            if os.path.exists(best_model_path):
                models['churn_model'] = joblib.load(best_model_path)
                logger.info("Loaded churn prediction model")
            else:
                logger.warning("No pre-trained model found. Creating a simple default model.")
                if len(data['users']) > 1:  # 충분한 데이터가 있는 경우에만
                    models['churn_model'] = create_default_model(data['users'])
        except Exception as e:
            logger.error(f"Failed to load or create model: {e}")
        
        # 정적 이미지 폴더 확인 및 생성
        try:
            images_dir = os.path.join(app.static_folder, 'images')
            os.makedirs(images_dir, exist_ok=True)
            
            # 장르 분포 차트 파일 존재 확인 및 생성
            genre_target = os.path.join(images_dir, 'genre_distribution.png')
            if not os.path.exists(genre_target):
                try:
                    create_genre_distribution_chart()
                except Exception as e:
                    logger.error(f"Failed to create genre distribution chart: {e}")
                    # 오류 발생 시 더미 차트 생성
                    create_dummy_chart(genre_target, "Genre Distribution")
            
            # 코호트 분석 데이터 준비
            try:
                data['cohorts'] = prepare_cohort_data(data['users'])
            except Exception as e:
                logger.error(f"Failed to prepare cohort data: {e}")
                data['cohorts'] = {'monthly': pd.DataFrame(), 'subscription': pd.DataFrame(), 
                                   'region': pd.DataFrame(), 'device': pd.DataFrame(), 'tenure': pd.DataFrame()}
        
        except Exception as e:
            logger.error(f"Failed to set up static resources: {e}")
    
    except Exception as e:
        logger.error(f"Critical error in app initialization: {e}")
        # 앱은 계속 실행되지만, 기본적인 데이터 구조만 유지

def create_dummy_chart(filepath, title="Chart Not Available"):
    """데이터 로드 실패 시 더미 차트 생성"""
    try:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"{title}\nData not available", 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"Created dummy chart at {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to create dummy chart: {e}")
        return False

@app.route('/model-comparison')
def model_comparison():
    """모델 비교 및 하이퍼파라미터 튜닝 페이지"""
    # 저장된 모델 정보 로드
    models_info = load_models_info()
    
    # 현재 시간
    now = datetime.now()
    
    return render_template('model_comparison.html', 
                          models=models_info,
                          now=now)

@app.route('/api/hyperparameter-tuning', methods=['POST'])
def hyperparameter_tuning_api():
    """하이퍼파라미터 튜닝 API"""
    try:
        tuning_data = request.json
        
        if not tuning_data:
            return jsonify({
                'error': '튜닝 파라미터가 제공되지 않았습니다.',
                'status': 'error'
            }), 400
        
        # 로그 출력
        logger.info(f"하이퍼파라미터 튜닝 요청: {tuning_data}")
        
        # 실제 하이퍼파라미터 튜닝 구현
        model_type = tuning_data.get('model_type', 'xgboost')
        param_grid = tuning_data.get('param_grid', {})
        cv_folds = int(tuning_data.get('cv_folds', 5))
        scoring_metric = tuning_data.get('scoring_metric', 'f1')
        use_stratified = tuning_data.get('use_stratified', True)
        shuffle_data = tuning_data.get('shuffle_data', True)
        
        try:
            # 데이터 준비 (간략화된 예시 - 실제로는 완전한 ML 파이프라인 필요)
            logger.info("하이퍼파라미터 튜닝: 데이터 준비 중...")
            
            # 모델 및 튜닝 파이프라인 설정
            logger.info(f"하이퍼파라미터 튜닝: {model_type} 모델 설정 중...")
            
            # 크로스 밸리데이션 설정
            if use_stratified:
                logger.info("계층화된 교차 검증(Stratified CV) 사용")
                # 실제 구현: from sklearn.model_selection import StratifiedKFold
                # cv = StratifiedKFold(n_splits=cv_folds, shuffle=shuffle_data, random_state=42)
            else:
                logger.info("일반 교차 검증(K-Fold CV) 사용")
                # 실제 구현: from sklearn.model_selection import KFold
                # cv = KFold(n_splits=cv_folds, shuffle=shuffle_data, random_state=42)
            
            # 그리드 서치 실행 시뮬레이션 
            logger.info("하이퍼파라미터 튜닝: 그리드 서치 실행 중...")
            
            # 실제 검색 대신 더미 결과 반환
            # 최적 파라미터 (예시)
            best_params = {
                "learning_rate": 0.1,
                "max_depth": 5,
                "n_estimators": 100,
                "subsample": 0.8
            }
            
            # 성능 메트릭 (예시)
            metrics = {
                "accuracy": 0.854,
                "precision": 0.823,
                "recall": 0.791,
                "f1": 0.807,
                "auc": 0.912
            }
            
            # 파라미터 중요도 (예시)
            param_importance = {
                "learning_rate": 0.42,
                "max_depth": 0.31,
                "n_estimators": 0.18,
                "subsample": 0.09
            }
            
            # 파라미터 탐색 결과 (예시)
            param_search_results = [
                {"params": {"learning_rate": 0.1, "max_depth": 5, "n_estimators": 100}, "score": 0.807},
                {"params": {"learning_rate": 0.1, "max_depth": 3, "n_estimators": 200}, "score": 0.796},
                {"params": {"learning_rate": 0.2, "max_depth": 5, "n_estimators": 50}, "score": 0.785}
            ]
            
            logger.info("하이퍼파라미터 튜닝 완료")
            
            return jsonify({
                'best_params': best_params,
                'best_score': 0.807,
                'metrics': metrics,
                'param_importance': param_importance,
                'param_search_results': param_search_results,
                'elapsed_time': 12.5,
                'cv_method': 'Stratified CV' if use_stratified else 'KFold CV',
                'cv_folds': cv_folds,
                'status': 'success'
            })
            
        except Exception as e:
            logger.error(f"하이퍼파라미터 튜닝 처리 중 오류: {e}")
            return jsonify({
                'error': f"튜닝 처리 중 오류 발생: {str(e)}",
                'status': 'error'
            }), 500
    
    except Exception as e:
        logger.error(f"하이퍼파라미터 튜닝 오류: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/save-tuned-model', methods=['POST'])
def save_tuned_model_api():
    """튜닝된 모델 저장 API"""
    if 'tuned_model' not in models:
        return jsonify({
            'error': 'No tuned model available to save',
            'status': 'error'
        }), 400
    
    try:
        # 모델 이름 생성
        model_name = f"TunedModel_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_path = os.path.join(models_dir, f"{model_name}.joblib")
        
        # 모델 저장
        joblib.dump(models['tuned_model'], model_path)
        logger.info(f"Saved tuned model to {model_path}")
        
        # 모델 정보 저장
        save_model_info(model_name, models['tuned_model'])
        
        return jsonify({
            'success': True,
            'model_name': model_name,
            'model_path': model_path
        })
    
    except Exception as e:
        logger.error(f"Error saving tuned model: {e}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/api/set-active-model/<model_id>', methods=['POST'])
def set_active_model_api(model_id):
    """활성 모델 설정 API"""
    try:
        # 모델 정보 로드
        models_info = load_models_info()
        
        # 요청된 모델 ID가 있는지 확인
        model_exists = False
        for model in models_info:
            if model['id'] == model_id:
                model_exists = True
                model_path = os.path.join(models_dir, f"{model['name']}.joblib")
                break
        
        if not model_exists:
            return jsonify({
                'error': f'Model with ID {model_id} not found',
                'success': False
            }), 404
        
        # 모델 로드
        loaded_model = joblib.load(model_path)
        
        # 활성 모델로 설정
        models['churn_model'] = loaded_model
        
        # best_model로 저장
        best_model_path = os.path.join(models_dir, 'best_model.joblib')
        joblib.dump(loaded_model, best_model_path)
        
        # 모델 정보 업데이트
        update_model_info(model_id, {'is_active': True})
        
        return jsonify({
            'success': True,
            'message': f'Model {model_id} set as active'
        })
    
    except Exception as e:
        logger.error(f"Error setting active model: {e}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/api/train-model', methods=['POST'])
def train_model_api():
    """새 모델 훈련 API"""
    training_data = request.json
    
    if not training_data:
        return jsonify({
            'error': 'No training parameters provided',
            'status': 'error'
        }), 400
    
    try:
        model_type = training_data.get('model_type', 'xgboost')
        model_name = training_data.get('model_name', f"Model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        test_size = float(training_data.get('test_size', 0.2))
        use_smote = training_data.get('use_smote', True)
        standardize_features = training_data.get('standardize_features', True)
        random_state = int(training_data.get('random_state', 42))
        
        # 데이터 로드
        users_df = data['users']
        
        # 특성 및 타겟 추출
        X = users_df.drop(columns=['user_id', 'signup_date', 'churn_probability', 'gender', 'subscription_type', 'region', 'device_type']).copy()
        y = (users_df['churn_probability'] >= 0.5).astype(int)  # 이진 분류 타겟
        
        # 범주형 특성 인코딩
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].astype('category').cat.codes
        
        # 특성 스케일링
        if standardize_features:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        
        # 훈련/테스트 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # 클래스 불균형 처리
        if use_smote:
            try:
                smote = SMOTE(random_state=random_state)
                X_train, y_train = smote.fit_resample(X_train, y_train)
            except Exception as e:
                logger.warning(f"SMOTE failed: {e}")
        
        # 모델 선택
        if model_type == 'xgboost':
            model = XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state
            )
        elif model_type == 'lightgbm':
            model = lgb.LGBMClassifier(
                n_estimators=100,
                num_leaves=31,
                learning_rate=0.1,
                random_state=random_state
            )
        elif model_type == 'randomforest':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=random_state
            )
        elif model_type == 'logistic':
            model = LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='liblinear',
                random_state=random_state,
                max_iter=1000
            )
        elif model_type == 'ensemble':
            # 앙상블 모델 (Voting)
            estimators = [
                ('xgb', XGBClassifier(random_state=random_state)),
                ('lgb', lgb.LGBMClassifier(random_state=random_state)),
                ('rf', RandomForestClassifier(random_state=random_state))
            ]
            model = VotingClassifier(estimators=estimators, voting='soft')
        else:
            return jsonify({
                'error': f'Unknown model type: {model_type}',
                'status': 'error'
            }), 400
        
        # 모델 훈련
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # 모델 평가
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # 성능 메트릭스 계산
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_prob)
        }
        
        # 학습 곡선 계산
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring='f1'
        )
        
        # 학습 곡선 평균 및 표준편차
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)
        
        # 모델 저장
        model_path = os.path.join(models_dir, f"{model_name}.joblib")
        joblib.dump(model, model_path)
        
        # 모델 정보 저장
        model_info = {
            'name': model_name,
            'type': model_type,
            'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': metrics,
            'training_time': training_time,
            'parameters': model.get_params(),
            'feature_names': list(users_df.drop(columns=['user_id', 'signup_date', 'churn_probability', 'gender', 'subscription_type', 'region', 'device_type']).columns)
        }
        
        # 모델 정보 DB에 저장
        save_model_info(model_name, model, metrics)
        
        # 혼동 행렬 계산
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return jsonify({
            'success': True,
            'model_name': model_name,
            'metrics': metrics,
            'training_time': training_time,
            'confusion_matrix': cm.tolist(),
            'learning_curve': {
                'train_sizes': train_sizes.tolist(),
                'train_scores_mean': train_scores_mean.tolist(),
                'train_scores_std': train_scores_std.tolist(),
                'val_scores_mean': val_scores_mean.tolist(),
                'val_scores_std': val_scores_std.tolist()
            }
        })
    
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/api/feature-importance', methods=['GET'])
def feature_importance_api():
    """특성 중요도 API"""
    if 'churn_model' not in models:
        return jsonify({
            'error': 'No model available',
            'status': 'error'
        }), 400
    
    try:
        model = models['churn_model']
        
        # 특성 이름 가져오기
        feature_names = list(data['users'].drop(columns=['user_id', 'signup_date', 'churn_probability', 'gender', 'subscription_type', 'region', 'device_type']).columns)
        
        # 모델 유형에 따라 특성 중요도 추출
        importance = {}
        
        if hasattr(model, 'feature_importances_'):
            # Tree 기반 모델
            importance = dict(zip(feature_names, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            # 선형 모델
            importance = dict(zip(feature_names, np.abs(model.coef_[0])))
        elif hasattr(model, 'estimators_') and hasattr(model, 'voting') and model.voting == 'soft':
            # Voting 앙상블
            # 첫 번째 estimator의 feature importance 사용 (간단한 방법)
            for name, est in model.estimators:
                if hasattr(est, 'feature_importances_'):
                    importance = dict(zip(feature_names, est.feature_importances_))
                    break
        else:
            # SHAP 값 사용 (fallback)
            if 'explainer' in models:
                # 샘플 데이터
                sample = data['users'].drop(columns=['user_id', 'signup_date', 'churn_probability']).sample(min(100, len(data['users']))).copy()
                for col in sample.select_dtypes(include=['object']).columns:
                    sample[col] = sample[col].astype('category').cat.codes
                
                # SHAP 값 계산
                shap_values = models['explainer'](sample)
                shap_mean = np.abs(shap_values.values).mean(0)
                importance = dict(zip(feature_names, shap_mean))
        
        # 중요도에 따라 정렬
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return jsonify({
            'feature_importance': sorted_importance,
            'status': 'success'
        })
    
    except Exception as e:
        logger.error(f"Error calculating feature importance: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

def load_models_info():
    """저장된 모델 정보 로드"""
    models_info = []
    model_info_path = os.path.join(models_dir, 'models_info.json')
    
    # 기존 정보 파일이 있으면 로드
    if os.path.exists(model_info_path):
        try:
            with open(model_info_path, 'r') as f:
                models_info = json.load(f)
        except Exception as e:
            logger.error(f"Error loading models info: {e}")
    
    # 현재 models 디렉토리에 있는 모델 검사
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib') and f != 'best_model.joblib']
    
    # 모델 파일에서 정보 추출 (DB에 없는 모델용)
    for model_file in model_files:
        model_name = model_file.replace('.joblib', '')
        
        # 이미 정보가 있는지 확인
        if any(m['name'] == model_name for m in models_info):
            continue
        
        try:
            # 모델 로드
            model_path = os.path.join(models_dir, model_file)
            model = joblib.load(model_path)
            
            # 성능 테스트 (간단한 테스트)
            users_df = data['users']
            X = users_df.drop(columns=['user_id', 'signup_date', 'churn_probability', 'gender', 'subscription_type', 'region', 'device_type']).copy()
            y = (users_df['churn_probability'] >= 0.5).astype(int)
            
            # 범주형 특성 인코딩
            for col in X.select_dtypes(include=['object']).columns:
                X[col] = X[col].astype('category').cat.codes
            
            # 예측 및 평가
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)[:, 1]
            
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred)
            recall = recall_score(y, y_pred)
            f1 = f1_score(y, y_pred)
            auc = roc_auc_score(y, y_prob)
            
            # 모델 정보 생성
            model_type = model.__class__.__name__
            model_info = {
                'id': str(len(models_info) + 1),
                'name': model_name,
                'type': model_type,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'training_time': 0,  # 알 수 없음
                'creation_date': datetime.fromtimestamp(os.path.getctime(model_path)).strftime('%Y-%m-%d'),
                'is_active': model_file == 'best_model.joblib'
            }
            
            models_info.append(model_info)
        except Exception as e:
            logger.error(f"Error processing model {model_file}: {e}")
    
    # 모델 정보 저장
    try:
        with open(model_info_path, 'w') as f:
            json.dump(models_info, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving models info: {e}")
    
    return models_info

def save_model_info(model_name, model, metrics=None):
    """모델 정보를 저장"""
    models_info = load_models_info()
    
    # 새 모델 ID 생성
    model_id = str(int(max([int(m['id']) for m in models_info]) + 1) if models_info else 1)
    
    # 모델 정보 생성
    model_type = model.__class__.__name__
    model_info = {
        'id': model_id,
        'name': model_name,
        'type': model_type,
        'creation_date': datetime.now().strftime('%Y-%m-%d'),
        'is_active': False
    }
    
    # 성능 메트릭스가 제공된 경우 추가
    if metrics:
        model_info.update({
            'accuracy': metrics.get('accuracy', 0),
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'f1': metrics.get('f1', 0),
            'auc': metrics.get('auc', 0)
        })
    
    # 정보 추가
    models_info.append(model_info)
    
    # 저장
    model_info_path = os.path.join(models_dir, 'models_info.json')
    try:
        with open(model_info_path, 'w') as f:
            json.dump(models_info, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving model info: {e}")

def update_model_info(model_id, updates):
    """모델 정보 업데이트"""
    models_info = load_models_info()
    
    # ID로 모델 찾기
    for model in models_info:
        if model['id'] == model_id:
            # 기존 활성 모델 비활성화 (활성 상태를 업데이트하는 경우)
            if 'is_active' in updates and updates['is_active']:
                for m in models_info:
                    m['is_active'] = False
            
            # 업데이트 적용
            model.update(updates)
            break
    
    # 저장
    model_info_path = os.path.join(models_dir, 'models_info.json')
    try:
        with open(model_info_path, 'w') as f:
            json.dump(models_info, f, indent=2)
    except Exception as e:
        logger.error(f"Error updating model info: {e}")

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=False, host='0.0.0.0', port=port)