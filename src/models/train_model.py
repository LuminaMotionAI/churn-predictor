"""
머신러닝 이탈 예측 모델 훈련 스크립트
=====================================
이 스크립트는 다양한 머신러닝 모델을 훈련하고 최적화하는 기능을 제공합니다.
"""
import os
import joblib
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import lightgbm as lgb

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(train_path, val_path, test_path):
    """훈련, 검증, 테스트 데이터 로드"""
    logger.info("Loading training, validation, and test data")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    return train_df, val_df, test_df

def prepare_data(train_df, val_df, test_df, target_col='churn'):
    """데이터 준비: 특성과 타겟 분리"""
    logger.info("Preparing data for model training")
    
    # 타겟 변수와 특성 분리
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    
    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]
    
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    # 컬럼 일치 확인 및 조정
    train_cols = X_train.columns
    X_val = X_val[train_cols]
    X_test = X_test[train_cols]
    
    logger.info(f"Data prepared with {X_train.shape[1]} features")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def evaluate_model(model, X, y, dataset_name=""):
    """모델 평가 함수"""
    # 예측 수행
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # 다양한 평가 지표 계산
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_pred_proba)
    }
    
    # 결과 로깅
    logger.info(f"{dataset_name} metrics:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    return metrics

def train_logistic_regression(X_train, y_train, X_val, y_val):
    """로지스틱 회귀 모델 훈련"""
    logger.info("Training Logistic Regression model")
    
    # 하이퍼파라미터 그리드
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
        'class_weight': [None, 'balanced']
    }
    
    # 그리드 서치
    grid_search = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    # 최적 모델
    best_model = grid_search.best_estimator_
    logger.info(f"Best logistic regression parameters: {grid_search.best_params_}")
    
    # 검증 세트에서 평가
    val_metrics = evaluate_model(best_model, X_val, y_val, "Validation (Logistic Regression)")
    
    return best_model, val_metrics

def train_random_forest(X_train, y_train, X_val, y_val):
    """랜덤 포레스트 모델 훈련"""
    logger.info("Training Random Forest model")
    
    # 하이퍼파라미터 그리드
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': [None, 'balanced']
    }
    
    # 그리드 서치
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    # 최적 모델
    best_model = grid_search.best_estimator_
    logger.info(f"Best random forest parameters: {grid_search.best_params_}")
    
    # 검증 세트에서 평가
    val_metrics = evaluate_model(best_model, X_val, y_val, "Validation (Random Forest)")
    
    return best_model, val_metrics

def train_gradient_boosting(X_train, y_train, X_val, y_val):
    """그래디언트 부스팅 모델 훈련"""
    logger.info("Training Gradient Boosting model")
    
    # 하이퍼파라미터 그리드
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0],
        'min_samples_split': [2, 5]
    }
    
    # 그리드 서치
    grid_search = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    # 최적 모델
    best_model = grid_search.best_estimator_
    logger.info(f"Best gradient boosting parameters: {grid_search.best_params_}")
    
    # 검증 세트에서 평가
    val_metrics = evaluate_model(best_model, X_val, y_val, "Validation (Gradient Boosting)")
    
    return best_model, val_metrics

def train_xgboost(X_train, y_train, X_val, y_val):
    """XGBoost 모델 훈련"""
    logger.info("Training XGBoost model")
    
    # 하이퍼파라미터 그리드
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'scale_pos_weight': [1, sum(y_train==0)/sum(y_train==1)]
    }
    
    # 그리드 서치
    grid_search = GridSearchCV(
        XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    # 최적 모델
    best_model = grid_search.best_estimator_
    logger.info(f"Best XGBoost parameters: {grid_search.best_params_}")
    
    # 검증 세트에서 평가
    val_metrics = evaluate_model(best_model, X_val, y_val, "Validation (XGBoost)")
    
    return best_model, val_metrics

def train_lightgbm(X_train, y_train, X_val, y_val):
    """LightGBM 모델 훈련"""
    logger.info("Training LightGBM model")
    
    # 하이퍼파라미터 그리드
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [0, 0.1],
        'class_weight': [None, 'balanced']
    }
    
    # 그리드 서치
    grid_search = GridSearchCV(
        lgb.LGBMClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    # 최적 모델
    best_model = grid_search.best_estimator_
    logger.info(f"Best LightGBM parameters: {grid_search.best_params_}")
    
    # 검증 세트에서 평가
    val_metrics = evaluate_model(best_model, X_val, y_val, "Validation (LightGBM)")
    
    return best_model, val_metrics

def get_feature_importance(model, feature_names, model_name):
    """특성 중요도 추출"""
    # 모델 타입에 따라 다른 방식으로 특성 중요도 가져오기
    if model_name == 'LogisticRegression':
        importance = np.abs(model.coef_[0])
    elif model_name in ['RandomForest', 'GradientBoosting', 'XGBoost', 'LightGBM']:
        importance = model.feature_importances_
    else:
        return None
    
    # 중요도를 데이터프레임으로 변환
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    
    # 중요도 기준 정렬
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    return feature_importance

def save_model(model, metrics, feature_importance, model_dir, model_name):
    """모델, 평가 지표, 특성 중요도 저장"""
    # 디렉토리 생성
    os.makedirs(model_dir, exist_ok=True)
    
    # 모델 저장
    model_path = os.path.join(model_dir, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # 평가 지표 저장
    metrics_path = os.path.join(model_dir, f"{model_name}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # 특성 중요도 저장
    if feature_importance is not None:
        importance_path = os.path.join(model_dir, f"{model_name}_feature_importance.csv")
        feature_importance.to_csv(importance_path, index=False)
        logger.info(f"Feature importance saved to {importance_path}")

def train_all_models(X_train, y_train, X_val, y_val, X_test, y_test, model_dir):
    """모든 모델 훈련 및 평가"""
    models = {}
    metrics = {}
    feature_names = X_train.columns
    
    # 로지스틱 회귀 모델
    models['LogisticRegression'], metrics['LogisticRegression'] = train_logistic_regression(X_train, y_train, X_val, y_val)
    
    # 랜덤 포레스트 모델
    models['RandomForest'], metrics['RandomForest'] = train_random_forest(X_train, y_train, X_val, y_val)
    
    # 그래디언트 부스팅 모델
    models['GradientBoosting'], metrics['GradientBoosting'] = train_gradient_boosting(X_train, y_train, X_val, y_val)
    
    # XGBoost 모델
    models['XGBoost'], metrics['XGBoost'] = train_xgboost(X_train, y_train, X_val, y_val)
    
    # LightGBM 모델
    models['LightGBM'], metrics['LightGBM'] = train_lightgbm(X_train, y_train, X_val, y_val)
    
    # 테스트 데이터에서 모든 모델 평가
    test_metrics = {}
    for model_name, model in models.items():
        test_metrics[model_name] = evaluate_model(model, X_test, y_test, f"Test ({model_name})")
        
        # 특성 중요도 추출 및 저장
        feature_importance = get_feature_importance(model, feature_names, model_name)
        
        # 모델 및 관련 정보 저장
        save_model(model, test_metrics[model_name], feature_importance, model_dir, model_name)
    
    # 최고 성능 모델 선택 (ROC AUC 기준)
    best_model_name = max(test_metrics, key=lambda k: test_metrics[k]['roc_auc'])
    best_model = models[best_model_name]
    
    logger.info(f"Best model: {best_model_name} with ROC AUC = {test_metrics[best_model_name]['roc_auc']:.4f}")
    
    # 최고 모델 별도 저장
    joblib.dump(best_model, os.path.join(model_dir, "best_model.joblib"))
    
    return models, test_metrics, best_model_name

def main():
    """메인 실행 함수"""
    # 프로젝트 경로 설정
    project_dir = Path(__file__).resolve().parents[2]
    processed_data_dir = os.path.join(project_dir, 'data', 'processed')
    model_dir = os.path.join(project_dir, 'models')
    
    # 디렉토리 생성
    os.makedirs(model_dir, exist_ok=True)
    
    # 특성 엔지니어링이 적용된 데이터 경로
    train_path = os.path.join(processed_data_dir, 'train_featured.csv')
    val_path = os.path.join(processed_data_dir, 'val_featured.csv')
    test_path = os.path.join(processed_data_dir, 'test_featured.csv')
    
    # 데이터 로드
    train_df, val_df, test_df = load_data(train_path, val_path, test_path)
    
    # 데이터 준비
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(train_df, val_df, test_df)
    
    # 모델 훈련 및 평가
    models, metrics, best_model_name = train_all_models(X_train, y_train, X_val, y_val, X_test, y_test, model_dir)
    
    logger.info('✓ Model training completed successfully')

if __name__ == '__main__':
    main()