"""
딥러닝 이탈 예측 모델 훈련 스크립트
==================================
이 스크립트는 TensorFlow/Keras를 활용한 딥러닝 모델을 구성하고 훈련합니다.
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# GPU 메모리 할당 설정
def setup_gpu():
    """GPU 메모리 설정"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s)")
        except RuntimeError as e:
            logger.error(f"Error setting up GPU: {e}")

def load_data(train_path, val_path, test_path):
    """훈련, 검증, 테스트 데이터 로드"""
    logger.info("Loading training, validation, and test data")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    return train_df, val_df, test_df

def prepare_data(train_df, val_df, test_df, target_col='churn'):
    """데이터 준비: 특성과 타겟 분리, 스케일링"""
    logger.info("Preparing data for deep learning model")
    
    # 타겟 변수와 특성 분리
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].values
    
    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col].values
    
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].values
    
    # 컬럼 일치 확인 및 조정
    train_cols = X_train.columns
    X_val = X_val[train_cols]
    X_test = X_test[train_cols]
    
    # 표준화 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"Data prepared with {X_train.shape[1]} features")
    
    return (X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, 
            X_train.columns.tolist())

def create_dnn_model(input_dim, hidden_units=[128, 64, 32], dropout_rate=0.3):
    """기본 DNN 모델 생성"""
    logger.info("Creating DNN model")
    
    model = Sequential()
    
    # 입력층
    model.add(Input(shape=(input_dim,)))
    
    # 히든 레이어
    for units in hidden_units:
        model.add(Dense(units, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    
    # 출력층
    model.add(Dense(1, activation='sigmoid'))
    
    # 모델 컴파일
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary(print_fn=logger.info)
    
    return model

def create_wide_and_deep_model(input_dim, wide_cols=None, deep_cols=None):
    """Wide & Deep 모델 생성"""
    logger.info("Creating Wide & Deep model")
    
    if wide_cols is None or deep_cols is None:
        # 별도 지정이 없으면 모든 특성을 deep 부분에 사용
        wide_cols = list(range(input_dim))
        deep_cols = list(range(input_dim))
    
    # 입력층
    input_layer = Input(shape=(input_dim,))
    
    # Wide 부분
    wide = Dense(16, activation='relu')(input_layer)
    wide = Dense(8, activation='relu')(wide)
    
    # Deep 부분
    deep = Dense(128, activation='relu')(input_layer)
    deep = BatchNormalization()(deep)
    deep = Dropout(0.3)(deep)
    deep = Dense(64, activation='relu')(deep)
    deep = BatchNormalization()(deep)
    deep = Dropout(0.3)(deep)
    deep = Dense(32, activation='relu')(deep)
    
    # Wide와 Deep 부분 합치기 (Merge)
    merged = tf.keras.layers.concatenate([wide, deep])
    
    # 출력층
    output = Dense(1, activation='sigmoid')(merged)
    
    # 모델 정의 및 컴파일
    model = Model(inputs=input_layer, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary(print_fn=logger.info)
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=100, model_dir=None, model_name="deep_model"):
    """모델 훈련"""
    logger.info(f"Training {model_name}")
    
    # 콜백 정의
    callbacks = []
    
    # 얼리 스토핑
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    callbacks.append(early_stopping)
    
    # 모델 체크포인트 (최고 모델 저장)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        checkpoint_path = os.path.join(model_dir, f"{model_name}_best.h5")
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint)
    
    # 모델 훈련
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2
    )
    
    # 훈련 이력 반환
    return model, history

def evaluate_model(model, X, y, dataset_name=""):
    """모델 평가"""
    # 예측 수행
    y_pred_proba = model.predict(X)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # 다양한 평가 지표 계산
    metrics = {
        'accuracy': float(accuracy_score(y, y_pred)),
        'precision': float(precision_score(y, y_pred)),
        'recall': float(recall_score(y, y_pred)),
        'f1': float(f1_score(y, y_pred)),
        'roc_auc': float(roc_auc_score(y, y_pred_proba))
    }
    
    # 결과 로깅
    logger.info(f"{dataset_name} metrics:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    return metrics

def save_model_results(model, history, metrics, model_dir, model_name):
    """모델, 훈련 이력, 평가 지표 저장"""
    # 디렉토리 생성
    os.makedirs(model_dir, exist_ok=True)
    
    # 모델 저장 (TensorFlow SavedModel 형식)
    model_path = os.path.join(model_dir, model_name)
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # 훈련 이력 저장
    history_dict = history.history
    history_json = {key: [float(val) for val in values] for key, values in history_dict.items()}
    history_path = os.path.join(model_dir, f"{model_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history_json, f, indent=4)
    logger.info(f"Training history saved to {history_path}")
    
    # 평가 지표 저장
    metrics_path = os.path.join(model_dir, f"{model_name}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Evaluation metrics saved to {metrics_path}")
    
    # 모델 다이어그램 저장 (선택 사항)
    try:
        diagram_path = os.path.join(model_dir, f"{model_name}_diagram.png")
        plot_model(model, to_file=diagram_path, show_shapes=True, show_layer_names=True)
        logger.info(f"Model diagram saved to {diagram_path}")
    except Exception as e:
        logger.warning(f"Unable to save model diagram: {e}")

def main():
    """메인 실행 함수"""
    # GPU 설정
    setup_gpu()
    
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
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = prepare_data(train_df, val_df, test_df)
    
    # 모델 훈련 설정
    input_dim = X_train.shape[1]
    
    # 기본 DNN 모델
    dnn_model = create_dnn_model(input_dim)
    dnn_model, dnn_history = train_model(
        dnn_model, X_train, y_train, X_val, y_val, 
        batch_size=64, epochs=50, model_dir=model_dir, model_name="dnn_model"
    )
    dnn_metrics = evaluate_model(dnn_model, X_test, y_test, "Test (DNN)")
    save_model_results(dnn_model, dnn_history, dnn_metrics, model_dir, "dnn_model")
    
    # Wide & Deep 모델
    wide_deep_model = create_wide_and_deep_model(input_dim)
    wide_deep_model, wide_deep_history = train_model(
        wide_deep_model, X_train, y_train, X_val, y_val, 
        batch_size=64, epochs=50, model_dir=model_dir, model_name="wide_deep_model"
    )
    wide_deep_metrics = evaluate_model(wide_deep_model, X_test, y_test, "Test (Wide & Deep)")
    save_model_results(wide_deep_model, wide_deep_history, wide_deep_metrics, model_dir, "wide_deep_model")
    
    # 최고 성능 모델 선택 (ROC AUC 기준)
    models_metrics = {
        "dnn_model": dnn_metrics,
        "wide_deep_model": wide_deep_metrics
    }
    
    best_model_name = max(models_metrics, key=lambda k: models_metrics[k]['roc_auc'])
    logger.info(f"Best deep learning model: {best_model_name} with ROC AUC = {models_metrics[best_model_name]['roc_auc']:.4f}")
    
    logger.info('✓ Deep learning model training completed successfully')

if __name__ == '__main__':
    main() 