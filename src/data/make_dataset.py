"""
데이터 다운로드 및 전처리 스크립트
===================================
이 스크립트는 Kaggle API를 사용하여 넷플릭스 이탈률 예측을 위한 데이터셋을 다운로드하고 전처리합니다.
"""
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from kaggle.api.kaggle_api_extended import KaggleApi

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_data(dataset_name, raw_data_dir):
    """Kaggle API를 사용하여 데이터셋 다운로드"""
    logger.info(f'Downloading dataset {dataset_name}')
    try:
        # Kaggle API 인증 (kaggle.json 파일이 필요)
        api = KaggleApi()
        api.authenticate()
        
        # 데이터셋 다운로드
        api.dataset_download_files(dataset_name, path=raw_data_dir, unzip=True)
        logger.info(f'Successfully downloaded dataset to {raw_data_dir}')
    except Exception as e:
        logger.error(f'Error downloading dataset: {e}')
        logger.info('Proceeding with synthetic data generation as fallback')
        generate_synthetic_data(raw_data_dir)

def generate_synthetic_data(output_dir):
    """실제 데이터 다운로드에 실패한 경우 합성 데이터 생성"""
    logger.info('Generating synthetic Netflix churn dataset')
    
    # 샘플 크기 설정
    n_samples = 10000
    
    # 가입 기간(개월)
    tenure = np.random.randint(1, 60, n_samples)
    
    # 월 구독 요금 (기본, 표준, 프리미엄)
    subscription_type = np.random.choice(['Basic', 'Standard', 'Premium'], n_samples)
    monthly_fee = np.where(subscription_type == 'Basic', 9.99, 
                    np.where(subscription_type == 'Standard', 13.99, 17.99))
    
    # 주간 시청 시간(시간)
    weekly_viewing_hours = np.clip(np.random.normal(10, 5, n_samples), 0, 50)
    
    # 장치 유형 (모바일, TV, 컴퓨터, 태블릿)
    device_type = np.random.choice(['Mobile', 'TV', 'Computer', 'Tablet'], n_samples)
    
    # 콘텐츠 다양성 (시청한 장르 수)
    content_diversity = np.random.randint(1, 10, n_samples)
    
    # 사용자 등급 (1-5)
    user_rating = np.random.uniform(1, 5, n_samples)
    
    # 고객 서비스 접촉 횟수
    customer_service_calls = np.random.poisson(1, n_samples)
    
    # 기술적 문제 횟수
    technical_issues = np.random.poisson(0.5, n_samples)
    
    # 추천 콘텐츠 시청 비율
    recommended_content_watched = np.random.uniform(0, 1, n_samples)
    
    # 가격 인상 경험 여부
    price_increase = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    
    # 계정 공유 여부
    account_sharing = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    
    # 지역
    region = np.random.choice(['North America', 'Europe', 'Asia', 'Latin America', 'Oceania'], n_samples)
    
    # 나이
    age = np.random.randint(18, 70, n_samples)
    
    # 성별
    gender = np.random.choice(['M', 'F', 'Other'], n_samples, p=[0.48, 0.48, 0.04])
    
    # 경쟁 서비스 구독 수
    competing_services = np.random.randint(0, 5, n_samples)
    
    # 이탈 여부 (타겟 변수) - 다양한 특성들을 기반으로 계산
    # 시청 시간이 적거나, 이용 기간이 짧거나, 가격 인상이 있었거나, 기술적 문제가 많거나, 경쟁 서비스가 많을수록 이탈 확률 높아짐
    churn_prob = (
        0.5 - 0.02 * weekly_viewing_hours / 10 
        - 0.02 * tenure / 12 
        + 0.15 * price_increase 
        + 0.1 * (technical_issues > 2) 
        + 0.05 * competing_services / 2
        - 0.1 * user_rating / 5
        - 0.05 * recommended_content_watched
        + 0.02 * (monthly_fee > 15)
    )
    # 확률을 0-1 사이로 조정
    churn_prob = np.clip(churn_prob, 0.05, 0.95)
    churn = np.random.binomial(1, churn_prob)
    
    # 데이터프레임 생성
    df = pd.DataFrame({
        'tenure_months': tenure,
        'subscription_type': subscription_type,
        'monthly_fee': monthly_fee,
        'weekly_viewing_hours': weekly_viewing_hours,
        'device_type': device_type,
        'content_diversity': content_diversity,
        'user_rating': user_rating,
        'customer_service_calls': customer_service_calls,
        'technical_issues': technical_issues,
        'recommended_content_watched': recommended_content_watched,
        'price_increase': price_increase,
        'account_sharing': account_sharing,
        'region': region,
        'age': age,
        'gender': gender,
        'competing_services': competing_services,
        'churn': churn
    })
    
    # CSV 파일로 저장
    output_path = os.path.join(output_dir, 'netflix_churn_data.csv')
    df.to_csv(output_path, index=False)
    logger.info(f'Successfully generated synthetic data: {output_path}')
    
    return output_path

def preprocess_data(input_filepath, output_filepath):
    """데이터 전처리 함수"""
    logger.info(f'Preprocessing data from {input_filepath}')
    
    # 데이터 읽기
    df = pd.read_csv(input_filepath)
    
    # 누락된 값 처리
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())
    
    # 범주형 변수 원-핫 인코딩
    categorical_cols = ['subscription_type', 'device_type', 'region', 'gender']
    for col in categorical_cols:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], prefix=[col], drop_first=True)
    
    # 특성 스케일링
    from sklearn.preprocessing import StandardScaler
    numeric_cols = [
        'tenure_months', 'monthly_fee', 'weekly_viewing_hours', 
        'content_diversity', 'user_rating', 'customer_service_calls',
        'technical_issues', 'recommended_content_watched', 'age',
        'competing_services'
    ]
    
    # 존재하는 숫자형 컬럼만 스케일링
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    if numeric_cols:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # 훈련, 검증, 테스트 세트 분할
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['churn'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['churn'])
    
    # 저장
    processed_dir = os.path.dirname(output_filepath)
    train_path = os.path.join(processed_dir, 'train.csv')
    val_path = os.path.join(processed_dir, 'val.csv')
    test_path = os.path.join(processed_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f'Preprocessed data saved: train={train_path}, val={val_path}, test={test_path}')
    
    return train_path, val_path, test_path

def main():
    """메인 함수"""
    # 프로젝트 디렉토리 설정
    project_dir = Path(__file__).resolve().parents[2]
    raw_data_dir = os.path.join(project_dir, 'data', 'raw')
    processed_data_dir = os.path.join(project_dir, 'data', 'processed')
    
    # 디렉토리가 없으면 생성
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # Kaggle 데이터셋 이름 (예시)
    dataset_name = "syedsaadahmed/netflix-subscription-and-streaming-data"  # 실제 Kaggle 데이터셋 이름으로 교체해야 함
    
    # 데이터 다운로드 (또는 Kaggle API가 실패하면 합성 데이터 생성)
    try:
        download_data(dataset_name, raw_data_dir)
        input_filepath = os.path.join(raw_data_dir, 'netflix_churn_data.csv')  # 실제 파일 이름으로 교체해야 함
    except:
        logger.warning("Falling back to synthetic data generation")
        input_filepath = generate_synthetic_data(raw_data_dir)
    
    # 데이터 전처리
    output_filepath = os.path.join(processed_data_dir, 'netflix_churn_data_processed.csv')
    preprocess_data(input_filepath, output_filepath)
    
    logger.info('✓ Data processing completed successfully')

if __name__ == '__main__':
    main() 