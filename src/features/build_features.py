"""
특성 엔지니어링 스크립트
========================
이 스크립트는 이탈률 예측을 위해 고급 특성들을 생성하는 기능을 제공합니다.
"""
import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def add_interaction_features(df, interaction_cols=None):
    """숫자형 변수 간 상호작용 특성 추가"""
    if interaction_cols is None:
        # 기본 상호작용 컬럼
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        # 타겟 변수 제외
        if 'churn' in numeric_cols:
            numeric_cols.remove('churn')
        interaction_cols = numeric_cols
    
    logger.info(f"Creating interaction features from columns: {interaction_cols}")
    
    # 차원 유지를 위해 degree=2, interaction_only=True 사용
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    interaction_features = poly.fit_transform(df[interaction_cols])
    
    # 새 특성 이름 생성
    feature_names = poly.get_feature_names_out(interaction_cols)
    
    # 원본 특성 인덱스 제외 (첫 len(interaction_cols)개 컬럼)
    new_feature_names = feature_names[len(interaction_cols):]
    new_features = interaction_features[:, len(interaction_cols):]
    
    # 새 특성 이름 배열을 DataFrame의 컬럼 이름으로 사용
    interaction_df = pd.DataFrame(new_features, columns=new_feature_names, index=df.index)
    
    # 원본 DataFrame과 합치기
    result_df = pd.concat([df, interaction_df], axis=1)
    logger.info(f"Added {len(new_feature_names)} interaction features")
    
    return result_df

def add_time_based_features(df):
    """시간 기반 특성 추가 (이탈률 예측에 중요한 시간적 패턴 포착)"""
    logger.info("Adding time-based features")
    
    # 이 예제에서는 'tenure_months'가 있다고 가정
    if 'tenure_months' in df.columns:
        # 고객 생애 주기 특성
        df['is_new_user'] = (df['tenure_months'] <= 3).astype(int)
        df['is_established_user'] = ((df['tenure_months'] > 3) & (df['tenure_months'] <= 12)).astype(int)
        df['is_loyal_user'] = (df['tenure_months'] > 12).astype(int)
        
        # 1년, 2년 경계 특성
        df['approaching_1yr'] = ((df['tenure_months'] >= 10) & (df['tenure_months'] <= 14)).astype(int)
        df['approaching_2yr'] = ((df['tenure_months'] >= 22) & (df['tenure_months'] <= 26)).astype(int)
        
        # 비선형 변환
        df['tenure_sqrt'] = np.sqrt(df['tenure_months'])
        df['tenure_squared'] = df['tenure_months'] ** 2
        
        logger.info("Added 7 time-based features")
    else:
        logger.warning("No 'tenure_months' column found. Skipping time-based features.")
    
    return df

def add_engagement_features(df):
    """참여도 관련 특성 추가"""
    logger.info("Adding engagement features")
    
    # 시청 시간과 콘텐츠 다양성 관련 특성
    if 'weekly_viewing_hours' in df.columns and 'content_diversity' in df.columns:
        # 시청 시간 기준 사용자 참여도 분류
        df['high_engagement'] = (df['weekly_viewing_hours'] > 15).astype(int)
        df['low_engagement'] = (df['weekly_viewing_hours'] < 5).astype(int)
        
        # 시청 시간과 콘텐츠 다양성 비율
        df['hours_per_genre'] = df['weekly_viewing_hours'] / (df['content_diversity'] + 0.1)  # 0으로 나누는 것 방지
        
        # 비선형 변환
        df['viewing_sqrt'] = np.sqrt(df['weekly_viewing_hours'])
        df['viewing_log'] = np.log1p(df['weekly_viewing_hours'])  # log(1+x)
        
        logger.info("Added 5 engagement features")
    else:
        logger.warning("Missing columns for engagement features. Skipping.")
    
    # 추천 콘텐츠 관련 특성
    if 'recommended_content_watched' in df.columns:
        # 추천 반응성 범주
        df['high_recommendation_follower'] = (df['recommended_content_watched'] > 0.7).astype(int)
        df['low_recommendation_follower'] = (df['recommended_content_watched'] < 0.3).astype(int)
        
        logger.info("Added 2 recommendation-related features")
    
    return df

def add_customer_experience_features(df):
    """고객 경험 관련 특성 추가"""
    logger.info("Adding customer experience features")
    
    # 서비스 문제와 관련된 특성
    if 'customer_service_calls' in df.columns and 'technical_issues' in df.columns:
        # 총 문제 횟수
        df['total_issues'] = df['customer_service_calls'] + df['technical_issues']
        
        # 문제 심각도 (기술적 이슈가 고객 서비스 호출로 이어진 비율)
        df['issue_severity'] = df['customer_service_calls'] / (df['technical_issues'] + 1)
        
        # 빈번한 문제 여부
        df['frequent_problems'] = (df['total_issues'] > 3).astype(int)
        
        logger.info("Added 3 customer experience features")
    else:
        logger.warning("Missing columns for customer experience features. Skipping.")
    
    # 가격 관련 특성
    if 'price_increase' in df.columns and 'monthly_fee' in df.columns:
        # 가격 인상에 대한 민감도 (가격 vs 서비스 사용)
        if 'weekly_viewing_hours' in df.columns:
            df['price_sensitivity'] = df['monthly_fee'] / (df['weekly_viewing_hours'] + 0.1)
            df['price_increase_sensitivity'] = df['price_increase'] * df['price_sensitivity']
            
            logger.info("Added 2 price-related features")
    
    return df

def add_competitive_features(df):
    """경쟁 환경 관련 특성 추가"""
    logger.info("Adding competitive features")
    
    if 'competing_services' in df.columns:
        # 경쟁사 서비스 범주
        df['no_competition'] = (df['competing_services'] == 0).astype(int)
        df['high_competition'] = (df['competing_services'] >= 3).astype(int)
        
        # 비선형 변환
        df['competition_squared'] = df['competing_services'] ** 2
        
        logger.info("Added 3 competitive features")
    else:
        logger.warning("No 'competing_services' column found. Skipping competitive features.")
    
    return df

def perform_feature_selection(df, target_col='churn', k=20):
    """중요 특성 선택"""
    logger.info("Performing feature selection")
    
    # 타겟 분리
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 수치형 컬럼만 선택
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    X_numeric = X[numeric_cols]
    
    # SelectKBest를 사용한 특성 선택
    selector = SelectKBest(f_classif, k=min(k, len(numeric_cols)))
    selector.fit(X_numeric, y)
    
    # 선택된 특성의 인덱스
    selected_indices = selector.get_support(indices=True)
    selected_features = [numeric_cols[i] for i in selected_indices]
    
    logger.info(f"Selected top {len(selected_features)} features: {selected_features}")
    
    # 선택된 특성과 비수치형 특성을 포함한 데이터프레임 반환
    categorical_cols = X.select_dtypes(exclude=['float64', 'int64']).columns.tolist()
    final_cols = selected_features + categorical_cols + [target_col]
    
    return df[final_cols], selected_features

def perform_dimensionality_reduction(df, target_col='churn', n_components=10):
    """차원 축소를 통한 주성분 추가"""
    logger.info("Performing PCA for dimensionality reduction")
    
    # 타겟 분리
    X = df.drop(columns=[target_col])
    
    # 수치형 컬럼만 선택
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    X_numeric = X[numeric_cols]
    
    # PCA 적용
    n_components = min(n_components, len(numeric_cols), len(X_numeric))
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X_numeric)
    
    # 주성분을 데이터프레임으로 변환
    pc_cols = [f'PC{i+1}' for i in range(n_components)]
    pc_df = pd.DataFrame(principal_components, columns=pc_cols, index=df.index)
    
    # 설명된 분산 비율 기록
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = explained_variance.cumsum()
    logger.info(f"Cumulative explained variance: {cumulative_variance[-1]:.4f}")
    logger.info(f"Added {n_components} principal components")
    
    # 원본 데이터프레임과 주성분 데이터프레임 합치기
    result_df = pd.concat([df, pc_df], axis=1)
    
    return result_df, pc_cols

def build_features(input_filepath, output_filepath):
    """특성 엔지니어링 메인 함수"""
    logger.info(f"Building features from {input_filepath}")
    
    # 데이터 로드
    df = pd.read_csv(input_filepath)
    
    # 고급 특성 추가
    df = add_time_based_features(df)
    df = add_engagement_features(df)
    df = add_customer_experience_features(df)
    df = add_competitive_features(df)
    df = add_interaction_features(df)
    
    # 차원 축소
    df, pc_cols = perform_dimensionality_reduction(df, n_components=5)
    
    # 특성 선택
    df, selected_features = perform_feature_selection(df, k=30)
    
    # 결과 저장
    df.to_csv(output_filepath, index=False)
    logger.info(f"Enhanced features saved to {output_filepath}")
    
    # 특성 중요도 정보 저장
    features_info = {
        'principal_components': pc_cols,
        'selected_features': selected_features
    }
    
    return df, features_info

def main():
    """메인 실행 함수"""
    # 프로젝트 경로 설정
    project_dir = Path(__file__).resolve().parents[2]
    processed_data_dir = os.path.join(project_dir, 'data', 'processed')
    
    # 입력 파일 경로 (train, val, test)
    for dataset in ['train', 'val', 'test']:
        input_filepath = os.path.join(processed_data_dir, f'{dataset}.csv')
        output_filepath = os.path.join(processed_data_dir, f'{dataset}_featured.csv')
        
        if os.path.exists(input_filepath):
            build_features(input_filepath, output_filepath)
        else:
            logger.warning(f"Input file not found: {input_filepath}")
    
    logger.info('✓ Feature engineering completed successfully')

if __name__ == '__main__':
    main() 