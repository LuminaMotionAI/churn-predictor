"""
개인화된 추천 시스템 모델
=====================
이 스크립트는 이탈 위험이 있는 사용자를 위한 개인화된 콘텐츠 추천 모델을 구현합니다.
협업 필터링과 콘텐츠 기반 필터링을 결합한 하이브리드 접근 방식을 사용합니다.
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import joblib
import matplotlib.pyplot as plt

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_ratings_data(filepath):
    """
    사용자-콘텐츠 평점 데이터 로드
    참고: 실제 데이터가 없으므로 합성 데이터 생성
    """
    logger.info("Loading or generating user-content rating data")
    
    if os.path.exists(filepath):
        ratings_df = pd.read_csv(filepath)
        logger.info(f"Loaded ratings data from {filepath} with shape {ratings_df.shape}")
        return ratings_df
    
    # 합성 데이터 생성
    logger.info("Generating synthetic ratings data")
    
    # 사용자, 콘텐츠, 시청 시간 데이터 생성
    n_users = 1000
    n_contents = 500
    n_ratings = 50000  # 희소 행렬 (모든 사용자가 모든 콘텐츠를 평가하지 않음)
    
    # 랜덤 사용자 ID와 콘텐츠 ID 생성
    user_ids = np.random.randint(1, n_users + 1, n_ratings)
    content_ids = np.random.randint(1, n_contents + 1, n_ratings)
    
    # 평점 (시청 시간 기준, 1-5점) 생성
    # 대부분의 콘텐츠는 3점 이상의 높은 평점을 받도록 편향
    ratings = np.clip(np.random.normal(3.5, 1.0, n_ratings), 1, 5).round(1)
    
    # 콘텐츠 장르 생성
    genres = ['Action', 'Comedy', 'Drama', 'Documentary', 'SciFi', 
              'Thriller', 'Romance', 'Animation', 'Crime', 'Fantasy']
    
    # 콘텐츠 메타데이터 생성
    content_metadata = pd.DataFrame({
        'content_id': range(1, n_contents + 1),
        'title': [f'Content {i}' for i in range(1, n_contents + 1)],
        'genre': np.random.choice(genres, n_contents),
        'release_year': np.random.randint(2010, 2023, n_contents),
        'popularity': np.random.uniform(1, 10, n_contents).round(1)
    })
    
    # 사용자-콘텐츠 평점 데이터프레임 생성
    ratings_df = pd.DataFrame({
        'user_id': user_ids,
        'content_id': content_ids,
        'rating': ratings,
        'timestamp': np.random.randint(1577836800, 1672531200, n_ratings)  # 2020-2023
    })
    
    # 중복 제거 (같은 사용자가 같은 콘텐츠를 여러 번 평가하지 않도록)
    ratings_df = ratings_df.drop_duplicates(subset=['user_id', 'content_id'])
    
    # 콘텐츠 메타데이터 조인
    ratings_df = pd.merge(ratings_df, content_metadata, on='content_id')
    
    # CSV 저장
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    ratings_df.to_csv(filepath, index=False)
    logger.info(f"Generated and saved ratings data to {filepath} with shape {ratings_df.shape}")
    
    return ratings_df

def load_user_data(filepath):
    """
    사용자 프로필 데이터 로드
    참고: 실제 데이터가 없으므로 합성 데이터 생성
    """
    logger.info("Loading or generating user profile data")
    
    if os.path.exists(filepath):
        users_df = pd.read_csv(filepath)
        logger.info(f"Loaded user data from {filepath} with shape {users_df.shape}")
        return users_df
    
    # 합성 데이터 생성
    logger.info("Generating synthetic user profile data")
    
    n_users = 1000
    
    # 사용자 데이터 생성
    users_df = pd.DataFrame({
        'user_id': range(1, n_users + 1),
        'age': np.random.randint(18, 70, n_users),
        'gender': np.random.choice(['M', 'F', 'Other'], n_users, p=[0.48, 0.48, 0.04]),
        'subscription_type': np.random.choice(['Basic', 'Standard', 'Premium'], n_users),
        'tenure_months': np.random.randint(1, 60, n_users),
        'weekly_viewing_hours': np.clip(np.random.normal(10, 5, n_users), 0, 50).round(1),
        'churn_probability': np.random.uniform(0, 1, n_users).round(3)  # 모델에서 예측된 이탈 확률
    })
    
    # CSV 저장
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    users_df.to_csv(filepath, index=False)
    logger.info(f"Generated and saved user profile data to {filepath} with shape {users_df.shape}")
    
    return users_df

def create_user_item_matrix(ratings_df):
    """사용자-아이템 평점 행렬 생성"""
    logger.info("Creating user-item rating matrix")
    
    # 피벗 테이블 생성 (사용자-아이템 행렬)
    user_item_matrix = ratings_df.pivot_table(
        index='user_id',
        columns='content_id',
        values='rating'
    ).fillna(0)
    
    logger.info(f"Created user-item matrix with shape {user_item_matrix.shape}")
    
    return user_item_matrix

def train_collaborative_filtering_model(user_item_matrix, n_factors=50):
    """
    협업 필터링 모델 훈련
    행렬 분해 (Matrix Factorization)를 사용하여 잠재 요인 추출
    """
    logger.info(f"Training collaborative filtering model with {n_factors} latent factors")
    
    # 행렬의 평균 대체
    user_ratings_mean = np.mean(user_item_matrix.values, axis=1)
    user_ratings_centered = user_item_matrix.values - user_ratings_mean.reshape(-1, 1)
    
    # 특이값 분해 (SVD)
    U, sigma, Vt = svds(user_ratings_centered, k=n_factors)
    
    # 특이값을 대각 행렬로 변환
    sigma_diag = np.diag(sigma)
    
    # 사용자 및 아이템 잠재 요인
    user_factors = np.dot(U, np.sqrt(sigma_diag))
    item_factors = np.dot(np.sqrt(sigma_diag), Vt)
    
    # 예측 행렬 계산
    predicted_ratings = np.dot(user_factors, item_factors) + user_ratings_mean.reshape(-1, 1)
    
    # 데이터프레임으로 변환
    predicted_df = pd.DataFrame(
        predicted_ratings,
        index=user_item_matrix.index,
        columns=user_item_matrix.columns
    )
    
    # 모델 정보
    model_info = {
        'user_factors': user_factors,
        'item_factors': item_factors,
        'user_ratings_mean': user_ratings_mean,
        'user_ids': user_item_matrix.index.tolist(),
        'content_ids': user_item_matrix.columns.tolist()
    }
    
    logger.info("Collaborative filtering model trained successfully")
    
    return predicted_df, model_info

def build_content_features(ratings_df):
    """콘텐츠 특성 벡터 구축"""
    logger.info("Building content feature vectors")
    
    # 콘텐츠 메타데이터 추출
    content_df = ratings_df[['content_id', 'title', 'genre', 'release_year', 'popularity']].drop_duplicates()
    
    # 장르 원-핫 인코딩
    genre_dummies = pd.get_dummies(content_df['genre'], prefix='genre')
    
    # 출시 연도 정규화
    year_min = content_df['release_year'].min()
    year_max = content_df['release_year'].max()
    content_df['year_norm'] = (content_df['release_year'] - year_min) / (year_max - year_min)
    
    # 인기도 정규화
    pop_min = content_df['popularity'].min()
    pop_max = content_df['popularity'].max()
    content_df['popularity_norm'] = (content_df['popularity'] - pop_min) / (pop_max - pop_min)
    
    # 특성 데이터프레임
    content_features = pd.concat([
        content_df[['content_id', 'year_norm', 'popularity_norm']],
        genre_dummies
    ], axis=1)
    
    # 콘텐츠 ID를 인덱스로 설정
    content_features.set_index('content_id', inplace=True)
    
    logger.info(f"Built content features with shape {content_features.shape}")
    
    return content_features

def calculate_content_similarity(content_features):
    """콘텐츠 간 유사도 계산"""
    logger.info("Calculating content similarity matrix")
    
    # 코사인 유사도 계산
    content_similarity = cosine_similarity(content_features)
    
    # 데이터프레임으로 변환
    similarity_df = pd.DataFrame(
        content_similarity,
        index=content_features.index,
        columns=content_features.index
    )
    
    logger.info(f"Created content similarity matrix with shape {similarity_df.shape}")
    
    return similarity_df

def identify_at_risk_users(users_df, churn_threshold=0.5):
    """이탈 위험이 있는 사용자 식별"""
    logger.info(f"Identifying users at risk of churn (threshold: {churn_threshold})")
    
    # 이탈 확률이 임계값보다 높은 사용자 필터링
    at_risk_users = users_df[users_df['churn_probability'] >= churn_threshold]
    
    logger.info(f"Identified {len(at_risk_users)} users at risk of churn")
    
    return at_risk_users

def generate_recommendations(
    user_id,
    predicted_ratings,
    content_similarity,
    ratings_df,
    n_recommendations=10,
    alpha=0.7  # 협업 필터링과 콘텐츠 기반 필터링의 가중치 조정
):
    """사용자별 개인화된 추천 생성"""
    # 사용자가 이미 시청한 콘텐츠 확인
    user_watched = set(ratings_df[ratings_df['user_id'] == user_id]['content_id'].tolist())
    
    # 협업 필터링 예측 가져오기
    if user_id in predicted_ratings.index:
        cf_predictions = predicted_ratings.loc[user_id]
    else:
        # 사용자가 행렬에 없으면 (콜드 스타트) 콘텐츠 기반만 사용
        alpha = 0
        cf_predictions = pd.Series(0, index=content_similarity.index)
    
    # 콘텐츠 기반 예측 초기화
    cb_predictions = pd.Series(0.0, index=content_similarity.index)
    
    # 사용자가 시청한 콘텐츠 기반으로 콘텐츠 기반 점수 계산
    for content_id in user_watched:
        if content_id in content_similarity.index:
            # 이미 시청한 콘텐츠와의 유사도 정도에 따라 점수 부여
            watched_rating = ratings_df[(ratings_df['user_id'] == user_id) & 
                                        (ratings_df['content_id'] == content_id)]['rating'].iloc[0]
            
            # 평점으로 가중치를 부여한 유사도 계산
            weighted_similarity = content_similarity[content_id] * (watched_rating / 5.0)
            cb_predictions += weighted_similarity
    
    # 평균으로 정규화
    if len(user_watched) > 0:
        cb_predictions = cb_predictions / len(user_watched)
    
    # 하이브리드 예측: 협업 필터링과 콘텐츠 기반 필터링 조합
    hybrid_predictions = alpha * cf_predictions + (1 - alpha) * cb_predictions
    
    # 이미 시청한 콘텐츠 제외
    hybrid_predictions = hybrid_predictions.drop(list(user_watched), errors='ignore')
    
    # 상위 N개 추천
    top_recommendations = hybrid_predictions.sort_values(ascending=False).head(n_recommendations)
    
    # 추천 콘텐츠의 메타데이터 조회
    content_metadata = ratings_df.drop_duplicates(subset=['content_id'])[['content_id', 'title', 'genre', 'release_year', 'popularity']]
    content_metadata = content_metadata.set_index('content_id')
    
    # 추천 결과 데이터프레임 생성
    recommendations = pd.DataFrame({
        'content_id': top_recommendations.index,
        'predicted_rating': top_recommendations.values,
        'title': [content_metadata.loc[cid, 'title'] if cid in content_metadata.index else f'Content {cid}' for cid in top_recommendations.index],
        'genre': [content_metadata.loc[cid, 'genre'] if cid in content_metadata.index else 'Unknown' for cid in top_recommendations.index],
        'release_year': [content_metadata.loc[cid, 'release_year'] if cid in content_metadata.index else 0 for cid in top_recommendations.index],
        'popularity': [content_metadata.loc[cid, 'popularity'] if cid in content_metadata.index else 0 for cid in top_recommendations.index]
    })
    
    return recommendations

def generate_personalized_recommendations_for_at_risk_users(
    at_risk_users,
    predicted_ratings,
    content_similarity,
    ratings_df,
    n_recommendations=10
):
    """이탈 위험이 있는 모든 사용자에 대한 개인화된 추천 생성"""
    logger.info(f"Generating personalized recommendations for {len(at_risk_users)} at-risk users")
    
    all_recommendations = {}
    
    for idx, user in at_risk_users.iterrows():
        user_id = user['user_id']
        
        # 사용자별 추천 생성
        user_recommendations = generate_recommendations(
            user_id,
            predicted_ratings,
            content_similarity,
            ratings_df,
            n_recommendations
        )
        
        all_recommendations[user_id] = user_recommendations
    
    logger.info(f"Generated recommendations for {len(all_recommendations)} users")
    
    return all_recommendations

def evaluate_recommendations(recommendations, ratings_df, at_risk_users):
    """추천 시스템 평가"""
    logger.info("Evaluating recommendation quality")
    
    # 평가 지표 초기화
    metrics = {
        'coverage': 0,  # 추천된 콘텐츠의 다양성
        'average_popularity': 0,  # 추천된 콘텐츠의 평균 인기도
        'genre_diversity': 0,  # 장르 다양성
    }
    
    # 모든 추천 콘텐츠 수집
    all_recommended_content = set()
    all_recommended_genres = set()
    total_popularity = 0
    total_items = 0
    
    for user_id, user_recs in recommendations.items():
        # 추천된 콘텐츠 추가
        recommended_content_ids = user_recs['content_id'].tolist()
        all_recommended_content.update(recommended_content_ids)
        
        # 장르 다양성
        recommended_genres = user_recs['genre'].tolist()
        all_recommended_genres.update(recommended_genres)
        
        # 인기도 합산
        total_popularity += user_recs['popularity'].sum()
        total_items += len(user_recs)
    
    # 전체 콘텐츠 대비 추천된 비율 (커버리지)
    all_content = set(ratings_df['content_id'].unique())
    metrics['coverage'] = len(all_recommended_content) / len(all_content)
    
    # 평균 인기도
    metrics['average_popularity'] = total_popularity / total_items if total_items > 0 else 0
    
    # 장르 다양성
    all_genres = set(ratings_df['genre'].unique())
    metrics['genre_diversity'] = len(all_recommended_genres) / len(all_genres)
    
    logger.info(f"Recommendation metrics: {metrics}")
    
    return metrics

def save_model_results(model_info, content_similarity, metrics, model_dir):
    """모델 정보 및 결과 저장"""
    logger.info("Saving recommendation model results")
    
    # 디렉토리 생성
    os.makedirs(model_dir, exist_ok=True)
    
    # 모델 정보 저장 (잠재 요인 등)
    model_path = os.path.join(model_dir, "recommendation_model_info.joblib")
    joblib.dump(model_info, model_path)
    logger.info(f"Model info saved to {model_path}")
    
    # 콘텐츠 유사도 행렬 저장
    similarity_path = os.path.join(model_dir, "content_similarity.joblib")
    joblib.dump(content_similarity, similarity_path)
    logger.info(f"Content similarity matrix saved to {similarity_path}")
    
    # 평가 지표 저장
    metrics_path = os.path.join(model_dir, "recommendation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Evaluation metrics saved to {metrics_path}")

def visualize_recommendations(recommendations, at_risk_users, output_dir):
    """추천 결과 시각화"""
    logger.info("Visualizing recommendation results")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 샘플 사용자 추천 시각화 (최대 5명)
    sample_users = list(recommendations.keys())[:5]
    
    for user_id in sample_users:
        user_recs = recommendations[user_id]
        user_info = at_risk_users[at_risk_users['user_id'] == user_id].iloc[0]
        
        plt.figure(figsize=(10, 6))
        
        # 추천 점수 시각화
        plt.barh(user_recs['title'], user_recs['predicted_rating'], color='skyblue')
        plt.xlabel('Predicted Rating')
        plt.ylabel('Content Title')
        plt.title(f'Top Recommendations for User {user_id}\n'
                  f'Churn Probability: {user_info["churn_probability"]:.2f}, '
                  f'Subscription: {user_info["subscription_type"]}')
        plt.tight_layout()
        
        # 저장
        output_path = os.path.join(output_dir, f'user_{user_id}_recommendations.png')
        plt.savefig(output_path)
        plt.close()
    
    # 장르 분포 시각화
    plt.figure(figsize=(12, 8))
    
    all_genres = []
    for user_recs in recommendations.values():
        all_genres.extend(user_recs['genre'].tolist())
    
    genre_counts = pd.Series(all_genres).value_counts()
    
    plt.bar(genre_counts.index, genre_counts.values, color='lightgreen')
    plt.xlabel('Genre')
    plt.ylabel('Frequency in Recommendations')
    plt.title('Genre Distribution in Recommendations for At-Risk Users')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # 저장
    output_path = os.path.join(output_dir, 'genre_distribution.png')
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Visualizations saved to {output_dir}")

def main():
    """메인 실행 함수"""
    # 프로젝트 경로 설정
    project_dir = Path(__file__).resolve().parents[2]
    data_dir = os.path.join(project_dir, 'data')
    model_dir = os.path.join(project_dir, 'models')
    reports_dir = os.path.join(project_dir, 'reports')
    
    # 디렉토리 생성
    os.makedirs(os.path.join(data_dir, 'external'), exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # 데이터 파일 경로
    ratings_path = os.path.join(data_dir, 'external', 'ratings.csv')
    users_path = os.path.join(data_dir, 'external', 'users.csv')
    
    # 데이터 로드 또는 생성
    ratings_df = load_ratings_data(ratings_path)
    users_df = load_user_data(users_path)
    
    # 사용자-아이템 행렬 생성
    user_item_matrix = create_user_item_matrix(ratings_df)
    
    # 협업 필터링 모델 훈련
    predicted_ratings, model_info = train_collaborative_filtering_model(user_item_matrix, n_factors=30)
    
    # 콘텐츠 특성 및 유사도 계산
    content_features = build_content_features(ratings_df)
    content_similarity = calculate_content_similarity(content_features)
    
    # 이탈 위험 사용자 식별
    at_risk_users = identify_at_risk_users(users_df, churn_threshold=0.6)
    
    # 이탈 위험 사용자를 위한 개인화된 추천 생성
    recommendations = generate_personalized_recommendations_for_at_risk_users(
        at_risk_users,
        predicted_ratings,
        content_similarity,
        ratings_df,
        n_recommendations=10
    )
    
    # 추천 시스템 평가
    metrics = evaluate_recommendations(recommendations, ratings_df, at_risk_users)
    
    # 결과 저장
    save_model_results(model_info, content_similarity, metrics, model_dir)
    
    # 결과 시각화
    visualize_recommendations(recommendations, at_risk_users, reports_dir)
    
    logger.info('✓ Recommendation model training completed successfully')

if __name__ == '__main__':
    main() 