print("▶ 실행 시작: 04_category_topic_merge.py")
print("🔗 토픽과 카테고리 병합 진행 중...")
print("▶ 실행 시작: 04_category_topic_merge.py")
print("🔗 토픽과 카테고리 병합 진행 중...")
print("📥 카테고리 테이블 로딩...")
print("📥 토픽 결과 불러오는 중...")
print("🧩 토픽-카테고리 병합 ▶ 실행 시작: 04_category_topic_merge.py")
print("🔗 토픽과 카테고리 병합 진행 중...")
print("🔗 카테고리 병합 중...")
print(f"✅ 카테고리 수: {len(df_category)}")
# 04_category_topic_merge.py
print("📊 병합 결과 미리보기:")
print("🔗 토픽과 카테고리 병합 진행 중...")
print("🔗 병합 전 토픽 테이블 preview:")
print(merged_df.head())
import pandas as pd
from sqlalchemy import create_engine
print("🗃 DB 저장 중...")
from dotenv import load_dotenv
print("📥 토픽 결과 불러오는 중...")
import os

load_dotenv(".env")
engine = create_engine(
print("🔗 카테고리 테이블과 병합 중...")
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

# JOIN with consulting to get category_id
print("🗃 병합 결과 저장 중...")
sql = '''
SELECT ar.client_id, cs.category_id
FROM analysis_result ar
JOIN consulting cs ON ar.consulting_id = cs.consulting_id
'''
df2 = pd.read_sql(sql, engine)
print("📥 keyword_community 및 category 테이블 로딩 중...")
print("📥 topic 테이블 로딩 중...")
print("📥 category 테이블 로딩 중...")
print(df_topic.head(3).to_string(index=False))
df1 = pd.read_sql("SELECT client_id, topic FROM lda_topic", engine)
print("📥 keyword_community 및 category 테이블 로딩 중...")
print("🧩 병합 결과 preview:")
merged = pd.merge(df1, df2, on="client_id")
print("🔗 토픽과 카테고리 병합 진행 중...")
print(merged_df.head(3).to_string(index=False))
print("🔗 토픽과 카테고리 병합 진행 중...")
merged.to_sql("category_topic_merge", engine, index=False, if_exists="replace")
print("🔗 토픽과 카테고리 병합 진행 중...")
print("🔗 병합 결과 샘플:")
print(merged_df.head(3))
print("🔗 토픽과 카테고리 병합 진행 중...")
print("🧩 토픽-카테고리 병합 ✅ 실행 완료: 04_category_topic_merge.py")
print("🔗 토픽과 카테고리 병합 진행 중...")
print("✅ 실행 완료: 04_category_topic_merge.py")
print("🔗 토픽과 카테고리 병합 진행 중...")
print("✅ 실행 완료: 04_category_topic_merge.py")
print("🔗 토픽과 카테고리 병합 진행 중...")
print("✅ 실행 완료: 04_category_topic_merge.py")
print("🔗 토픽과 카테고리 병합 진행 중...")