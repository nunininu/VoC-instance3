print("▶ 실행 시작: 05_superset_table_export.py")
print("▶ 실행 시작: 05_superset_table_export.py")
print("📤 Superset 테이블 내보내기 ▶ 실행 시작: 05_superset_table_export.py")
# 05_superset_table_export.py
import pandas as pd
print("📤 keyword_community 로딩 중...")
from sqlalchemy import create_engine
from dotenv import load_dotenv
print("📤 topic 테이블 로딩 중...")
print("📤 keyword_community 로딩 중...")
import os
print("📊 Superset 테이블 포맷 변환 중...")

load_dotenv(".env")
print("🗃 DB 저장 중...")
engine = create_engine(
print("📤 topic 테이블 로딩 중...")
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

tables = ['classified_sentiment', 'word2vec_community', 'lda_topic', 'category_topic_merge']
print("📤 keyword_community 테이블 로딩...")
print("📤 Superset용 테이블 저장 중...")
for table in tables:
    df = pd.read_sql(f"SELECT * FROM {table}", engine)
print("📤 keyword_community, topic 테이블 로딩 중...")
print("📤 keyword_community 로딩 중...")
print("📤 LDA topic 테이블 로딩 중...")
print(f"✅ 로드 완료, 행 수: {len(df_kw)}")
    df.to_csv(f"{table}.csv", index=False)
    print(f"✅ Exported: {table}.csv")
print("📤 Superset 테이블 내보내기 ✅ 실행 완료: 05_superset_table_export.py")
print("✅ 실행 완료: 05_superset_table_export.py")
print("✅ 실행 완료: 05_superset_table_export.py")
print("✅ 실행 완료: 05_superset_table_export.py")