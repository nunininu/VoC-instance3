from sqlalchemy import create_engine
from dotenv import load_dotenv
import pandas as pd
import os
import traceback

print("▶ keyword_edge_grouped 테이블 생성 시작")

# 환경 변수 로드
try:
    load_dotenv()
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME")
    db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(db_url)
    print("✅ DB 연결 성공")
except Exception as e:
    print("❌ DB 연결 실패:", e)
    traceback.print_exc()

# cleaned 테이블 불러오기
try:
    df = pd.read_sql("SELECT * FROM keyword_edge_cleaned", engine)
    print(f"📥 keyword_edge_cleaned 로딩 완료: {len(df)}건")
except Exception as e:
    print("❌ 테이블 로딩 실패:", e)
    traceback.print_exc()

# source-target 기준 집계
try:
    df_grouped = df.groupby(["source", "target"], as_index=False).agg({"similarity": "sum"})
    df_grouped = df_grouped.rename(columns={"similarity": "weight"})
    print("✅ source-target 기준 groupby 완료")
    print("📌 샘플:", df_grouped.head(3))
except Exception as e:
    print("❌ groupby 실패:", e)
    traceback.print_exc()

# 저장
try:
    df_grouped.to_sql("keyword_edge_grouped", engine, if_exists="replace", index=False)
    print("✅ keyword_edge_grouped 테이블 저장 완료")
except Exception as e:
    print("❌ 저장 실패:", e)
    traceback.print_exc()