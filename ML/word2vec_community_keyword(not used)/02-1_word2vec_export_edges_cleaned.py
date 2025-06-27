from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import pandas as pd
import traceback

print("▶ keyword_edge 사이클 및 중복 제거 버전 실행 시작")

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

# keyword_edge 불러오기
try:
    df = pd.read_sql("SELECT * FROM keyword_edge", engine)
    print(f"📥 keyword_edge 로딩 완료: {len(df)}건")
except Exception as e:
    print("❌ 테이블 로딩 실패:", e)
    traceback.print_exc()

# 자기 자신에게 연결 제거
df = df[df["source"] != df["target"]]
print(f"🚫 자기참조 제거 후: {len(df)}건")

# 역방향 쌍 중복 제거
df["undirected"] = df.apply(lambda row: tuple(sorted([row["source"], row["target"]])), axis=1)
df = df.drop_duplicates(subset="undirected")
df = df.drop(columns=["undirected"])
print(f"🔁 역방향 중복 제거 후: {len(df)}건")

# 저장
try:
    df.to_sql("keyword_edge_cleaned", engine, if_exists="replace", index=False)
    print("✅ keyword_edge_cleaned 테이블 저장 완료")
except Exception as e:
    print("❌ 저장 실패:", e)
    traceback.print_exc()