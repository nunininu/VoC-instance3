from dotenv import load_dotenv
import os
import pandas as pd
from sqlalchemy import create_engine, text

# .env 로드
load_dotenv()

# DB 환경변수 로딩
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# DB 연결
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

# CSV 로딩
nodes_df = pd.read_csv("complaint_nodes_clean.csv")
edges_df = pd.read_csv("complaint_edges_clean.csv")

with engine.begin() as conn:
    # 기존 테이블 제거
    conn.execute(text("DROP TABLE IF EXISTS complaint_nodes_clean"))
    conn.execute(text("DROP TABLE IF EXISTS complaint_edges_clean"))

    # complaint_nodes_clean: category 컬럼 포함
    conn.execute(text("""
        CREATE TABLE complaint_nodes_clean (
            id TEXT PRIMARY KEY,
            label TEXT,
            category TEXT,
            size FLOAT
        )
    """))

    # complaint_edges_clean
    conn.execute(text("""
        CREATE TABLE complaint_edges_clean (
            id SERIAL PRIMARY KEY,
            source TEXT,
            target TEXT,
            weight FLOAT,
            similarity FLOAT
        )
    """))

# 데이터 삽입
nodes_df.to_sql("complaint_nodes_clean", engine, if_exists="append", index=False)
edges_df.to_sql("complaint_edges_clean", engine, if_exists="append", index=False)

print("PostgreSQL 테이블 덮어쓰기 완료: complaint_nodes_clean, complaint_edges_clean")
