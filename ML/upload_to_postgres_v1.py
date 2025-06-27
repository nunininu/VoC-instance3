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

# 테이블명 설정
nodes_table = "complaint_nodes_clean_v1"
edges_table = "complaint_edges_clean_v1"

with engine.begin() as conn:
    # 기존 _v1 테이블 제거
    conn.execute(text(f"DROP TABLE IF EXISTS {nodes_table}"))
    conn.execute(text(f"DROP TABLE IF EXISTS {edges_table}"))

    # _v1 테이블 생성 (category 포함)
    conn.execute(text(f"""
        CREATE TABLE {nodes_table} (
            id TEXT PRIMARY KEY,
            label TEXT,
            category TEXT,
            size FLOAT
        )
    """))

    conn.execute(text(f"""
        CREATE TABLE {edges_table} (
            id SERIAL PRIMARY KEY,
            source TEXT,
            target TEXT,
            weight FLOAT,
            similarity FLOAT,
            source_category TEXT,
            target_category TEXT
        )
    """))

# 데이터 삽입
nodes_df.to_sql(nodes_table, engine, if_exists="append", index=False)
edges_df.to_sql(edges_table, engine, if_exists="append", index=False)

print(f"PostgreSQL 테이블 업로드 완료: {nodes_table}, {edges_table}")
