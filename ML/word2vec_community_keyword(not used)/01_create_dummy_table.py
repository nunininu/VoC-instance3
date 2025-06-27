from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

# .env 파일 불러오기
load_dotenv()

# 환경변수에서 DB 접속 정보 읽기
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")

# 연결 문자열 생성
db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(db_url)

# DDL 실행
query = """
DROP TABLE IF EXISTS dummy_consulting;

CREATE TABLE dummy_consulting AS
SELECT
    *,
    CASE WHEN random() > 0.5 THEN 1 ELSE 0 END AS label
FROM consulting;
"""

with engine.begin() as conn:
    conn.execute(text(query))

print("dummy_consulting 테이블 생성 완료.")
