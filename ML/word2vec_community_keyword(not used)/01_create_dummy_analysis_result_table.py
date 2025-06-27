from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

print("📦 환경변수 로드 중...")
os.environ.clear()  # 캐시된 잘못된 환경변수 초기화
load_dotenv(dotenv_path=".env", override=True)

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")

db_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
print(f"🔌 DB 연결 주소 구성 완료: {DB_NAME}@{DB_HOST}:{DB_PORT}")


engine = create_engine(db_url, echo=False)

# 실행할 SQL 쿼리
query = """
DROP TABLE IF EXISTS dummy_analysis_result;

CREATE TABLE dummy_analysis_result AS
SELECT
    *,
    CASE WHEN random() > 0.5 THEN 1 ELSE 0 END AS label
FROM analysis_result;
"""

print("🚀 쿼리 실행 준비 완료")

try:
    with engine.begin() as conn:
        print("🔧 쿼리 실행 중...")
        conn.execute(text(query))  # ✔ commit 보장됨
        print("✅ dummy_analysis_result 테이블 생성 완료")
except Exception as e:
    print("❌ 쿼리 실행 중 오류 발생:", e)

