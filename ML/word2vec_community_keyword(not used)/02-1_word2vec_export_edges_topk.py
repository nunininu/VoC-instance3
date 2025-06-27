from sqlalchemy import create_engine
from dotenv import load_dotenv
import pandas as pd
import os
import traceback

print("▶ keyword_edge_topk 테이블 생성 시작")

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

# grouped 테이블 불러오기
try:
    df = pd.read_sql("SELECT * FROM keyword_edge_grouped", engine)
    print(f"📥 keyword_edge_grouped 로딩 완료: {len(df)}건")
except Exception as e:
    print("❌ 테이블 로딩 실패:", e)
    traceback.print_exc()

# 상위 1000개 edge 추출
try:
    df_topk = df.sort_values(by="weight", ascending=False).head(1000)
    print("✅ weight 기준 상위 1000개 edge 추출 완료")
    print("📌 샘플:")
    print(df_topk.head(3))
except Exception as e:
    print("❌ 정렬 또는 추출 실패:", e)
    traceback.print_exc()

# 저장
try:
    df_topk.to_sql("keyword_edge_topk", engine, if_exists="replace", index=False)
    print("✅ keyword_edge_topk 테이블 저장 완료")
except Exception as e:
    print("❌ 저장 실패:", e)
    traceback.print_exc()
