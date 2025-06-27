from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import pandas as pd
from gensim.models import Word2Vec
import traceback

print("▶ keyword_edge 디버깅 실행 시작")

# 환경변수 로드
try:
    load_dotenv()
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME")
    print("✅ 환경변수 로드 완료")
except Exception as e:
    print("❌ 환경변수 로드 실패:", e)
    traceback.print_exc()

try:
    db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(db_url)
    print("✅ DB 연결 객체 생성 완료")
except Exception as e:
    print("❌ DB 연결 객체 생성 실패:", e)
    traceback.print_exc()

# 데이터 로딩
try:
    print("📥 dummy_consulting에서 label=1 텍스트 로딩 중...")
    query = "SELECT content AS text FROM dummy_consulting WHERE label = 1"
    df = pd.read_sql(query, engine)
    print(f"✅ {len(df)}건 로딩됨")
except Exception as e:
    print("❌ 데이터 로딩 실패:", e)
    traceback.print_exc()

# 토큰화
try:
    print("✂️ 텍스트 토큰화 중...")
    tokenized = [row.split() for row in df["text"].dropna()]
    print(f"✅ 토큰화 완료 (샘플): {tokenized[:2]}")
except Exception as e:
    print("❌ 토큰화 실패:", e)
    traceback.print_exc()

# Word2Vec 학습
try:
    print("🧠 Word2Vec 학습 중...")
    model = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=2, workers=4, sg=1)
    print(f"✅ Word2Vec 학습 완료: 단어 수 = {len(model.wv.index_to_key)}")
except Exception as e:
    print("❌ Word2Vec 학습 실패:", e)
    traceback.print_exc()

# 유사도 edge 추출
try:
    print("🔗 유사도 0.5 이상 edge 추출 중...")
    edges = []
    for word in model.wv.index_to_key:
        similar_words = model.wv.most_similar(word, topn=20)
        for similar_word, score in similar_words:
            if score >= 0.5:
                edges.append((word, similar_word, score))
    print(f"✅ 추출된 edge 수: {len(edges)}")
    print(f"🔍 샘플 edge: {edges[:3]}")
except Exception as e:
    print("❌ edge 추출 실패:", e)
    traceback.print_exc()

# 테이블 저장
try:
    print("📤 DB 저장 준비 중...")
    df_edges = pd.DataFrame(edges, columns=["source", "target", "similarity"])
    print(f"📄 저장 테이블 샘플:\n{df_edges.head()}")
    df_edges.to_sql("keyword_edge", engine, if_exists="replace", index=False)
    print("✅ keyword_edge 테이블 저장 완료")
except Exception as e:
    print("❌ DB 저장 실패:", e)
    traceback.print_exc()