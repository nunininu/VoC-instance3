from sqlalchemy import create_engine
from dotenv import load_dotenv
import pandas as pd
import os
from gensim.models import Word2Vec

print("▶ 실행 시작: Word2Vec edge 추출")

# 환경변수 로드
load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(db_url)
print("✅ DB 연결 성공")

# 데이터 로딩
print("📥 불만 응답자 데이터 로딩 중...")
df = pd.read_sql("SELECT content FROM dummy_consulting WHERE label = 1", engine)
print(f"✅ 로딩 완료: {len(df)}건")

# 토큰화
print("✂️ 토큰화 중...")
tokenized = [row.split() for row in df["content"].dropna()]
print(f"✅ 토큰화 샘플: {tokenized[:2]}")

# Word2Vec 학습
print("🧠 Word2Vec 학습 중...")
model = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=2, workers=4, sg=1)
print(f"✅ 학습 완료: 단어 수 = {len(model.wv.index_to_key)}")

# 유사도 필터링
print("🔗 유사도 ≥ 0.5 edge 추출 중...")
edges = []
for word in model.wv.index_to_key:
    for similar_word, score in model.wv.most_similar(word, topn=20):
        if score >= 0.5:
            edges.append((word, similar_word, score))
print(f"✅ edge 수: {len(edges)}")

# DataFrame 생성 및 정제
df_edges = pd.DataFrame(edges, columns=["source", "target", "weight"])
df_edges = df_edges[df_edges["source"] != df_edges["target"]]
df_edges["pair"] = df_edges.apply(lambda row: tuple(sorted([row["source"], row["target"]])), axis=1)
df_edges = df_edges.drop_duplicates(subset="pair").drop(columns=["pair"])
print(f"🧹 중복 제거 후 edge 수: {len(df_edges)}")

# Top-K 필터링
TOP_K = 300
df_edges = df_edges.sort_values(by="weight", ascending=False).head(TOP_K)
print(f"🎯 Top-{TOP_K} 추출 완료")

# 저장
df_edges.to_sql("keyword_edge_final", engine, if_exists="replace", index=False)
df_edges.to_csv("keyword_edge_final.csv", index=False)
print("📤 저장 완료: keyword_edge_final (DB + CSV)")

print("✅ 실행 완료: Word2Vec edge 추출")