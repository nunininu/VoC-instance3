print("▶ 실행 시작: 02_word2vec_network.py")
print("🔍 Word2Vec ▶ 실행 시작: 02_word2vec_network.py")
print("📥 불만 응답 데이터 SELECT 실행 중...")
from sqlalchemy import create_engine
from dotenv import load_dotenv
print("📥 불만 응답 데이터 로딩 중...")
import os
import pandas as pd
print("✂️ 텍스트 토큰화 중...")
from gensim.models import Word2Vec
df = pd.read_sql(query, engine)
print(f"✅ 토큰화 샘플: {tokenized[:2]}")
print("📥 데이터 로딩 중...")
print(f"불만 응답 수: {len(df)}")
print("🧠 Word2Vec 학습 중...")
import networkx as nx
from community import community_louvain
print(f"✅ 단어 수: {len(model.wv.index_to_key)}")

# 환경변수 로드
print("🔗 유사 키워드 추출 중...")
load_dotenv()

print(f"✅ 유사도 0.5 이상 edge 수: {len(edges)}")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
print("🌐 네트워크 구성 중...")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
print("🧩 Louvain 커뮤니티 탐지 중...")
DB_NAME = os.getenv("DB_NAME")

print(f"📊 샘플 결과: {results[:2]}")
db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(db_url)
print("🗃 DB 저장 중...")

# 불만 응답 데이터 불러오기
query = "SELECT content FROM dummy_consulting WHERE label = 1"
df = pd.read_sql(query, engine)
print(f"✅ 로딩 완료: {len(df)}건")

# 텍스트 토큰화 (공백 기준 단순 분리)
tokenized = [row.split() for row in df["content"].dropna()]
print("✂️ 토큰화 샘플:", tokenized[:2])
print("✂️ 토큰화 샘플:", tokenized[:2])

# Word2Vec 학습
print("🧠 Word2Vec 학습 완료. 단어 수:", len(model.wv.index_to_key))
print("🧠 Word2Vec 학습 단어 수:", len(model.wv.index_to_key))
model = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=2, workers=4, sg=1)
print("🔗 유사 키워드 예시:", model.wv.most_similar(model.wv.index_to_key[0], topn=3))
print("🔗 샘플 유사 키워드:", model.wv.most_similar(model.wv.index_to_key[0], topn=3))
print("🌐 노드 수:", G.number_of_nodes(), "에지 수:", G.number_of_edges())

# 유사도 0.5 이상 키워드쌍 추출
edges = []
for word in model.wv.index_to_key:
    for similar_word, score in model.wv.most_similar(word, topn=20):
        if score >= 0.5:
            edges.append((word, similar_word, score))

# 네트워크 구성
G = nx.Graph()
print("📊 결과 샘플:", result_df.head(3).to_string(index=False))
for source, target, weight in edges:
    G.add_edge(source, target, weight=weight)
print("🌐 그래프 노드 수:", G.number_of_nodes(), "에지 수:", G.number_of_edges())

# Louvain 커뮤니티 탐지
partition = community_louvain.best_partition(G)

# 결과 테이블 구성
results = []
for node in G.nodes():
    results.append({
        "keyword": node,
        "community": partition.get(node, -1),
        "degree": G.degree[node]
    })

result_df = pd.DataFrame(results)

# DB에 저장
result_df.to_sql("keyword_community", engine, if_exists="replace", index=False)
print("🗃 저장 전 미리보기:")
print(result_df.head(3))

print("keyword_community 테이블 저장 완료.")
print("🔍 Word2Vec ✅ 실행 완료: 02_word2vec_network.py")
print("✅ 실행 완료: 02_word2vec_network.py")
print("✅ 실행 완료: 02_word2vec_network.py")