from sqlalchemy import create_engine
from dotenv import load_dotenv
import pandas as pd
import os
from gensim.models import Word2Vec
import networkx as nx
from community import community_louvain

print("▶ Word2Vec 네트워크 + 커뮤니티 통합 스크립트 시작")

# 환경 변수 로드
load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(db_url)
print("✅ DB 연결 완료")

# 불만 응답 로딩
df = pd.read_sql("SELECT content FROM dummy_consulting WHERE label = 1", engine)
print(f"📥 불만 응답 {len(df)}건 로딩 완료")

# 토큰화
tokenized = [row.split() for row in df["content"].dropna()]
print("✂️ 토큰화 완료")

# Word2Vec 학습
model = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=2, workers=4, sg=1)
print(f"🧠 Word2Vec 학습 완료 (단어 수: {len(model.wv.index_to_key)})")

# sim ≥ 0.5 edge 추출
edges = []
for word in model.wv.index_to_key:
    for similar_word, score in model.wv.most_similar(word, topn=20):
        if score >= 0.5:
            edges.append((word, similar_word, score))
print(f"🔗 유사도 필터링 후 edge 수: {len(edges)}")

# DataFrame 생성
df_edges = pd.DataFrame(edges, columns=["source", "target", "weight"])

# 자기참조 제거
df_edges = df_edges[df_edges["source"] != df_edges["target"]]

# 무방향 중복 제거
df_edges["pair"] = df_edges.apply(lambda row: tuple(sorted([row["source"], row["target"]])), axis=1)
df_edges = df_edges.drop_duplicates(subset="pair").drop(columns=["pair"])
print(f"🧹 중복 제거 후 edge 수: {len(df_edges)}")

# Top-K 필터링
TOP_K = 300
df_edges = df_edges.sort_values(by="weight", ascending=False).head(TOP_K)
print(f"🎯 Top-{TOP_K} edge 추출 완료")

# keyword_edge_final 저장
df_edges.to_sql("keyword_edge_final", engine, if_exists="replace", index=False)
df_edges.to_csv("keyword_edge_final.csv", index=False)
print("📤 keyword_edge_final 저장 완료")

# 커뮤니티 탐지
G = nx.Graph()
for source, target, weight in df_edges.values:
    G.add_edge(source, target, weight=weight)

partition = community_louvain.best_partition(G)

# 노드별 community 저장
node_records = []
for node in G.nodes():
    node_records.append({
        "keyword": node,
        "community": partition.get(node, -1),
        "degree": G.degree[node]
    })

df_community = pd.DataFrame(node_records)
df_community.to_sql("keyword_community_final", engine, if_exists="replace", index=False)
df_community.to_csv("keyword_community_final.csv", index=False)
print("📤 keyword_community_final 저장 완료")

print("✅ 실행 완료: Word2Vec 네트워크 + 커뮤니티 통합")
