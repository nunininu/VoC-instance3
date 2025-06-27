import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine
from gensim.models import Word2Vec
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from community import community_louvain

# ===============================
# 설정
# ===============================
MIN_COUNT = 3
SIMILARITY_THRESHOLD = 0.95
TOP_N_PER_COMMUNITY = 5

# ===============================
# 환경변수 로드 및 DB 연결
# ===============================
os.environ.clear()
load_dotenv(dotenv_path=".env", override=True)

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# ===============================
# 키워드 불러오기
# ===============================
query = "SELECT keywords FROM dummy_analysis_result WHERE keywords IS NOT NULL AND keywords != ''"
df = pd.read_sql(query, engine)
documents = [[w.strip(",.?!") for w in row.split()] for row in df["keywords"]]

# ===============================
# Word2Vec 훈련
# ===============================
model = Word2Vec(sentences=documents, vector_size=100, window=5, min_count=MIN_COUNT, workers=4)
vocab = list(model.wv.index_to_key)

# ===============================
# 유사도 그래프 구성
# ===============================
edges = []
for i, w1 in enumerate(vocab):
    for w2 in vocab[i+1:]:
        sim = model.wv.similarity(w1, w2)
        if sim >= SIMILARITY_THRESHOLD:
            edges.append((w1, w2, float(sim)))

G = nx.Graph()
G.add_weighted_edges_from(edges)

# ===============================
# 커뮤니티 탐지
# ===============================
partition = community_louvain.best_partition(G)

# ===============================
# 중심성 계산 및 정규화
# ===============================
centrality = nx.degree_centrality(G)
scaled = MinMaxScaler().fit_transform(np.array(list(centrality.values())).reshape(-1, 1)).reshape(-1)
centrality_scaled = dict(zip(centrality.keys(), scaled))

# ===============================
# 커뮤니티별 중심 키워드 추출
# ===============================
df_result = pd.DataFrame({
    "keywords": list(G.nodes()),
    "community": [partition[n] for n in G.nodes()],
    "centrality": [centrality_scaled[n] for n in G.nodes()]
})

summary = (
    df_result.sort_values(["community", "centrality"], ascending=[True, False])
    .groupby("community")
    .head(TOP_N_PER_COMMUNITY)
    .sort_values(["community", "centrality"], ascending=[True, False])
    .reset_index(drop=True)
)

# ===============================
# 저장
# ===============================
summary.to_csv("community_central_keywords.csv", index=False, encoding="utf-8-sig")
print("✅ 중심 키워드 요약 저장 완료: community_central_keywords.csv")
