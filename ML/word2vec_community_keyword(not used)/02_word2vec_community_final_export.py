from sqlalchemy import create_engine
from dotenv import load_dotenv
import pandas as pd
import os
import networkx as nx
from community import community_louvain

print("▶ 실행 시작: Word2Vec community 분할")

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
print("📥 keyword_edge_final 로딩 중...")
df = pd.read_sql("SELECT * FROM keyword_edge_final", engine)
print(f"✅ edge 수: {len(df)}")

# 그래프 구성
print("🌐 네트워크 그래프 구성 중...")
G = nx.Graph()
for source, target, weight in df.values:
    G.add_edge(source, target, weight=weight)
print(f"✅ 노드 수: {G.number_of_nodes()} / 엣지 수: {G.number_of_edges()}")

# Louvain 분할
print("🧠 Louvain community 탐지 중...")
partition = community_louvain.best_partition(G)
print(f"✅ 커뮤니티 수: {len(set(partition.values()))}")

# 결과 저장
print("📄 community 테이블 생성 중...")
records = []
for node in G.nodes():
    records.append({
        "keyword": node,
        "community": partition.get(node, -1),
        "degree": G.degree[node]
    })
df_nodes = pd.DataFrame(records)

df_nodes.to_sql("keyword_community_final", engine, if_exists="replace", index=False)
df_nodes.to_csv("keyword_community_final.csv", index=False)
print("📤 저장 완료: keyword_community_final (DB + CSV)")

print("✅ 실행 완료: Word2Vec community 분할")