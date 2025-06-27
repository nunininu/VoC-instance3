import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from gensim.models import Word2Vec
import networkx as nx
import pyvis.network
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
import os
from community import community_louvain
from jinja2 import Template

# ------------------ 설정 ------------------
TOP_N_NODES = 120                  # 시각화에 사용할 상위 노드 수
SIM_RANGE = np.arange(0.95, 0.60, -0.01)  # 유사도 임계값 탐색 범위

# ------------------ 템플릿 로드 ------------------
template_path = os.path.join(os.path.dirname(pyvis.network.__file__), "templates", "template.html")
with open(template_path, "r", encoding="utf-8") as f:
    html_template = Template(f.read())

# ------------------ DB 연결 ------------------
print("📦 환경변수 로드 중...")
os.environ.clear()
load_dotenv(dotenv_path=".env", override=True)

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")

engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# ------------------ 데이터 불러오기 ------------------
print("📥 키워드 불러오는 중...")
query = "SELECT keywords FROM dummy_analysis_result WHERE keywords IS NOT NULL AND keywords != ''"
df = pd.read_sql(query, engine)
documents = [[w.strip(",.?!") for w in row.split()] for row in df["keywords"]] # 쉼표 제거

# ------------------ Word2Vec 학습 ------------------
print("🧠 Word2Vec 훈련 중...")
model = Word2Vec(sentences=documents, vector_size=100, window=5, min_count=1, workers=4)
vocab = list(model.wv.index_to_key)

# ------------------ 유사도 임계값 자동 탐색 ------------------
print("🔍 최적 유사도 임계값 탐색 중...")
best_edges = []
best_threshold = None
for threshold in SIM_RANGE:
    edges = []
    for i, word1 in enumerate(vocab):
        for word2 in vocab[i+1:]:
            sim = model.wv.similarity(word1, word2)
            if sim > threshold:
                edges.append((word1, word2, float(sim)))
    G_test = nx.Graph()
    G_test.add_weighted_edges_from(edges)
    if G_test.number_of_nodes() == 0:
        continue
    partition = community_louvain.best_partition(G_test)
    n_communities = len(set(partition.values()))
    print(f"  유사도 {threshold:.2f}: 노드 {len(G_test.nodes())}, 커뮤니티 {n_communities}")
    if n_communities > 1:
        best_edges = edges
        best_threshold = threshold
        break

if best_threshold is None:
    raise RuntimeError("❌ 임계값 자동 튜닝 실패: 커뮤니티가 1개 이하로만 분리됩니다.")

print(f"✅ 선택된 유사도 임계값: {best_threshold:.2f}")

# ------------------ 전체 그래프 구성 ------------------
print("🌐 전체 그래프 구성 중...")
G_full = nx.Graph()
G_full.add_weighted_edges_from(best_edges)

# ------------------ 중심성 계산 및 상위 노드 선택 ------------------
print("📈 중심성 계산 및 필터링 중...")
centrality = nx.degree_centrality(G_full)
scaled = MinMaxScaler().fit_transform(np.array(list(centrality.values())).reshape(-1,1)).reshape(-1)
centrality_scaled = dict(zip(centrality.keys(), scaled))

top_nodes = sorted(centrality_scaled.items(), key=lambda x: x[1], reverse=True)[:TOP_N_NODES]
selected_nodes = set(k for k, _ in top_nodes)
G = G_full.subgraph(selected_nodes).copy()

# ------------------ 커뮤니티 탐지 ------------------
print("🧩 커뮤니티 탐지 중...")
partition = community_louvain.best_partition(G)
print("📊 최종 커뮤니티 수:", len(set(partition.values())))

# ------------------ 시각화 ------------------
print("🖼️ PyVis 시각화 구성 중...")
net = pyvis.network.Network(height="800px", width="100%", bgcolor="#222222", font_color="white", notebook=False)
net.template = html_template
net.repulsion(node_distance=200, central_gravity=0.3, spring_length=100, spring_strength=0.05, damping=0.09)

print("📌 시각화 대상 노드 샘플:")
print(list(G.nodes())[:10])

print("📌 시각화 대상 엣지 수:", G.number_of_edges())

for node in G.nodes():
    net.add_node(
        node,
        label=node,
        title=f"커뮤니티: {partition[node]}\\n중심성: {centrality_scaled[node]:.4f}",
        color=f"hsl({partition[node]*37 % 360}, 80%, 60%)",
        value=float(centrality_scaled[node])
    )

for u, v, d in G.edges(data=True):
    net.add_edge(u, v, value=float(d["weight"]))

# ------------------ 저장 ------------------
print("💾 HTML 파일 저장 중...")
html = net.generate_html()
with open("force_directed_keyword_graph_filtered_tuned.html", "w", encoding="utf-8") as f:
    f.write(html)

print("✅ force_directed_keyword_graph_filtered_tuned.html 생성 완료")
