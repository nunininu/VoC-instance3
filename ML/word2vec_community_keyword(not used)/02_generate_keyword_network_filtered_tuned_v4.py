import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine
from gensim.models import Word2Vec
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from community import community_louvain
import matplotlib.pyplot as plt
import pyvis.network
from jinja2 import Template
from collections import Counter

# ===============================
# 설정: 파라미터
# ===============================
TOP_N_NODES = 120
THRESHOLD_START = 0.99
THRESHOLD_END = 0.85
THRESHOLD_STEP = -0.01
EDGE_LIMIT = 5000
DOMINANT_CENTRALITY_THRESHOLD = 0.95
DOMINANT_CENTRALITY_PORTION = 0.4
MIN_COUNT = 4

# ===============================
# PyVis 템플릿 수동 할당
# ===============================
template_path = os.path.join(os.path.dirname(pyvis.network.__file__), "templates", "template.html")
with open(template_path, "r", encoding="utf-8") as f:
    html_template = Template(f.read())

# ===============================
# 환경변수 로드 및 DB 연결
# ===============================
print("📦 환경변수 로드 중...")
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
print("📥 키워드 불러오는 중...")
query = "SELECT keywords FROM dummy_analysis_result WHERE keywords IS NOT NULL AND keywords != ''"
df = pd.read_sql(query, engine)
documents = [[w.strip(",.?!") for w in row.split()] for row in df["keywords"]]

# ===============================
# Word2Vec 훈련
# ===============================
print(f"🧠 Word2Vec 훈련 중... (min_count={MIN_COUNT})")
model = Word2Vec(sentences=documents, vector_size=100, window=5, min_count=MIN_COUNT, workers=4)

# ===============================
# 임계값 자동 탐색
# ===============================
print("🔍 최적 유사도 임계값 탐색 중...")
best_threshold = None
vocab = list(model.wv.index_to_key)

for threshold in np.arange(THRESHOLD_START, THRESHOLD_END, THRESHOLD_STEP):
    G_temp = nx.Graph()
    edges_temp = []
    for i, w1 in enumerate(vocab):
        for w2 in vocab[i+1:]:
            sim = model.wv.similarity(w1, w2)
            if sim >= threshold:
                edges_temp.append((w1, w2, float(sim)))
    G_temp.add_weighted_edges_from(edges_temp)

    if G_temp.number_of_nodes() == 0:
        continue

    partition = community_louvain.best_partition(G_temp)
    num_communities = len(set(partition.values()))
    edge_count = len(edges_temp)

    print(f"  ▶ threshold={threshold:.2f} | nodes={G_temp.number_of_nodes()} | edges={edge_count} | communities={num_communities}")

    # 커뮤니티 수 제한
    if num_communities < 2:
        continue

    # 엣지 수 제한
    if edge_count > EDGE_LIMIT:
        continue

    # 중심성 분포 제한
    centrality = nx.degree_centrality(G_temp)
    scaled = MinMaxScaler().fit_transform(np.array(list(centrality.values())).reshape(-1, 1)).reshape(-1)
    high = np.sum(scaled > DOMINANT_CENTRALITY_THRESHOLD) / len(scaled)
    if high > DOMINANT_CENTRALITY_PORTION:
        print(f"    ⚠️ 중심성 상위 몰림 비율 {high:.2f} → 탈락")
        continue

    best_threshold = threshold
    break

if best_threshold is None:
    print("❌ 적절한 임계값을 찾지 못했습니다.")
    exit()

print(f"✅ 선택된 유사도 임계값: {best_threshold}")

# ===============================
# 최종 그래프 구성
# ===============================
print("🌐 전체 그래프 구성 중...")
edges = []
for i, w1 in enumerate(vocab):
    for w2 in vocab[i+1:]:
        sim = model.wv.similarity(w1, w2)
        if sim >= best_threshold:
            edges.append((w1, w2, float(sim)))

G_full = nx.Graph()
G_full.add_weighted_edges_from(edges)

# ===============================
# 중심성 계산 및 필터링
# ===============================
print("📈 중심성 계산 및 필터링 중...")
centrality = nx.degree_centrality(G_full)
scaled = MinMaxScaler().fit_transform(np.array(list(centrality.values())).reshape(-1, 1)).reshape(-1)
centrality_scaled = dict(zip(centrality.keys(), scaled))

# 상위 노드 필터링
top_nodes = sorted(centrality_scaled.items(), key=lambda x: x[1], reverse=True)[:TOP_N_NODES]
selected_nodes = set(k for k, _ in top_nodes)
G = G_full.subgraph(selected_nodes).copy()

# ===============================
# 커뮤니티 탐지
# ===============================
print("🧩 커뮤니티 탐지 중...")
partition = community_louvain.best_partition(G)
print(f"📊 최종 커뮤니티 수: {len(set(partition.values()))}")

# ===============================
# PyVis 시각화
# ===============================
print("🖼️ PyVis 시각화 구성 중...")
net = pyvis.network.Network(height="800px", width="100%", bgcolor="#222222", font_color="white", notebook=False)
net.template = html_template
net.repulsion(node_distance=300, central_gravity=0.3, spring_length=150, spring_strength=0.06, damping=0.09)

print("📌 시각화 대상 노드 샘플:")
print(list(G.nodes())[:10])
print(f"📌 시각화 대상 엣지 수: {G.number_of_edges()}")

for node in G.nodes():
    net.add_node(
        node,
        label=node,
        title=f"커뮤니티: {partition[node]}\n중심성: {centrality_scaled[node]:.4f}",
        color=f"hsl({partition[node]*37 % 360}, 80%, 60%)",
        value=float(centrality_scaled[node])
    )

for u, v, d in G.edges(data=True):
    net.add_edge(u, v, value=float(d["weight"]))

# ===============================
# 중심성 히스토그램
# ===============================
plt.figure(figsize=(8, 5))
plt.hist(scaled, bins=30, color="skyblue", edgecolor="black")
plt.title("Normalized Degree Centrality Distribution")
plt.xlabel("Degree Centrality (normalized)")
plt.ylabel("Node Count")
plt.tight_layout()
plt.savefig("centrality_distribution.png")
plt.close()
print("✅ 중심성 분포 이미지 저장 완료: centrality_distribution.png")

# ===============================
# HTML 저장
# ===============================
html = net.generate_html()
with open("force_directed_keyword_graph_filtered_tuned_v4.html", "w", encoding="utf-8") as f:
    f.write(html)

print("✅ force_directed_keyword_graph_filtered_tuned_v4.html 생성 완료")
