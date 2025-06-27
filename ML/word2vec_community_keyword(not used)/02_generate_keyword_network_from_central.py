
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

# ===============================
# 설정: 파라미터
# ===============================
MIN_COUNT = 7
TOP_N_NODES = 80
EDGE_LIMIT = 5000
THRESHOLD_START = 0.99
THRESHOLD_END = 0.94
THRESHOLD_STEP = -0.005
DOMINANT_CENTRALITY_THRESHOLD = 0.95
DOMINANT_CENTRALITY_PORTION = 0.2

# ===============================
# 템플릿 로드
# ===============================
template_path = os.path.join(os.path.dirname(pyvis.network.__file__), "templates", "template.html")
with open(template_path, "r", encoding="utf-8") as f:
    html_template = Template(f.read())

# ===============================
# 환경변수 및 DB 연결
# ===============================
print("📦 환경변수 로드 중...")
os.environ.clear()
load_dotenv(dotenv_path=".env", override=True)

engine = create_engine(f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT','5432')}/{os.getenv('DB_NAME')}")

# ===============================
# 키워드 불러오기
# ===============================
print("📥 키워드 불러오는 중...")
df = pd.read_sql("SELECT keywords FROM dummy_analysis_result WHERE keywords IS NOT NULL AND keywords != ''", engine)
documents = [[w.strip(",.?!") for w in row.split()] for row in df["keywords"]]

# ===============================
# Word2Vec 훈련
# ===============================
print(f"🧠 Word2Vec 훈련 중... (min_count={MIN_COUNT})")
model = Word2Vec(sentences=documents, vector_size=100, window=5, min_count=MIN_COUNT, workers=4)

# ===============================
# 유사도 기반 그래프 구성
# ===============================
print("🔍 최적 유사도 임계값 탐색 중...")
vocab = list(model.wv.index_to_key)
best_threshold = None

for threshold in np.arange(THRESHOLD_START, THRESHOLD_END, THRESHOLD_STEP):
    edges_temp = []
    for i, w1 in enumerate(vocab):
        for w2 in vocab[i+1:]:
            sim = model.wv.similarity(w1, w2)
            if sim >= threshold:
                edges_temp.append((w1, w2, float(sim)))
    G_temp = nx.Graph()
    G_temp.add_weighted_edges_from(edges_temp)

    if G_temp.number_of_nodes() == 0:
        continue

    partition = community_louvain.best_partition(G_temp)
    if len(set(partition.values())) < 2 or len(edges_temp) > EDGE_LIMIT:
        continue

    centrality = nx.degree_centrality(G_temp)
    scaled = MinMaxScaler().fit_transform(np.array(list(centrality.values())).reshape(-1,1)).reshape(-1)
    if np.sum(scaled > DOMINANT_CENTRALITY_THRESHOLD) / len(scaled) > DOMINANT_CENTRALITY_PORTION:
        continue

    best_threshold = threshold
    break

if best_threshold is None:
    print("❌ 적절한 임계값을 찾지 못했습니다.")
    exit()

print(f"✅ 선택된 유사도 임계값: {best_threshold}")

# ===============================
# 최종 그래프 및 중심성 계산
# ===============================
print("🌐 전체 그래프 구성 중...")
edges = [(w1, w2, float(model.wv.similarity(w1, w2)))
         for i, w1 in enumerate(vocab)
         for w2 in vocab[i+1:]
         if model.wv.similarity(w1, w2) >= best_threshold]
G_full = nx.Graph()
G_full.add_weighted_edges_from(edges)

centrality = nx.degree_centrality(G_full)
scaled = MinMaxScaler().fit_transform(np.array(list(centrality.values())).reshape(-1, 1)).reshape(-1)
centrality_scaled = dict(zip(centrality.keys(), scaled))

top_nodes = sorted(centrality_scaled.items(), key=lambda x: x[1], reverse=True)[:TOP_N_NODES]
selected_nodes = set(k for k, _ in top_nodes)
G = G_full.subgraph(selected_nodes).copy()

# ===============================
# 중심 키워드 불러오기
# ===============================
print("📌 중심 키워드 불러오는 중...")
central_keywords_path = "community_central_keywords"
central_keywords = pd.read_csv(central_keywords_path)["keywords"].tolist()

# ===============================
# PyVis 시각화
# ===============================
print("🖼️ 시각화 구성 중...")
net = pyvis.network.Network(height="800px", width="100%", bgcolor="#222222", font_color="white", notebook=False)
net.template = html_template
net.force_atlas_2based(gravity=0.05, central_gravity=0.05, spring_length=100, spring_strength=0.01, damping=0.09)

central_color_map = {kw: f"hsl({i*50 % 360}, 90%, 60%)" for i, kw in enumerate(central_keywords)}
def resolve_color(node):
    if node in central_keywords:
        return central_color_map[node]
    for ck in central_keywords:
        if G.has_edge(node, ck):
            return central_color_map[ck]
    return "#888888"

for node in G.nodes():
    net.add_node(
        node,
        label=node,
        title=f"중심성: {centrality_scaled.get(node, 0):.4f}",
        color=resolve_color(node),
        value=1.2 if node in central_keywords else float(centrality_scaled.get(node, 0))
    )

for u, v, d in G.edges(data=True):
    net.add_edge(u, v, value=float(d["weight"]))

net.save_graph("force_directed_keyword_graph_by_central.html")
print("✅ force_directed_keyword_graph_by_central.html 생성 완료")
