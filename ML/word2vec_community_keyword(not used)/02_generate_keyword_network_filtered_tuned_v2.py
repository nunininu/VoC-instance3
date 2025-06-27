import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine
from gensim.models import Word2Vec
import networkx as nx
import pyvis.network
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
from community import community_louvain
from jinja2 import Template

# 시각화 설정
TOP_N_NODES = 120  # 중심성이 높은 상위 노드 수

# 템플릿 로드 (PyVis 커스터마이징)
template_path = os.path.join(os.path.dirname(pyvis.network.__file__), "templates", "template.html")
with open(template_path, "r", encoding="utf-8") as f:
    html_template = Template(f.read())

# Step 0: 환경변수 로드 및 DB 연결
print("📦 환경변수 로드 중...")
os.environ.clear()
load_dotenv(dotenv_path=".env", override=True)

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# Step 1: 키워드 불러오기
print("📥 키워드 불러오는 중...")
query = "SELECT keywords FROM dummy_analysis_result WHERE keywords IS NOT NULL AND keywords != ''"
df = pd.read_sql(query, engine)
documents = [[w.strip(",.?!") for w in row.split()] for row in df["keywords"]]

# Step 2: Word2Vec 학습
print("🧠 Word2Vec 훈련 중...")
model = Word2Vec(sentences=documents, vector_size=100, window=5, min_count=1, workers=4)

# Step 3: 최적 유사도 임계값 자동 탐색
print("🔍 최적 유사도 임계값 탐색 중...")
vocab = list(model.wv.index_to_key)
best_threshold = None
best_score = -1
hist_data = []

for threshold in np.arange(0.95, 0.60, -0.05):
    temp_edges = []
    for i, w1 in enumerate(vocab):
        for w2 in vocab[i+1:]:
            sim = model.wv.similarity(w1, w2)
            if sim > threshold:
                temp_edges.append((w1, w2, float(sim)))
    
    G_tmp = nx.Graph()
    G_tmp.add_weighted_edges_from(temp_edges)
    if G_tmp.number_of_nodes() < 50:
        continue

    partition = community_louvain.best_partition(G_tmp)
    num_comms = len(set(partition.values()))
    hist_data.append((threshold, G_tmp.number_of_nodes(), num_comms))

    if num_comms >= 2:
        score = num_comms * np.log(G_tmp.number_of_nodes() + 1)
        if score > best_score:
            best_score = score
            best_threshold = threshold

print(f"✅ 선택된 유사도 임계값: {best_threshold}")

# Step 4: 최종 그래프 구성
print("🌐 전체 그래프 구성 중...")
edges = []
for i, w1 in enumerate(vocab):
    for w2 in vocab[i+1:]:
        sim = model.wv.similarity(w1, w2)
        if sim > best_threshold:
            if w1 != w2 and (w2, w1, sim) not in edges:
                edges.append((w1, w2, float(sim)))

G_full = nx.Graph()
G_full.add_weighted_edges_from(edges)

# Step 5: 중심성 계산 및 필터링
print("📈 중심성 계산 및 필터링 중...")
centrality = nx.degree_centrality(G_full)
scaled = MinMaxScaler().fit_transform(np.array(list(centrality.values())).reshape(-1,1)).reshape(-1)
centrality_scaled = dict(zip(centrality.keys(), scaled))

top_nodes = sorted(centrality_scaled.items(), key=lambda x: x[1], reverse=True)[:TOP_N_NODES]
selected_nodes = set(k for k, _ in top_nodes)
G = G_full.subgraph(selected_nodes).copy()

# Step 6: 커뮤니티 탐지
print("🧩 커뮤니티 탐지 중...")
partition = community_louvain.best_partition(G)
print("📊 최종 커뮤니티 수:", len(set(partition.values())))

# Step 7: 시각화 구성
print("🖼️ PyVis 시각화 구성 중...")
net = pyvis.network.Network(height="800px", width="100%", bgcolor="#222222", font_color="white", notebook=False)
net.template = html_template
net.repulsion(node_distance=180, central_gravity=0.3, spring_length=90, spring_strength=0.05, damping=0.09)

print("📌 시각화 대상 노드 샘플:")
print(list(G.nodes)[:10])

filtered_edges = [(u, v, d["weight"]) for u, v, d in G.edges(data=True)]
print(f"📌 시각화 대상 엣지 수: {len(filtered_edges)}")

for node in G.nodes():
    net.add_node(
        node,
        label=node,
        title=f"커뮤니티: {partition[node]}\\n중심성: {float(centrality_scaled[node]):.4f}",
        color=f"hsl({partition[node]*37 % 360}, 80%, 60%)",
        value=float(centrality_scaled[node])
    )

for u, v, w in filtered_edges:
    net.add_edge(u, v, value=float(w))

# Step 8: 저장
print("💾 HTML 파일 저장 중...")
html = net.generate_html()
with open("force_directed_keyword_graph_filtered_tuned_v2.html", "w", encoding="utf-8") as f:
    f.write(html)

print("✅ force_directed_keyword_graph_filtered_tuned_v2.html 생성 완료")

# Step 9: 중심성 분포 시각화
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
# 중심성 분포 시각화
plt.figure(figsize=(8, 4))
plt.hist(list(centrality_scaled.values()), bins=30, color="skyblue", edgecolor="black")
plt.title("Normalized Degree Centrality Distribution")
plt.xlabel("Degree Centrality (normalized)")
plt.ylabel("Node Count")
plt.grid(True)
plt.tight_layout()
plt.savefig("centrality_distribution.png")
plt.close()
# 중심성 분포 저장
print("✅ 중심성 분포 이미지 저장 완료: centrality_distribution.png")