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

# 유사도 기준 강화 및 중심성 상위 노드 제한 축소
SIMILARITY_THRESHOLD = 0.85
TOP_N_NODES = 120

# 템플릿 로드 및 인스턴스에 직접 할당
template_path = os.path.join(os.path.dirname(pyvis.network.__file__), "templates", "template.html")
with open(template_path, "r", encoding="utf-8") as f:
    html_template = Template(f.read())

print("📦 환경변수 로드 중...")
os.environ.clear()  # 캐시된 잘못된 환경변수 초기화
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

# 쉼표 제거
documents = [[w.strip(",.?!") for w in row.split()] for row in df["keywords"]]


# Step 2: Word2Vec 훈련
print("🧠 Word2Vec 훈련 중...")
model = Word2Vec(sentences=documents, vector_size=100, window=5, min_count=1, workers=4)

# Step 3: 유사도 기반 엣지 추출
print("🔗 엣지 구성 중...")
edges = []
vocab = list(model.wv.index_to_key)
for i, word1 in enumerate(vocab):
    for word2 in vocab[i+1:]:
        sim = model.wv.similarity(word1, word2)
        if sim > SIMILARITY_THRESHOLD:
            # 유사도가 기준 이상인 경우에만 엣지 추가
            if word1 != word2 and (word2, word1, sim) not in edges:
                edges.append((word1, word2, float(sim)))



# Step 4: NetworkX 그래프 생성 및 중심성 계산
print("🌐 그래프 구성 중...")
G_full = nx.Graph()
G_full.add_weighted_edges_from(edges)

# 중심성 계산 및 정규화
centrality = nx.degree_centrality(G_full)
scaled = MinMaxScaler().fit_transform(np.array(list(centrality.values())).reshape(-1,1)).reshape(-1)
centrality_scaled = dict(zip(centrality.keys(), scaled))

# 상위 노드 필터링 centrality_scaled에서 상위 80개 노드 선택
top_nodes = sorted(centrality_scaled.items(), key=lambda x: x[1], reverse=True)[:TOP_N_NODES]
selected_nodes = set(k for k, _ in top_nodes)
G = G_full.subgraph(selected_nodes).copy()

# Step 5: 커뮤니티 라벨링
print("🧩 커뮤니티 탐지 중...")
partition = community_louvain.best_partition(G)
print("📊 커뮤니티 수:", len(set(partition.values())))

# Step 6: PyVis 시각화
print("🖼️ 시각화 구성 중...")
net = pyvis.network.Network(height="800px", width="100%", bgcolor="#222222", font_color="white", notebook=False)
net.template = html_template  # 인스턴스에 직접 할당

# 그래프 모양
# net.force_atlas_2based()
net.repulsion(node_distance=200, central_gravity=0.3, spring_length=100, spring_strength=0.05, damping=0.09)

assert len(G.nodes) > 0, "⚠️ 노드 수 0"
assert len(G.edges) > 0, "⚠️ 엣지 수 0"

print("시각화 대상 노드 목록:")
print(list(G.nodes())[:10])  # 일부만 확인

print("시각화 대상 엣지 샘플:")
print(edges[:5])


for node in G.nodes():
    net.add_node(
        node,
        label=node,
        title=f"커뮤니티: {partition[node]}\\n중심성: {float(centrality_scaled[node]):.4f}",
        color=f"hsl({partition[node]*37 % 360}, 80%, 60%)",
        value=float(centrality_scaled[node])  # float32 → float
    )

# G 기준 엣지만 필터링하여 재구성
filtered_edges = [(u, v, d["weight"]) for u, v, d in G.edges(data=True)]
for source, target, weight in filtered_edges:
    net.add_edge(source, target, value=float(weight))

print(f"노드 수: {len(G.nodes())}, 엣지 수: {len(filtered_edges)}")

for source, target, weight in edges:
    if source in G and target in G:
        # G에 있는 노드만 엣지 추가
        net.add_edge(source, target, value=float(weight))  # float32 → float

# Step 7: HTML 파일로 저장
print("💾 HTML 파일로 저장 중..."
      )
# 오류 회피를 위한 수동 HTML 저장
html = net.generate_html()
with open("force_directed_keyword_graph_filtered4.html", "w", encoding="utf-8") as f:
    f.write(html)

print("✅ force_directed_keyword_graph_filtered4.html 생성 완료")
