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
documents = [row.split() for row in df["keywords"]]

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
        if sim > 0.6:
            edges.append((word1, word2, sim))

# Step 4: NetworkX 그래프 생성 및 중심성 계산
print("🌐 그래프 구성 중...")
G = nx.Graph()
G.add_weighted_edges_from(edges)
centrality = nx.degree_centrality(G)
scaled = MinMaxScaler().fit_transform(np.array(list(centrality.values())).reshape(-1,1)).reshape(-1)
centrality_scaled = dict(zip(centrality.keys(), scaled))

# Step 5: 커뮤니티 라벨링
print("🧩 커뮤니티 탐지 중...")
partition = community_louvain.best_partition(G)


# Step 6: PyVis 시각화
print("🖼️ 시각화 구성 중...")
net = pyvis.network.Network(height="800px", width="100%", bgcolor="#222222", font_color="white", notebook=False)
net.template = html_template  # 인스턴스에 직접 할당
net.force_atlas_2based()

for node in G.nodes():
    net.add_node(
        node,
        label=node,
        title=f"커뮤니티: {partition[node]}\\n중심성: {float(centrality_scaled[node]):.4f}",
        color=f"hsl({partition[node]*37 % 360}, 80%, 60%)",
        value=float(centrality_scaled[node])  # float32 → float
    )

for source, target, weight in edges:
    net.add_edge(source, target, value=float(weight))  # float32 → float

# 오류 회피를 위한 수동 HTML 저장
html = net.generate_html()
with open("force_directed_keyword_graph.html", "w", encoding="utf-8") as f:
    f.write(html)

print("✅ force_directed_keyword_graph.html 생성 완료")
