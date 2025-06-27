import pandas as pd
from pyvis.network import Network

# 데이터 로드
df = pd.read_csv("keyword_edge_final.csv")

# 네트워크 초기화
net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black")
net.force_atlas_2based()

# 노드 추가
nodes = set(df['source']).union(set(df['target']))
for node in nodes:
    net.add_node(node, label=node)

# 엣지 추가
for _, row in df.iterrows():
    net.add_edge(row['source'], row['target'], value=row['weight'])

# 템플릿 우회 저장
net.save_graph("force_directed_graph.html")
print("✅ force_directed_graph.html 생성 완료 (템플릿 미사용)")
