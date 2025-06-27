import os
import pandas as pd
import pyvis
import jinja2
from pyvis.network import Network
from matplotlib import cm
from matplotlib.colors import to_hex

print("▶ 실행 시작: Force-Directed Community 시각화")

# 데이터 로딩
df_edges = pd.read_csv("keyword_edge_annotated.csv")
df_nodes = pd.read_csv("keyword_nodes_annotated.csv")
print(f"✅ 엣지 수: {len(df_edges)}, 노드 수: {len(df_nodes)}")

# 그래프 객체 생성
net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white", notebook=False)
net.force_atlas_2based()

# PyVis 템플릿 직접 설정
template_path = os.path.join(os.path.dirname(pyvis.__file__), 'templates', 'template.html')
with open(template_path, "r", encoding="utf-8") as f:
    html_template = jinja2.Template(f.read())
net.template = html_template

# 커뮤니티별 색상 지정
communities = sorted(df_nodes["community"].unique())
color_map = {comm: to_hex(cm.tab20(i % 20)) for i, comm in enumerate(communities)}

# 노드 추가
for _, row in df_nodes.iterrows():
    net.add_node(
        row["node"],
        label=row["node"],
        title=f"커뮤니티: {row['community']}\\n중심성: {row['centrality']:.4f}",
        color=color_map[row["community"]],
        value=row["centrality"]
    )

# 엣지 추가
for _, row in df_edges.iterrows():
    net.add_edge(row["source"], row["target"], value=row["weight"])

# 시각화 출력
net.show("force_directed_community.html")
print("📤 시각화 완료: force_directed_community.html")
