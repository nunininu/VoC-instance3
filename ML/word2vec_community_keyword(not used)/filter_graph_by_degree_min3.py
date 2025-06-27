import pandas as pd
import networkx as nx

print("📥 Loading edge list...")
df = pd.read_csv("keyword_edge_final.csv")  # 또는 정확한 edge 파일명

print("🔧 Building graph...")
G = nx.Graph()
for _, row in df.iterrows():
    G.add_edge(row["source"], row["target"], weight=row["weight"])

print("📊 Computing node degree...")
degree_dict = dict(G.degree())

# ✅ 여기만 수정
min_degree = 3
nodes_to_keep = [node for node, deg in degree_dict.items() if deg >= min_degree]
G_filtered = G.subgraph(nodes_to_keep)

print(f"✅ Filtered subgraph has {len(G_filtered.nodes)} nodes and {len(G_filtered.edges)} edges")

print("📄 Exporting filtered edge list...")
filtered_edges = nx.to_pandas_edgelist(G_filtered)
filtered_edges.to_csv("keyword_edge_filtered2.csv", index=False)

print("✅ Saved: keyword_edge_filtered2.csv")
