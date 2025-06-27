import pandas as pd
import networkx as nx

print("📥 Loading edge list...")
df = pd.read_csv("keyword_edge_final_cleaned.csv")

print("🔗 Building graph...")
G = nx.Graph()
for _, row in df.iterrows():
    G.add_edge(row["source"], row["target"], weight=row["weight"])

print("🧮 Computing node degree...")
degree_dict = dict(G.degree())
nx.set_node_attributes(G, degree_dict, "degree")

print("📊 Filtering nodes with degree >= 2...")
filtered_nodes = [n for n, d in G.degree() if d >= 2]
H = G.subgraph(filtered_nodes).copy()

print(f"✅ Filtered subgraph has {H.number_of_nodes()} nodes and {H.number_of_edges()} edges")

print("💾 Exporting filtered edge list...")
edges_out = nx.to_pandas_edgelist(H)
edges_out.to_csv("keyword_edge_filtered.csv", index=False)
print("📄 Saved: keyword_edge_filtered.csv")