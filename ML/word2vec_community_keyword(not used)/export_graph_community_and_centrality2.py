import pandas as pd
import networkx as nx
from community import community_louvain

print("📥 Loading edge data...")
df = pd.read_csv("keyword_edge_filtered2.csv")

print("🔗 Building graph...")
G = nx.Graph()
for _, row in df.iterrows():
    G.add_edge(row["source"], row["target"], weight=row["weight"])

print("🧭 Computing Louvain communities...")
partition = community_louvain.best_partition(G, weight='weight')
nx.set_node_attributes(G, partition, 'community')

print("🏛️ Computing degree centrality...")
centrality = nx.degree_centrality(G)
nx.set_node_attributes(G, centrality, 'centrality')

print("📄 Exporting annotated edge list...")
annotated_edges = nx.to_pandas_edgelist(G)
annotated_edges["source_community"] = annotated_edges["source"].map(partition)
annotated_edges["target_community"] = annotated_edges["target"].map(partition)
annotated_edges.to_csv("keyword_edge_annotated2.csv", index=False)

print("📄 Exporting node attributes...")
node_df = pd.DataFrame({
    "node": list(G.nodes),
    "community": [partition[n] for n in G.nodes],
    "centrality": [centrality[n] for n in G.nodes]
})
node_df.to_csv("keyword_nodes_annotated2.csv", index=False)

print("✅ Done: keyword_edge_annotated2.csv / keyword_nodes_annotated2.csv")
