import pandas as pd
import networkx as nx

print("🔹 CSV 파일 로드")
df = pd.read_csv("keyword_edge_cleaned.csv")
print(f"▶ 총 {len(df)}개의 엣지 로드됨")

print("🔹 DiGraph 생성")
G = nx.DiGraph()
for _, row in df.iterrows():
    G.add_edge(row["source"], row["target"], weight=row["similarity"])
print(f"▶ 현재 엣지 수: {G.number_of_edges()}")

print("🔹 순환 제거 루프 시작")
removed = 0
while True:
    try:
        cycle = nx.find_cycle(G, orientation="original")
        # 가장 낮은 weight 엣지를 제거
        edge_weights = [(e[0], e[1], G[e[0]][e[1]]["weight"]) for e in cycle]
        min_edge = min(edge_weights, key=lambda x: x[2])
        G.remove_edge(min_edge[0], min_edge[1])
        removed += 1
    except nx.NetworkXNoCycle:
        break

print(f"✅ 순환 제거 완료 — 제거된 엣지 수: {removed}")
print(f"▶ 최종 엣지 수: {G.number_of_edges()}")
print(f"▶ DAG 여부 확인: {nx.is_directed_acyclic_graph(G)}")

# CSV로 저장
output_df = pd.DataFrame([
    {"source": u, "target": v, "similarity": G[u][v]["weight"]}
    for u, v in G.edges()
])
output_df.to_csv("keyword_edge_acyclic.csv", index=False)
print("✅ 저장 완료: keyword_edge_acyclic.csv")
