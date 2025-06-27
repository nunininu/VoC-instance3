import pandas as pd
from pyvis.network import Network
from jinja2 import Template

# 📥 Load data
df = pd.read_csv("keyword_edge_final_cleaned.csv")

# 🌐 Initialize graph
net = Network(height="1000px", width="100%", bgcolor="#222222", font_color="white")

nodes = set(df["source"]).union(set(df["target"]))
for node in nodes:
    net.add_node(node, label=node)

for _, row in df.iterrows():
    net.add_edge(row["source"], row["target"], value=row["weight"])

# 🩹 Patch: force template assignment manually
with open(".venv/lib/python3.12/site-packages/pyvis/templates/template.html", "r", encoding="utf-8") as f:
    net.template = Template(f.read())

# 💾 Export HTML
net.show("force_directed_graph_cleaned.html")
print("✅ Exported: force_directed_graph_cleaned.html")
