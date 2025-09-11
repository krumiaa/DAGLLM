# send_agent_text_gpt.py
import requests
import json
import networkx as nx
import matplotlib.pyplot as plt

# Load agent_text text from file
with open("sample_agent_text.txt", "r", encoding="utf-8") as f:
    agent_text = f.read()

headers = {"Authorization": "Bearer supersecretdevkey"}
response = requests.post(
    "http://127.0.0.1:8001/store_agent_text",
    headers=headers,
    json={"user_id": "Don David", "text": agent_text}
)
# Parse response

print("Response Status Code:", response.status_code)
print("Raw Response Text:", response.text)

try:
    data = response.json()
except Exception as e:
    print("❌ Failed to parse JSON:", e)
    raise
    
data = response.json()
nodes = data.get("nodes", [])
edges = data.get("edges", [])

print("Nodes:", nodes)
print("Edges:", edges)

# Create graph
G = nx.DiGraph()
for node in nodes:
    G.add_node(node["name"], tags=node.get("tags", []))
for edge in edges:
    G.add_edge(
        edge["source"],
        edge["target"],
        label=f'{edge["relation"]} ({edge.get("confidence", "?")}%)'
    )

# Visualize graph
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=2000)
nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, width=2)
nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(
    G, pos, edge_labels=edge_labels,
    font_color='darkred', font_size=10,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.6)
)

plt.title("LLM to DAG Pipeline (GPT-based)", fontsize=16)
plt.axis("off")
plt.show()