import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Load saved results from the JSON file
results_file = "results/results.json"
if not os.path.exists(results_file):
    raise FileNotFoundError("No results.json file found in the 'results' directory.")

with open(results_file, "r") as f:
    results_data = json.load(f)

# Extract configuration labels and metrics
configs = results_data["configurations"]

labels = [f"emb{cfg['embedding_dim']}_mlp{'-'.join(map(str, cfg['mlp_layers']))}" for cfg in configs]
recalls = [cfg['recall@10'] for cfg in configs]
ndcgs = [cfg['ndcg@10'] for cfg in configs]

# Plot Recall@10 and NDCG@10 side-by-side
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, recalls, width, label='Recall@10')
bars2 = ax.bar(x + width/2, ndcgs, width, label='NDCG@10')

ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison (Recall@10 & NDCG@10)')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Annotate bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

# Save and show
os.makedirs("results/plots", exist_ok=True)
plt.tight_layout()
plt.savefig("results/plots/architecture_comparison_bar.png")
plt.show()