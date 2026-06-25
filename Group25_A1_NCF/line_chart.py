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

# Create x positions for each label
x = np.arange(len(labels))

# Plot as line chart
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_ylim(0.85, .9)
ax.plot(x, recalls, marker='o', linestyle='-', label='Recall@10')
ax.plot(x, ndcgs, marker='o', linestyle='-', label='NDCG@10')

ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison (Recall@10 & NDCG@10)')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Annotate each data point with its value
for i, value in enumerate(recalls):
    ax.annotate(f'{value:.3f}', xy=(x[i], value), xytext=(0, 5),
                textcoords="offset points", ha='center', va='bottom')
for i, value in enumerate(ndcgs):
    ax.annotate(f'{value:.3f}', xy=(x[i], value), xytext=(0, 5),
                textcoords="offset points", ha='center', va='bottom')

# Save and show the plot
os.makedirs("results/plots", exist_ok=True)
plt.tight_layout()
plt.savefig("results/plots/architecture_comparison_line.png")
plt.show()
