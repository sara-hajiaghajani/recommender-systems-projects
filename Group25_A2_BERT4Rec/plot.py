import matplotlib.pyplot as plt
import json
import glob

colors = plt.cm.tab10.colors


def plot_final_metrics(json_files, save_path='final_metrics.png'):
    configs = []
    recalls = []
    ndcgs = []

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            for config in data["configurations"]:
                label = f"{config['layer_size']}-{config['hidden_size']}-{config['masking_ratio']}"
                configs.append(label)
                recalls.append(config["recall@10"])
                ndcgs.append(config["ndcg@10"])

    x = range(len(configs))
    plt.figure(figsize=(14, 6))

    # Line plots
    plt.plot(x, recalls, marker='o', label='Recall@10', color='royalblue', linewidth=2)
    plt.plot(x, ndcgs, marker='o', label='NDCG@10', color='darkorange', linewidth=2)

    # Add value labels
    for i, (r, n) in enumerate(zip(recalls, ndcgs)):
        plt.text(i, r + 0.001, f"{r:.3f}", ha='center', va='bottom', fontsize=9, color='royalblue')
        plt.text(i, n - 0.0015, f"{n:.3f}", ha='center', va='top', fontsize=9, color='darkorange')

    plt.xticks(x, configs, rotation=45, ha='right')
    plt.ylabel('Score')
    plt.ylim(min(min(recalls), min(ndcgs)) - 0.01, max(max(recalls), max(ndcgs)) + 0.01)
    plt.title('Model Performance Comparison (Recall@10 & NDCG@10)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


if __name__ == '__main__':


    json_files = glob.glob('results/6 Layers/*.json')
    json_files.sort()
    plot_final_metrics(json_files, save_path='results/6 Layers/6_Layers_comparison_plot.png')

