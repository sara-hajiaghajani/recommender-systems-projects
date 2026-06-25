import datetime
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from evaluation import evaluate_model
from model import NCF
from preprocessing import preprocessing
from train import train_model

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Device configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


def plot_loss_curves(train_losses, valid_losses, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def run_experiment(embedding_dim, mlp_layers, neg_ratio=4, batch_size=1024,
                   learning_rate=0.001, weight_decay=1e-5, n_epochs=30, early_stopping_patience=5,
                   preprocessed_data=None, save_model=False):
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Create data loaders
    train_dataset, valid_dataset, test_dataset, n_users, n_items, id_mappings = preprocessed_data

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 2. Initialize model
    model = NCF(n_users, n_items, embedding_dim, mlp_layers).to(device)

    # Print model summary
    print(f"Model Configuration: embedding_dim={embedding_dim}, mlp_layers={mlp_layers}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # 3. Train model
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6
    )

    print("Training model...")
    model, train_losses, valid_losses = train_model(
        model, train_loader, valid_loader, criterion, optimizer,
        n_epochs=n_epochs, early_stopping_patience=early_stopping_patience, scheduler=scheduler, device=device
    )

    # Create a unique identifier for this configuration
    config_id = f"emb{embedding_dim}_mlp{'_'.join(map(str, mlp_layers))}"

    # Plot loss curves
    plot_loss_curves(train_losses, valid_losses, save_path=f"results/loss_curves_{config_id}.png")

    # 4. Evaluate model
    print("Evaluating model...")
    recall_10, ndcg_10 = evaluate_model(model, test_loader, k=10, device=device)

    print(f"Recall@10: {recall_10:.4f}")
    print(f"NDCG@10: {ndcg_10:.4f}")

    # Save model
    if save_model:
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'embedding_dim': embedding_dim,
                'mlp_layers': mlp_layers
            },
            'id_mappings': id_mappings
        }, f"results/ncf_model_{config_id}.pt")

    # Save results
    # Create a unique identifier for this configuration
    config_id = f"emb{embedding_dim}_mlp{'_'.join(map(str, mlp_layers))}"

    # Create a result entry for this run
    current_result = {
        'n_users': n_users,
        'n_items': n_items,
        'embedding_dim': embedding_dim,
        'mlp_layers': mlp_layers,
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'recall@10': recall_10,
        'ndcg@10': ndcg_10,
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Load previous results if they exist, or initialize empty list
    results_file = "results/results.json"
    if os.path.exists(results_file):
        try:
            with open(results_file, "r") as f:
                all_results = json.load(f)
        except:
            all_results = {"configurations": []}
    else:
        all_results = {"configurations": []}

    # Append new result
    all_results["configurations"].append(current_result)

    # Save all results
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=4)

    # Also save this specific configuration separately
    with open(f"results/config_{config_id}.json", "w") as f:
        json.dump(current_result, f, indent=4)

    print(f"Done! Results saved to results/ (Configuration ID: {config_id})")

    # Return results
    return model, {
        'embedding_dim': embedding_dim,
        'mlp_layers': mlp_layers,
        'neg_ratio': neg_ratio,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'recall@10': recall_10,
        'ndcg@10': ndcg_10
    }


def main():
    """Main function to run multiple experiments with different configurations"""

    # Create directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Define configurations to test
    configurations = [
        # Format: (embedding_dim, mlp_layers)
        (32, [64, 32, 16]),  # Smaller model
        (64, [128, 64, 32]),  # Medium model (default)
        (128, [256, 128, 64]),  # Larger model
        (64, [128, 64])  # Fewer MLP layers
    ]

    # Other hyperparameters
    common_params = {
        'neg_ratio': 4,
        'batch_size': 1024,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'n_epochs': 30,
        'early_stopping_patience': 5
    }

    # Run preprocessing once and reuse for all configurations
    preprocessed_data = preprocessing(neg_ratio=common_params['neg_ratio'])

    # Run experiments for each configuration
    for embedding_dim, mlp_layers in configurations:
        print(f"\n{'=' * 80}")
        print(f"Running experiment with embedding_dim={embedding_dim}, mlp_layers={mlp_layers}")
        print(f"{'=' * 80}\n")

        # Run experiment
        run_experiment(
            embedding_dim=embedding_dim,
            mlp_layers=mlp_layers,
            preprocessed_data=preprocessed_data,
            **common_params
        )


if __name__ == "__main__":
    main()
