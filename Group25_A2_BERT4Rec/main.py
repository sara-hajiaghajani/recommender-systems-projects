import os
import random

import pandas as pd
import numpy as np
import torch
import json
from torch.utils.data import DataLoader

from datetime import datetime
from evaluation import evaluate
from model import BERT4Rec
from preprocessing import BERT4RecDataset
from train import train

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(RANDOM_SEED)
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    print("Using CPU device")

DATA_PATH = 'data/ml-1m/ratings.dat'  # Path to MovieLens 1M ratings

# Data Preprocessing Parameters
MIN_USER_INTERACTIONS = 5
MAX_SEQ_LENGTH = 50
RATING_THRESHOLD = 4
TEST_SIZE = 0.15
VALID_SIZE = 0.15

# Model Parameters
HIDDEN_SIZE = 256
NUM_ATTENTION_HEADS = 8
NUM_TRANSFORMER_LAYERS = 6
DROPOUT_PROB = 0.2
MASK_PROB = 0.50

# Training Parameters
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
OPTIMIZER_WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.2  # Warmup steps as a ratio of total training steps

# Special Tokens
PAD_TOKEN = 0
MASK_TOKEN = -1  # Will be set during preprocessing


def load_and_preprocess_data(data_path, min_interactions, rating_threshold, max_len):
    global MASK_TOKEN
    df = pd.read_csv(data_path, sep='::', header=None, engine='python',
                     names=['userId', 'movieId', 'rating', 'timestamp'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

    # Filter by rating threshold (keep highly rated items)
    df = df[df['rating'] >= rating_threshold].copy()

    # Map movieIds to contiguous integers (starting from 1, 0 is PAD)
    unique_movie_ids = df['movieId'].unique()
    item_map = {movie_id: idx + 1 for idx, movie_id in enumerate(unique_movie_ids)}
    num_items = len(item_map)
    MASK_TOKEN = num_items + 1
    vocab_size = num_items + 2  # +1 for PAD, +1 for MASK

    df['itemId'] = df['movieId'].map(item_map)
    df.sort_values(by=['userId', 'timestamp'], inplace=True)
    user_sequences = {}

    # Group by user and create sequences
    for user_id, group in df.groupby('userId'):
        items = group['itemId'].tolist()
        # Only keep users with sufficient interactions
        if len(items) >= min_interactions:
            user_sequences[user_id] = items

    # Enhanced processing of sequences for training
    train_sequences = {}
    valid_sequences = {}
    test_sequences = {}

    # Split users into train/valid/test
    user_ids = list(user_sequences.keys())
    random.shuffle(user_ids)

    num_test_users = int(len(user_ids) * TEST_SIZE)
    num_valid_users = int(len(user_ids) * VALID_SIZE)

    test_users = user_ids[:num_test_users]
    valid_users = user_ids[num_test_users:num_test_users + num_valid_users]
    train_users = user_ids[num_test_users + num_valid_users:]

    # Process training sequences - generate multiple sequences per user
    for user_id in train_users:
        seq = user_sequences[user_id]

        # For training, create multiple subsequences of different lengths
        if len(seq) <= max_len:
            # If sequence fits, just pad it
            padded_seq = [PAD_TOKEN] * (max_len - len(seq)) + seq
            train_sequences[f"{user_id}_full"] = padded_seq
        else:
            # Create the full sequence (latest max_len items)
            train_sequences[f"{user_id}_full"] = seq[-max_len:]

            # Create additional training samples with different windows
            stride = max(1, max_len // 4)  # Stride of 1/4 the max length
            for start_idx in range(0, len(seq) - max_len, stride):
                window = seq[start_idx:start_idx + max_len]
                if len(window) == max_len:  # Only use complete windows
                    train_sequences[f"{user_id}_{start_idx}"] = window

    # Process validation and test sequences
    for user_id in valid_users:
        seq = user_sequences[user_id]
        if len(seq) > max_len:
            valid_sequences[user_id] = seq[-max_len:]  # Take last max_len items
        else:
            # Pad shorter sequences
            valid_sequences[user_id] = [PAD_TOKEN] * (max_len - len(seq)) + seq

    for user_id in test_users:
        seq = user_sequences[user_id]
        if len(seq) > max_len:
            test_sequences[user_id] = seq[-max_len:]  # Take last max_len items
        else:
            # Pad shorter sequences
            test_sequences[user_id] = [PAD_TOKEN] * (max_len - len(seq)) + seq

    print(
        f"Split: Train={len(train_sequences)}, Validation={len(valid_sequences)}, Test={len(test_sequences)} sequences")

    return train_sequences, valid_sequences, test_sequences, vocab_size, num_items


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    train_sequences, valid_sequences, test_sequences, vocab_size, num_items = \
        load_and_preprocess_data(DATA_PATH, MIN_USER_INTERACTIONS, RATING_THRESHOLD, MAX_SEQ_LENGTH)

    # Create Enhanced Datasets and DataLoaders
    train_dataset = BERT4RecDataset(
        train_sequences, MAX_SEQ_LENGTH, MASK_PROB, MASK_TOKEN, num_items, mode='train'
    )
    valid_dataset = BERT4RecDataset(
        valid_sequences, MAX_SEQ_LENGTH, MASK_PROB, MASK_TOKEN, num_items, mode='valid'
    )
    test_dataset = BERT4RecDataset(
        test_sequences, MAX_SEQ_LENGTH, MASK_PROB, MASK_TOKEN, num_items, mode='test'
    )

    # Number of workers based on CPU cores
    num_workers = min(4, os.cpu_count() or 1)

    # Create DataLoaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type != 'cpu' else False,
        drop_last=True  # Drop last batch if incomplete
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type != 'cpu' else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type != 'cpu' else False
    )

    # 2. Initialize  Model
    model = BERT4Rec(
        vocab_size=vocab_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_TRANSFORMER_LAYERS,
        num_heads=NUM_ATTENTION_HEADS,
        max_len=MAX_SEQ_LENGTH,
        dropout=DROPOUT_PROB,
        num_items=num_items
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {num_params:,}")

    # 3. Train the Model
    model, history = train(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=OPTIMIZER_WEIGHT_DECAY,
        patience=EARLY_STOPPING_PATIENCE,
        device=device,
        num_items=num_items,
        warmup_ratio=WARMUP_RATIO
    )

    # 4. Evaluate on Test Set
    print("\nEvaluating final model on Test Set...")
    test_recall, test_ndcg = evaluate(model, test_loader, device, num_items, top_k=10)

    print("=" * 50)
    print(f"Final Test Results:")
    print(f"  Recall@10: {test_recall:.4f}")
    print(f"  NDCG@10:   {test_ndcg:.4f}")
    print("=" * 50)

    current_result = {
        'hidden_size': HIDDEN_SIZE,
        'masking_ratio': MASK_PROB,
        'layer_size': NUM_TRANSFORMER_LAYERS,
        'train_losses': history['train_loss'],
        'valid_ndcg': history['valid_ndcg'],
        'valid_recall': history['valid_recall'],
        'recall@10': test_recall,
        'ndcg@10': test_ndcg,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

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


if __name__ == "__main__":
    main()
