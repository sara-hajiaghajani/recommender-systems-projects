import torch
from tqdm import tqdm
import numpy as np


def evaluate_model(model, test_loader, k=10, device='mps'):
    model.eval()

    # Group test data by user
    user_items = {}
    user_preds = {}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            user_id = batch['user_id'].to(device)
            movie_id = batch['movie_id'].to(device)
            label = batch['label'].to(device)

            output = model(user_id, movie_id)

            # Store predictions and actual labels
            for i in range(len(user_id)):
                u = user_id[i].item()
                m = movie_id[i].item()
                pred = output[i].item()
                true = label[i].item()

                if u not in user_items:
                    user_items[u] = []
                    user_preds[u] = []

                user_items[u].append((m, true))
                user_preds[u].append((m, pred))

    # Compute Recall@k and NDCG@k
    recall_sum = 0.0
    ndcg_sum = 0.0
    num_users = 0

    for u in user_items:
        # Skip users with no positive items in test set
        if not any(true > 0 for _, true in user_items[u]):
            continue

        num_users += 1

        # Get positive items for the user
        pos_items = set([m for m, true in user_items[u] if true > 0])

        # Get top-k predicted items for the user
        pred_items = [m for m, _ in sorted(user_preds[u], key=lambda x: x[1], reverse=True)[:k]]

        # Compute Recall@k
        hits = len(set(pred_items) & pos_items)
        recall = hits / min(k, len(pos_items))
        recall_sum += recall

        # Compute NDCG@k
        dcg = 0.0
        idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(pos_items), k))])

        for i, item in enumerate(pred_items):
            if item in pos_items:
                dcg += 1.0 / np.log2(i + 2)

        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_sum += ndcg

    # Average metrics over all users
    recall_avg = recall_sum / num_users if num_users > 0 else 0
    ndcg_avg = ndcg_sum / num_users if num_users > 0 else 0

    return recall_avg, ndcg_avg
