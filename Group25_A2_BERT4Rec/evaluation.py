import torch
from tqdm import tqdm
import math

PAD_TOKEN = 0
MASK_TOKEN = -1

TEST_SIZE = 0.15
VALID_SIZE = 0.15


def evaluate(model, data_loader, device, num_items, top_k=10):

    model.eval()
    total_recall = 0.0
    total_ndcg = 0.0
    total_samples = 0

    eval_iterator = tqdm(data_loader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for batch in eval_iterator:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            logits = model(input_ids, attention_mask)

            # Process each sequence in the batch
            for i in range(input_ids.size(0)):
                # Find positions where we have an actual label (not PAD_TOKEN)
                label_positions = (labels[i] != PAD_TOKEN).nonzero(as_tuple=True)[0]

                if len(label_positions) == 0:
                    continue  # Skip if no labels to predict

                # For each valid label position
                for pos in label_positions:
                    # Get predicted scores for this position (all items)
                    pos_logits = logits[i, pos, 1:]  # Skip PAD_TOKEN scores

                    # Get true item for this position
                    true_item = labels[i, pos].item()

                    # Skip if target is PAD_TOKEN (shouldn't happen due to our filter above)
                    if true_item == PAD_TOKEN:
                        continue

                    # Adjust true_item to match our logits indexing (offset by 1)
                    true_item_idx = true_item - 1

                    # Calculate top-k items
                    _, topk_indices = torch.topk(pos_logits, k=top_k)

                    # Calculate Recall@k (1 if true item in top-k, 0 otherwise)
                    hit = (topk_indices == true_item_idx).any().item()
                    recall = float(hit)

                    # Calculate NDCG@k
                    # Find position of target item in the sorted list
                    rank = (topk_indices == true_item_idx).nonzero()
                    if len(rank) > 0:
                        # Item is in top-k, calculate NDCG
                        rank = rank.item()
                        ndcg = 1.0 / math.log2(rank + 2)  # +2 because rank is 0-indexed
                    else:
                        # Item not in top-k
                        ndcg = 0.0

                    # Update totals
                    total_recall += recall
                    total_ndcg += ndcg
                    total_samples += 1

            # Update progress bar
            if total_samples > 0:
                eval_iterator.set_postfix({
                    'Recall': f"{total_recall / total_samples:.4f}",
                    'NDCG': f"{total_ndcg / total_samples:.4f}"
                })

    # Calculate final metrics
    avg_recall = total_recall / total_samples if total_samples > 0 else 0.0
    avg_ndcg = total_ndcg / total_samples if total_samples > 0 else 0.0

    return avg_recall, avg_ndcg
