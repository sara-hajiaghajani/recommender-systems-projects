import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from evaluation import evaluate

PAD_TOKEN = 0
MASK_TOKEN = -1  # Will be set during preprocessing

TEST_SIZE = 0.10
VALID_SIZE = 0.10


def train(model, train_loader, valid_loader, epochs, learning_rate, weight_decay,
                   patience, device, num_items, warmup_ratio=0.1):
    print(f"Starting training on {device}...")
    model.to(device)

    # Initialize optimizer with weight decay
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Learning rate scheduler with warmup and cosine decay
    total_steps = epochs * len(train_loader)
    warmup_steps = int(warmup_ratio * total_steps)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine annealing after warmup
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))  # min LR = 5% of max

    scheduler = LambdaLR(optimizer, lr_lambda)

    # Loss function with label smoothing for better regularization
    label_smoothing = 0.1  # Small smoothing factor
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN, label_smoothing=label_smoothing)

    # Setup for mixed precision training if available (speeds up training on CUDA)
    scaler = torch.cuda.amp.GradScaler() if (device.type == 'cuda') else None
    use_amp = (device.type == 'cuda')

    best_val_ndcg = -1.0
    patience_counter = 0
    best_model_state = None
    best_epoch = -1

    # Keep track of metrics for plotting
    train_losses = []
    valid_recalls = []
    valid_ndcgs = []

    # Training loop
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0

        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")

        for batch_idx, batch in enumerate(train_iterator):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # FP16 mixed precision training
            if use_amp:
                with torch.cuda.amp.autocast():
                    # Forward pass
                    logits = model(input_ids, attention_mask)
                    logits_flat = logits.view(-1, logits.size(-1))
                    labels_flat = labels.view(-1)
                    loss = criterion(logits_flat, labels_flat)

                # Mixed precision backward pass
                optimizer.zero_grad()
                scaler.scale(loss).backward()

                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Update with scaler
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision training
                logits = model(input_ids, attention_mask)
                logits_flat = logits.view(-1, logits.size(-1))
                labels_flat = labels.view(-1)
                loss = criterion(logits_flat, labels_flat)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Update learning rate
            scheduler.step()

            # Update metrics
            batch_count += 1
            total_loss += loss.item()
            current_lr = scheduler.get_last_lr()[0]
            train_iterator.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / batch_count:.4f}",
                'lr': f"{current_lr:.6f}"
            })

        avg_train_loss = total_loss / batch_count
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1} Average Training Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_recall, val_ndcg = evaluate(model, valid_loader, device, num_items, top_k=10)
        valid_recalls.append(val_recall)
        valid_ndcgs.append(val_ndcg)

        print(f"Epoch {epoch + 1} Validation: Recall@10={val_recall:.4f}, NDCG@10={val_ndcg:.4f}")

        # Early Stopping Logic
        if not np.isnan(val_ndcg) and val_ndcg > best_val_ndcg:
            best_val_ndcg = val_ndcg
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"New best validation NDCG@10: {best_val_ndcg:.4f}. Saving model...")
        else:
            patience_counter += 1
            status = "NaN" if np.isnan(val_ndcg) else "did not improve"
            print(f"Validation NDCG@10 {status}. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.2f} seconds.")

    if best_model_state:
        print(f"Best model from Epoch {best_epoch} with Validation NDCG@10: {best_val_ndcg:.4f}")
        model.load_state_dict(best_model_state)
    else:
        print("Warning: No best model state saved. Using final epoch model.")

    # Return training history along with the model
    history = {
        'train_loss': train_losses,
        'valid_recall': valid_recalls,
        'valid_ndcg': valid_ndcgs,
        'best_epoch': best_epoch,
        'best_ndcg': best_val_ndcg
    }

    return model, history