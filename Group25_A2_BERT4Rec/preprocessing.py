import random

import numpy as np
import torch
from torch.utils.data import Dataset

PAD_TOKEN = 0

class BERT4RecDataset(Dataset):
    def __init__(self, sequences_dict, max_len, mask_prob, mask_token_id, num_items, mode='train'):
        self.user_ids = list(sequences_dict.keys())
        self.sequences = list(sequences_dict.values())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self.num_items = num_items
        self.mode = mode

        if self.mask_token_id == -1:
            raise ValueError("MASK_TOKEN ID was not set during preprocessing.")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        masked_seq = list(seq)  # Make a copy
        labels = [PAD_TOKEN] * self.max_len  # Labels for masked positions

        if self.mode == 'train':
            # Enhanced masking strategy for training
            non_pad_indices = [i for i, item in enumerate(seq) if item != PAD_TOKEN]

            if not non_pad_indices:  # Skip if sequence is all padding
                return {
                    'input_ids': torch.tensor(masked_seq, dtype=torch.long),
                    'attention_mask': torch.tensor([0] * self.max_len, dtype=torch.long),
                    'labels': torch.tensor(labels, dtype=torch.long)
                }

            # Calculate number of tokens to mask (more strategic masking)
            # Ensure we mask at least one token, but not more than mask_prob allows
            num_to_mask = max(1, int(len(non_pad_indices) * self.mask_prob))

            # Prioritize masking the later items (more relevant for recommendation)
            # Sort indices and bias toward more recent items
            non_pad_indices.sort()  # Ensure they're in order
            mask_weights = np.linspace(0.5, 1.0, len(non_pad_indices))  # Higher weight for later items
            mask_weights = mask_weights / mask_weights.sum()  # Normalize to probabilities

            # Sample indices to mask based on weights
            indices_to_mask = np.random.choice(
                non_pad_indices,
                size=min(num_to_mask, len(non_pad_indices)),
                replace=False,
                p=mask_weights
            )

            # Apply masking
            for i in indices_to_mask:
                labels[i] = seq[i]  # Set label to the original item
                mask_decision = random.random()

                if mask_decision < 0.8:  # 80% replace with [MASK]
                    masked_seq[i] = self.mask_token_id
                elif mask_decision < 0.9:  # 10% replace with random item
                    random_item_id = random.randint(1, self.num_items)
                    masked_seq[i] = random_item_id
                # 10% keep original item (no change)

        elif self.mode in ['valid', 'test']:
            # For validation/test, mask the last item
            non_pad_indices = [i for i, item in enumerate(seq) if item != PAD_TOKEN]

            if non_pad_indices:
                # Mask the last non-padding item
                last_item_idx = non_pad_indices[-1]
                labels[last_item_idx] = seq[last_item_idx]
                masked_seq[last_item_idx] = self.mask_token_id

        # Create attention mask (1 for non-PAD tokens, 0 for PAD)
        attention_mask = [1 if token != PAD_TOKEN else 0 for token in seq]

        return {
            'input_ids': torch.tensor(masked_seq, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
