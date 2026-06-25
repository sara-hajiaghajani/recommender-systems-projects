import torch
from torch import nn

PAD_TOKEN = 0

class BERT4Rec(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, max_len, dropout, num_items):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.num_items = num_items

        # Item Embeddings
        self.item_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD_TOKEN)
        # Position Embeddings
        self.position_embedding = nn.Embedding(max_len, hidden_size)

        # Embedding Dropout and Layer Normalization
        self.emb_dropout = nn.Dropout(dropout)
        self.emb_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

        # Enhanced Transformer Encoder with GELU and Normalization
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Apply normalization first (Pre-LN architecture)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output projection
        self.output_layer = nn.Linear(hidden_size, num_items + 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Xavier initialization for embeddings
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # Initialize positional embeddings
        nn.init.xavier_uniform_(self.position_embedding.weight)

        # Initialize output layer with smaller weights
        nn.init.xavier_uniform_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

    def forward(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.size()

        # Item Embeddings
        item_emb = self.item_embedding(input_ids)  # (batch, seq_len, hidden)

        # Position Embeddings
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        pos_emb = self.position_embedding(position_ids)

        # Combine embeddings
        embeddings = item_emb + pos_emb
        embeddings = self.emb_layer_norm(embeddings)
        embeddings = self.emb_dropout(embeddings)

        # Create attention mask for transformer
        # Convert attention_mask from (batch, seq_len) to (batch, 1, seq_len)
        # where 1 indicates valid positions and 0 indicates padding
        attn_mask = (attention_mask == 0)  # True for padding positions

        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(
            src=embeddings,
            src_key_padding_mask=attn_mask
        )

        # Output projections
        logits = self.output_layer(transformer_output)

        return logits
