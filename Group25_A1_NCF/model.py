import torch
import torch.nn as nn


class NCF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=64, mlp_layers=[128, 64, 32]):
        super(NCF, self).__init__()

        # Embedding layers
        self.user_embedding_gmf = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(n_items, embedding_dim)
        self.user_embedding_mlp = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(n_items, embedding_dim)

        # GMF part (just element-wise product of embeddings)
        # MLP part
        self.mlp_layers = nn.ModuleList()
        layer_dims = [2 * embedding_dim] + mlp_layers

        for i in range(len(layer_dims) - 1):
            self.mlp_layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.BatchNorm1d(layer_dims[i + 1]))
            self.mlp_layers.append(nn.Dropout(p=0.2))

        # Fusion layer
        self.fusion = nn.Linear(embedding_dim + layer_dims[-1], 32)
        self.fusion_activation = nn.ReLU()

        # Prediction layer
        self.prediction = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, user_indices, item_indices):
        # GMF part
        user_embedding_gmf = self.user_embedding_gmf(user_indices)
        item_embedding_gmf = self.item_embedding_gmf(item_indices)
        gmf_output = user_embedding_gmf * item_embedding_gmf

        # MLP part
        user_embedding_mlp = self.user_embedding_mlp(user_indices)
        item_embedding_mlp = self.item_embedding_mlp(item_indices)
        mlp_input = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=1)

        # Apply MLP layers
        mlp_output = mlp_input
        for layer in self.mlp_layers:
            mlp_output = layer(mlp_output)

        # Fusion
        fusion_input = torch.cat([gmf_output, mlp_output], dim=1)
        fusion_output = self.fusion(fusion_input)
        fusion_output = self.fusion_activation(fusion_output)

        # Prediction
        prediction = self.prediction(fusion_output)
        output = self.sigmoid(prediction)

        return output.view(-1)
