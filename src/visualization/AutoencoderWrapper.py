from sklearn.base import BaseEstimator, RegressorMixin
from src.baselines import GCNMae
from torch.optim import Adam
import torch
import torch.nn.functional as F


class AutoencoderWrapper(RegressorMixin, BaseEstimator):
    def __init__(
            self,
            edge_index,
            edge_weight,
            node_feature_dim: int = 1,
            num_nodes: int = 68,
            encoder_embed_dim: int = 64,
            latent_dim: int = 8,
            encoder_depth=10,
            pool_ratio=0.7,
            lr=5e-3,
            gcn_type='gcn',
            epochs=100,
            decoder_norm=False,
    ):
        self.node_feature_dim = node_feature_dim
        self.num_nodes = num_nodes
        self.encoder_embed_dim = encoder_embed_dim
        self.latent_dim = latent_dim
        self.encoder_depth = encoder_depth
        self.pool_ratio = pool_ratio
        self.gcn_type = gcn_type
        self.epochs = epochs
        self.lr = lr
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.decoder_norm = decoder_norm

    def fit(self, X, y=None):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = GCNMae(
            node_feature_dim=self.node_feature_dim,
            num_nodes=self.num_nodes,
            encoder_embed_dim=self.encoder_embed_dim,
            latent_dim=self.latent_dim,
            encoder_depth=self.encoder_depth,
            pool_ratio=self.pool_ratio,
            gcn_type=self.gcn_type,
            decoder_norm=self.decoder_norm
        ).to(device)

        self.optimizer_ = Adam(self.model_.parameters(), lr=self.lr, weight_decay=1e-5)

        self.model_.train()
        for epoch in range(self.epochs):
            data = self.prepare_data(X)
            N, T, V, D = data['x'].shape
            target = data['x'].clone().view(N, V, D)

            batch_size = min(32, N // 4)

            indices = torch.randperm(N).to(device)
            for i in range(0, N, batch_size):
                batch_idx = indices[i:i + batch_size]
                batch_data = {k: v[batch_idx] for k, v in data.items()}
                batch_target = target[batch_idx]

                self.optimizer_.zero_grad()
                x_recon = self.model_(batch_data)
                loss = F.mse_loss(x_recon, batch_target)
                loss.backward()
                self.optimizer_.step()

            # Optionally print gradient norms
            # if (epoch % 50 == 0 or epoch == self.epochs - 1):
            #     for name, param in self.model_.named_parameters():
            #         if param.grad is not None:
            #             print(f"Gradient norm for {name}: {param.grad.norm().item()}")

            if (epoch == self.epochs - 1):
                print(f"  > Epoch {epoch + 1}/{self.epochs} - Loss: {loss.item():.4f}")
        print(self.model_.encoder_embed_dim, self.model_.latent_dim, self.model_.encoder_depth)
        print(self.lr)
        return self

    def transform(self, X):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model_.to(device)

        data = self.prepare_data(X)
        N, T, V, D = data['x'].shape
        self.model_.eval()  # assuming you instantiated self.model_ in fit()

        with torch.no_grad():
            latent = self.model_.encoder(data)
        return latent.detach().cpu()

    def predict(self, X):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model_.to(device)

        data = self.prepare_data(X)
        self.model_.eval()

        with torch.no_grad():
            x_recon = self.model_.encoder(data)
        return x_recon.detach().cpu()

    def score(self, X, y):
        print('using MSE loss for scoring')
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model_.to(device)

        data = self.prepare_data(X)
        N, T, V, D = data['x'].shape
        target = data['x'].clone().view(N, V, D)

        x_recon = self.predict(X)
        loss = F.mse_loss(x_recon, target)
        return -loss.item()

    def prepare_data(self, X):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X).to(device)
        else:
            X = X.to(device)
        if not isinstance(self.edge_index, torch.Tensor):
            edge_index = torch.tensor(self.edge_index).to(device)
        else:
            edge_index = self.edge_index.to(device)
        if not isinstance(self.edge_weight, torch.Tensor):
            edge_weight = torch.tensor(self.edge_weight).to(device)
        else:
            edge_weight = self.edge_weight.to(device)

        N = X.shape[0]
        X = X.view(N, 1, self.num_nodes, -1)
        edge_index = edge_index.repeat(N, 1, 1)
        edge_weight = edge_weight.repeat(N, 1)
        data = {
            'x': X,
            'edge_index': edge_index,
            'edge_attr': edge_weight
        }
        return data
