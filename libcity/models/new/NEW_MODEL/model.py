import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger

from libcity.models.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.models import loss


class DynamicGraphConstructor(nn.Module):
    """
    Dynamic Graph Constructor - learns similarity matrix based on node embeddings
    Implements dynamic graph convolution, referenced from DGCRAN and ADSTGCN
    """
    
    def __init__(self, num_nodes, embed_dim, feature_dim, cheb_k=3):
        super(DynamicGraphConstructor, self).__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        self.cheb_k = cheb_k
        
        # Node embeddings for computing similarity
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
        
        # Graph convolution weights
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, feature_dim, feature_dim))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, feature_dim))
        
        self._init_parameters()
    
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
    
    def forward(self, x):
        """
        Args:
            x: [B, N, D] - input features
        Returns:
            dynamic_adj: [B, N, N] - dynamic adjacency matrix
            graph_features: [B, N, D] - graph convolution output
        """
        batch_size = x.shape[0]
        
        # Compute dynamic similarity matrix (dynamic graph construction)
        # Use node embeddings to compute similarity
        node_emb = self.node_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, E]
        similarity = F.softmax(F.relu(torch.bmm(node_emb, node_emb.transpose(1, 2))), dim=-1)  # [B, N, N]
        
        # Build Chebyshev polynomial basis
        support_set = [torch.eye(self.num_nodes).to(x.device).unsqueeze(0).expand(batch_size, -1, -1), similarity]
        for k in range(2, self.cheb_k):
            support_set.append(torch.bmm(2 * similarity, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=1)  # [B, cheb_k, N, N]
        
        # Dynamic graph convolution computation
        weights = torch.einsum('bne,ekio->bnkio', node_emb, self.weights_pool)  # [B, N, cheb_k, D, D]
        bias = torch.einsum('bne,do->bno', node_emb, self.bias_pool)  # [B, N, D]
        
        x_g = torch.einsum("bknm,bmc->bknc", supports, x)  # [B, cheb_k, N, D]
        x_g = x_g.permute(0, 2, 1, 3)  # [B, N, cheb_k, D]
        
        graph_features = torch.einsum('bnki,bnkio->bno', x_g, weights) + bias  # [B, N, D]
        
        return similarity, graph_features


class TimeTransformer(nn.Module):
    """
    Time module - multi-head self-attention mechanism
    Makes up for RNN's insufficient long-range dependency modeling
    """
    
    def __init__(self, d_model, num_heads, num_layers, dropout=0.1):
        super(TimeTransformer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [B, T, N, D] - time series input
            mask: attention mask
        Returns:
            output: [B, T, N, D] - time features
        """
        batch_size, seq_len, num_nodes, d_model = x.shape
        
        # Reshape to [B*N, T, D] for Transformer
        x = x.reshape(batch_size * num_nodes, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoding
        output = self.transformer_encoder(x, mask=mask)
        
        # Reshape back to original dimension [B, T, N, D]
        output = output.reshape(batch_size, seq_len, num_nodes, d_model)
        
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ContextRefinement(nn.Module):
    """
    Context refinement module - ensemble learning/MLP
    Combines external variables (weather, holidays, etc.) for residual correction
    """
    
    def __init__(self, d_model, external_dim, hidden_dim=128, dropout=0.1):
        super(ContextRefinement, self).__init__()
        
        # Spatio-temporal feature processing
        self.spatio_temporal_fc = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # External variable processing
        self.external_fc = nn.Sequential(
            nn.Linear(external_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Feature fusion and residual correction
        self.fusion_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model)
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, spatio_temporal_features, external_features):
        """
        Args:
            spatio_temporal_features: [B, T, N, D] - spatio-temporal features
            external_features: [B, E] - external variables
        Returns:
            refined_features: [B, T, N, D] - refined features
        """
        batch_size, seq_len, num_nodes, d_model = spatio_temporal_features.shape
        
        # Process spatio-temporal features
        st_features = self.spatio_temporal_fc(spatio_temporal_features)  # [B, T, N, H]
        
        # Process external features and expand dimensions
        ext_features = self.external_fc(external_features)  # [B, H]
        ext_features = ext_features.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, H]
        ext_features = ext_features.expand(-1, seq_len, num_nodes, -1)  # [B, T, N, H]
        
        # Feature fusion
        combined_features = torch.cat([st_features, ext_features], dim=-1)  # [B, T, N, 2H]
        residual_correction = self.fusion_fc(combined_features)  # [B, T, N, D]
        
        # Residual connection and layer normalization
        refined_features = self.layer_norm(spatio_temporal_features + residual_correction)
        
        return refined_features


class NEW_MODEL(AbstractTrafficStateModel):
    """
    New spatio-temporal traffic prediction model
    Uses three-stage architecture: dynamic GNN + Transformer + ensemble learning
    """
    
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        
        # Basic configuration
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 1)
        self.output_dim = data_feature.get('output_dim', 1)
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.device = config.get('device', torch.device('cpu'))
        
        # Model hyperparameters
        self.d_model = config.get('d_model', 64)
        self.embed_dim = config.get('embed_dim', 10)
        self.num_heads = config.get('num_heads', 4)
        self.num_layers = config.get('num_layers', 2)
        self.cheb_k = config.get('cheb_order', 3)
        self.dropout = config.get('dropout', 0.1)
        self.external_dim = config.get('external_dim', 4)  # External variable dimension
        
        # Feature projection
        self.input_proj = nn.Linear(self.feature_dim, self.d_model)
        
        # 1. Spatial module - dynamic graph convolution
        self.dynamic_gnn = DynamicGraphConstructor(
            num_nodes=self.num_nodes,
            embed_dim=self.embed_dim,
            feature_dim=self.d_model,
            cheb_k=self.cheb_k
        )
        
        # 2. Time module - Transformer
        self.time_transformer = TimeTransformer(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        
        # 3. Context refinement module
        self.context_refiner = ContextRefinement(
            d_model=self.d_model,
            external_dim=self.external_dim,
            dropout=self.dropout
        )
        
        # Output layer
        self.output_fc = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model * 2, self.output_dim)
        )
        
        # Prediction head (for multi-step prediction)
        self.predictor = nn.Conv2d(
            in_channels=1,
            out_channels=self.output_window * self.output_dim,
            kernel_size=(1, self.d_model),
            bias=True
        )
        
        # Other settings
        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')
        self._init_parameters()
    
    def _init_parameters(self):
        """Parameter initialization"""
        for name, param in self.named_parameters():
            if 'node_embeddings' not in name:  # Skip already initialized node embeddings
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param)
    
    def forward(self, batch):
        """
        Forward pass - three-stage feature extraction process
        
        Args:
            batch: dictionary containing input data
                - 'X': [B, T_in, N, F] - historical observation data
                - 'y': [B, T_out, N, F] - target data
                - 'ext': [B, E] - external variables
        
        Returns:
            predictions: [B, T_out, N, output_dim]
        """
        # Input data
        source = batch['X']  # [B, T_in, N, F]
        external_features = None
        if hasattr(batch, 'feature_name') and 'ext' in batch.feature_name:
            external_features = batch['ext']  # [B, E]
        
        batch_size, seq_len, num_nodes, feature_dim = source.shape
        
        # Feature projection
        x = self.input_proj(source)  # [B, T_in, N, D]
        
        # Stage 1: Spatial feature extraction (dynamic graph convolution)
        spatial_features = []
        dynamic_adjs = []
        
        for t in range(seq_len):
            time_slice = x[:, t, :, :]  # [B, N, D]
            adj, sp_features = self.dynamic_gnn(time_slice)
            dynamic_adjs.append(adj)
            spatial_features.append(sp_features)
        
        # Merge spatio-temporal features [B, T, N, D]
        spatial_features = torch.stack(spatial_features, dim=1)
        
        # Stage 2: Time feature extraction (Transformer)
        temporal_features = self.time_transformer(spatial_features)  # [B, T, N, D]
        
        # Stage 3: Context refinement (combine external variables)
        if external_features is not None:
            refined_features = self.context_refiner(temporal_features, external_features)
        else:
            # If no external features, use zero tensor
            dummy_external = torch.zeros(batch_size, self.external_dim).to(self.device)
            refined_features = self.context_refiner(temporal_features, dummy_external)
        
        # Prediction generation
        # Take the last time step's features as prediction basis
        last_features = refined_features[:, -1:, :, :]  # [B, 1, N, D]
        
        # Use convolution for multi-step prediction
        predictions = self.predictor(last_features)  # [B, T_out*C, N, 1]
        predictions = predictions.squeeze(-1)  # [B, T_out*C, N]
        predictions = predictions.reshape(batch_size, self.output_window, self.output_dim, num_nodes)
        predictions = predictions.permute(0, 1, 3, 2)  # [B, T_out, N, C]
        
        return predictions
    
    def predict(self, batch):
        """
        Prediction interface
        """
        return self.forward(batch)
    
    def calculate_loss(self, batch):
        """
        Loss calculation - using Masked MAE
        """
        y_true = batch['y']
        y_predicted = self.predict(batch)
        
        # Inverse transform
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        
        # Calculate MAE loss
        return loss.masked_mae_torch(y_predicted, y_true, 0)
