import sys
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger

from libcity.models.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.models.loss import masked_mae_torch


try:
    from .STGformer import STGformer as STGformerBackbone
except ModuleNotFoundError as exc:
    if not exc.name or not exc.name.startswith('timm'):
        raise

    class _FallbackMlp(nn.Module):
        def __init__(self, in_features, hidden_features=None, act_layer=nn.ReLU, drop=0.0):
            super().__init__()
            hidden_features = hidden_features or in_features
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = act_layer() if isinstance(act_layer, type) else act_layer
            self.drop1 = nn.Dropout(drop)
            self.fc2 = nn.Linear(hidden_features, in_features)
            self.drop2 = nn.Dropout(drop)

        def forward(self, x):
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop1(x)
            x = self.fc2(x)
            x = self.drop2(x)
            return x

    timm_module = types.ModuleType('timm')
    timm_models_module = types.ModuleType('timm.models')
    timm_vision_transformer_module = types.ModuleType('timm.models.vision_transformer')
    timm_module.__path__ = []
    timm_models_module.__path__ = []
    timm_vision_transformer_module.Mlp = _FallbackMlp
    timm_models_module.vision_transformer = timm_vision_transformer_module
    timm_module.models = timm_models_module
    sys.modules.setdefault('timm', timm_module)
    sys.modules.setdefault('timm.models', timm_models_module)
    sys.modules.setdefault('timm.models.vision_transformer', timm_vision_transformer_module)

    from .STGformer import STGformer as STGformerBackbone


class STGformer(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.config = config
        self.data_feature = data_feature
        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')
        self.device = config.get('device', torch.device('cpu'))
        self._warned_missing_tod = False
        self._warned_missing_dow = False

        self.num_nodes = self.data_feature.get('num_nodes', config.get('num_nodes', 1))
        self.feature_dim = self.data_feature.get('feature_dim', config.get('input_dim', 1))
        self.output_dim = self.data_feature.get('output_dim', config.get('output_dim', 1))
        self.input_window = config.get('input_window', config.get('in_steps', 12))
        self.output_window = config.get('output_window', config.get('out_steps', 12))

        model_args = config.get('model_args', {})
        if not isinstance(model_args, dict):
            model_args = {}

        def _get(key, default):
            return config.get(key, model_args.get(key, default))

        self.input_dim = _get('input_dim', 1)
        self.steps_per_day = _get('steps_per_day', 288)
        self.input_embedding_dim = _get('input_embedding_dim', 24)
        self.tod_embedding_dim = _get('tod_embedding_dim', 0)
        self.dow_embedding_dim = _get('dow_embedding_dim', 0)
        self.spatial_embedding_dim = _get('spatial_embedding_dim', 0)
        self.adaptive_embedding_dim = _get('adaptive_embedding_dim', 12)
        self.num_heads = _get('num_heads', 4)
        self.num_layers = _get('num_layers', 3)
        self.dropout = _get('dropout', 0.1)
        self.mlp_ratio = _get('mlp_ratio', 2)
        self.use_mixed_proj = _get('use_mixed_proj', True)
        self.dropout_a = _get('dropout_a', 0.3)
        self.kernel_size = _get('kernel_size', [1])
        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size]

        adj_mx = self.data_feature.get('adj_mx')
        if adj_mx is None:
            adj_mx = np.eye(self.num_nodes, dtype=np.float32)
        supports = [torch.tensor(np.asarray(adj_mx), dtype=torch.float32, device=self.device)]

        self.backbone = STGformerBackbone(
            num_nodes=self.num_nodes,
            in_steps=self.input_window,
            out_steps=self.output_window,
            steps_per_day=self.steps_per_day,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            input_embedding_dim=self.input_embedding_dim,
            tod_embedding_dim=self.tod_embedding_dim,
            dow_embedding_dim=self.dow_embedding_dim,
            spatial_embedding_dim=self.spatial_embedding_dim,
            adaptive_embedding_dim=self.adaptive_embedding_dim,
            num_heads=self.num_heads,
            supports=supports,
            num_layers=self.num_layers,
            dropout=self.dropout,
            mlp_ratio=self.mlp_ratio,
            use_mixed_proj=self.use_mixed_proj,
            dropout_a=self.dropout_a,
            kernel_size=self.kernel_size,
        ).to(self.device)

    def forward(self, batch):
        x = batch['X'].to(self.device)
        batch_size = x.shape[0]
        feature_dim = x.shape[-1]

        x = x[..., :self.input_dim]
        x = self.backbone.input_proj(x)

        features = []
        if self.tod_embedding_dim > 0:
            tod_index = self.input_dim
            if feature_dim <= tod_index:
                if not self._warned_missing_tod:
                    self._logger.warning('Missing time-of-day channel, fallback to zero TOD embedding.')
                    self._warned_missing_tod = True
                tod_emb = torch.zeros(
                    batch_size,
                    self.input_window,
                    self.num_nodes,
                    self.tod_embedding_dim,
                    device=self.device,
                    dtype=x.dtype,
                )
            else:
                tod = batch['X'].to(self.device)[..., tod_index]
                tod_index_tensor = (tod * self.steps_per_day).long().clamp(min=0, max=self.steps_per_day - 1)
                tod_emb = self.backbone.tod_embedding(tod_index_tensor)
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_index = self.input_dim + (1 if self.tod_embedding_dim > 0 else 0)
            if feature_dim <= dow_index:
                if not self._warned_missing_dow:
                    self._logger.warning('Missing day-of-week channel, fallback to zero DOW embedding.')
                    self._warned_missing_dow = True
                dow_emb = torch.zeros(
                    batch_size,
                    self.input_window,
                    self.num_nodes,
                    self.dow_embedding_dim,
                    device=self.device,
                    dtype=x.dtype,
                )
            else:
                dow_raw = batch['X'].to(self.device)[..., dow_index:]
                if dow_raw.shape[-1] >= 7:
                    dow = dow_raw[..., :7].argmax(dim=-1)
                else:
                    dow = dow_raw[..., 0].long().clamp(min=0, max=6)
                dow_emb = self.backbone.dow_embedding(dow.long())
            features.append(dow_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.backbone.adaptive_embedding.expand(
                size=(batch_size, *self.backbone.adaptive_embedding.shape)
            )
            features.append(self.backbone.dropout(adp_emb))

        if features:
            x = torch.cat([x] + features, dim=-1)

        x = self.backbone.temporal_proj(x.transpose(1, 3)).transpose(1, 3)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        graph = torch.matmul(
            self.backbone.adaptive_embedding,
            self.backbone.adaptive_embedding.transpose(1, 2),
        )
        graph = self.backbone.pooling(graph.transpose(0, 2)).transpose(0, 2)
        graph = F.softmax(F.relu(graph), dim=-1)
        for attn in self.backbone.attn_layers_s:
            x = attn(x, graph)
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = self.backbone.encoder_proj(x.transpose(1, 2).flatten(-2))
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        for layer in self.backbone.encoder:
            x = x + layer(x)
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        out = self.backbone.output_proj(x).view(
            batch_size, self.num_nodes, self.output_window, self.output_dim
        )
        out = out.transpose(1, 2)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def predict(self, batch):
        return self.forward(batch)

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return masked_mae_torch(y_predicted, y_true)
