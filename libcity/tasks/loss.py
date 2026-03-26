"""Loss helpers used by models/executors/evaluators.

This module provides a compact, tolerant implementation of common loss
functions. The function signatures accept the legacy parameters used across
the codebase (including a `mask_val` kwarg) but ignore unused options.
"""
import torch
import torch.nn.functional as F
import numpy as np


def _mask_null(tensor, null_val=None):
    if null_val is None:
        return ~torch.isnan(tensor)
    if isinstance(null_val, float) and np.isnan(null_val):
        return ~torch.isnan(tensor)
    return tensor != null_val


def _apply_mask(y_pred, y_true, null_val=None):
    mask = _mask_null(y_true, null_val)
    y_pred = y_pred[mask]
    y_true = y_true[mask]
    return y_pred, y_true


def masked_mae_torch(y_pred, y_true, null_val=None, mask_val=None):
    y_pred, y_true = _apply_mask(y_pred, y_true, null_val)
    if y_true.numel() == 0:
        return torch.tensor(0., device=y_pred.device)
    return torch.mean(torch.abs(y_pred - y_true))


def masked_mse_torch(y_pred, y_true, null_val=None, mask_val=None):
    y_pred, y_true = _apply_mask(y_pred, y_true, null_val)
    if y_true.numel() == 0:
        return torch.tensor(0., device=y_pred.device)
    return torch.mean((y_pred - y_true) ** 2)


def masked_rmse_torch(y_pred, y_true, null_val=None, mask_val=None):
    return torch.sqrt(masked_mse_torch(y_pred, y_true, null_val, mask_val))


def masked_mape_torch(y_pred, y_true, null_val=None, mask_val=None):
    y_pred, y_true = _apply_mask(y_pred, y_true, null_val)
    if y_true.numel() == 0:
        return torch.tensor(0., device=y_pred.device)
    denom = torch.clamp(torch.abs(y_true), min=1e-6)
    return torch.mean(torch.abs((y_pred - y_true) / denom))


def log_cosh_loss(y_pred, y_true, null_val=None, mask_val=None):
    y_pred, y_true = _apply_mask(y_pred, y_true, null_val)
    x = y_pred - y_true
    return torch.mean(torch.log(torch.cosh(x + 1e-12)))


def huber_loss(y_pred, y_true, delta=1.0, null_val=None, mask_val=None):
    y_pred, y_true = _apply_mask(y_pred, y_true, null_val)
    return F.smooth_l1_loss(y_pred, y_true, reduction='mean')


def quantile_loss(y_pred, y_true, q=0.5, null_val=None, mask_val=None):
    y_pred, y_true = _apply_mask(y_pred, y_true, null_val)
    err = y_true - y_pred
    return torch.mean(torch.max((q - 1) * err, q * err))


def r2_score_torch(y_pred, y_true, null_val=None, mask_val=None):
    y_pred, y_true = _apply_mask(y_pred, y_true, null_val)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-12)


def explained_variance_score_torch(y_pred, y_true, null_val=None, mask_val=None):
    y_pred, y_true = _apply_mask(y_pred, y_true, null_val)
    var_diff = torch.var(y_true - y_pred)
    var_true = torch.var(y_true)
    return 1 - var_diff / (var_true + 1e-12)


__all__ = [
    'masked_mae_torch', 'masked_mse_torch', 'masked_rmse_torch', 'masked_mape_torch',
    'log_cosh_loss', 'huber_loss', 'quantile_loss', 'r2_score_torch', 'explained_variance_score_torch'
]
"""Basic loss helpers used by models and executors.

This is a lightweight replacement for the legacy `libcity.model.loss` module
to keep imports working under the new layout. Implementations are simple
PyTorch-based utilities sufficient for import-time usage and tests.
"""
import torch
import torch.nn.functional as F
import numpy as np


def _mask_null(tensor, null_val=None):
    if null_val is None:
        return torch.ones_like(tensor, dtype=torch.bool)
    if isinstance(null_val, float) and np.isnan(null_val):
        return ~torch.isnan(tensor)
    return tensor != null_val


def masked_mae_torch(y_pred, y_true, null_val=None):
    mask = _mask_null(y_true, null_val).float()
    diff = torch.abs(y_pred - y_true) * mask
    return diff.sum() / (mask.sum() + 1e-8)


def masked_mse_torch(y_pred, y_true, null_val=None):
    mask = _mask_null(y_true, null_val).float()
    diff = (y_pred - y_true) ** 2 * mask
    return diff.sum() / (mask.sum() + 1e-8)


def masked_rmse_torch(y_pred, y_true, null_val=None):
    return torch.sqrt(masked_mse_torch(y_pred, y_true, null_val))


def masked_mape_torch(y_pred, y_true, null_val=None):
    mask = _mask_null(y_true, null_val).float()
    denom = torch.abs(y_true)
    denom = torch.where(denom == 0, torch.ones_like(denom) * 1e-6, denom)
    diff = torch.abs((y_pred - y_true) / denom) * mask
    return diff.sum() / (mask.sum() + 1e-8)


def log_cosh_loss(y_pred, y_true):
    x = y_pred - y_true
    return torch.mean(x + torch.nn.functional.softplus(-2.0 * x) - np.log(2.0))


def huber_loss(y_pred, y_true, delta=1.0):
    x = y_pred - y_true
    return F.smooth_l1_loss(y_pred, y_true, reduction='mean')


def quantile_loss(y_pred, y_true, q=0.5):
    err = y_true - y_pred
    return torch.max(q * err, (q - 1) * err).mean()


def r2_score_torch(y_pred, y_true):
    var_y = torch.var(y_true)
    if var_y == 0:
        return torch.tensor(0.0)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-8)


def explained_variance_score_torch(y_pred, y_true):
    var_res = torch.var(y_true - y_pred)
    var_true = torch.var(y_true)
    return 1 - var_res / (var_true + 1e-8)


__all__ = [
    'masked_mae_torch',
    'masked_mse_torch',
    'masked_rmse_torch',
    'masked_mape_torch',
    'log_cosh_loss',
    'huber_loss',
    'quantile_loss',
    'r2_score_torch',
    'explained_variance_score_torch',
]
