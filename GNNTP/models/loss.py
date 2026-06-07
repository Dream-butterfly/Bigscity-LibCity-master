"""
Loss functions for traffic prediction models.

Design principles (v2):
- Mask is binary (0/1), never self-normalized — avoids gradient scale drift.
- Mean is computed as sum(loss * mask) / (mask.sum() + eps) — directly over
  valid samples only, batch-to-batch comparable.
- NaN is excluded via mask, never replaced with zero.
- R² / EVAR computed on-GPU to avoid CPU pipeline stall.
"""

import torch
import numpy as np


# ── Core mask helper ──────────────────────────────────────────────

def _build_valid_mask(labels, null_val):
    """Build a binary validity mask from labels.

    Returns a float tensor of same shape as labels: 1.0 = valid, 0.0 = invalid.
    """
    labels = labels.clone()
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = (~torch.isnan(labels)).float()
    else:
        mask = labels.ne(null_val).float()
    return mask


def _reduce_masked(loss, mask, eps=1e-8):
    """Reduce a masked loss tensor to a scalar: mean over valid samples only.

    Args:
        loss:  per-element loss tensor.
        mask:  binary float mask (same shape as loss). 1=valid, 0=invalid.
        eps:   small constant to avoid division by zero when mask is all-zero.
    """
    s = mask.sum()
    if s == 0:
        return loss.new_tensor(0.0)
    return (loss * mask).sum() / (s + eps)


# ── Torch losses ──────────────────────────────────────────────────

def masked_mae_loss(y_pred, y_true):
    """MAE loss for direct use in model.calculate_loss()."""
    mask = (y_true != 0).float()
    loss = torch.abs(y_pred - y_true)
    return _reduce_masked(loss, mask)


def masked_mae_torch(preds, labels, null_val=np.nan, reduce=True, mask_val=None):
    """Masked MAE (torch).

    Args:
        reduce:  if True, returns a scalar (mean over valid samples).
                 if False, returns per-element masked loss tensor.
    """
    mask = _build_valid_mask(labels, null_val)
    if mask_val is not None:
        mask = mask * labels.ge(mask_val).float()
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    if reduce:
        return _reduce_masked(loss, mask)
    return loss


def masked_mse_torch(preds, labels, null_val=np.nan, mask_val=None):
    """Masked MSE (torch)."""
    mask = _build_valid_mask(labels, null_val)
    if mask_val is not None:
        mask = mask * labels.ge(mask_val).float()
    loss = torch.square(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return _reduce_masked(loss, mask)


def masked_rmse_torch(preds, labels, null_val=np.nan, mask_val=None):
    """Masked RMSE (torch)."""
    return torch.sqrt(masked_mse_torch(preds=preds, labels=labels,
                                       null_val=null_val, mask_val=mask_val))


def masked_mape_torch(preds, labels, null_val=np.nan, eps=1e-5, mask_val=None):
    """Masked MAPE (torch).

    MAPE = |pred - label| / (|label| + eps)  averaged over valid samples.

    No adaptive threshold — definition is consistent across datasets.
    Use ``mask_val`` to exclude small ground-truth values (e.g. speed < 5 mph)
    where MAPE is numerically unreliable.
    """
    mask = _build_valid_mask(labels, null_val)
    if mask_val is not None:
        mask = mask * labels.ge(mask_val).float()
    if mask.sum() == 0:
        return torch.tensor(0.0, device=preds.device, dtype=preds.dtype)
    loss = torch.abs((preds - labels) / (labels + eps))
    return _reduce_masked(loss, mask)


def log_cosh_loss(preds, labels):
    """Log-cosh loss: log(cosh(pred - label)). Smooth approximation of MAE."""
    return torch.mean(torch.log(torch.cosh(preds - labels)))


def huber_loss(preds, labels, delta=1.0):
    """Huber loss (SmoothL1). Quadratic near zero, linear beyond delta."""
    residual = torch.abs(preds - labels)
    condition = torch.le(residual, delta)
    small_res = 0.5 * torch.square(residual)
    large_res = delta * residual - 0.5 * delta * delta
    return torch.mean(torch.where(condition, small_res, large_res))


def quantile_loss(preds, labels, delta=0.25):
    """Quantile (pinball) loss."""
    condition = torch.ge(labels, preds)
    large_res = delta * (labels - preds)
    small_res = (1 - delta) * (preds - labels)
    return torch.mean(torch.where(condition, large_res, small_res))


def r2_score_torch(preds, labels):
    """R² (coefficient of determination), computed on-device."""
    preds = preds.flatten().float()
    labels = labels.flatten().float()
    ss_res = ((labels - preds) ** 2).sum()
    ss_tot = ((labels - labels.mean()) ** 2).sum()
    if ss_tot == 0:
        return torch.tensor(0.0, device=preds.device, dtype=torch.float32)
    return 1.0 - ss_res / ss_tot


def explained_variance_score_torch(preds, labels):
    """Explained variance score, computed on-device."""
    preds = preds.flatten().float()
    labels = labels.flatten().float()
    diff = labels - preds
    var_res = diff.var(unbiased=False)
    var_tot = labels.var(unbiased=False)
    if var_tot == 0:
        return torch.tensor(0.0, device=preds.device, dtype=torch.float32)
    return 1.0 - var_res / var_tot


# ── NumPy losses (for non-torch evaluation paths) ─────────────────

def masked_mae_np(preds, labels, null_val=np.nan):
    """Masked MAE (NumPy)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = (~np.isnan(labels)).astype(np.float32)
        else:
            mask = np.not_equal(labels, null_val).astype(np.float32)
        loss = np.abs(np.subtract(preds, labels)).astype(np.float32)
        loss = np.nan_to_num(loss * mask)
        s = mask.sum()
        return float(loss.sum() / (s + 1e-8)) if s > 0 else 0.0


def masked_mse_np(preds, labels, null_val=np.nan):
    """Masked MSE (NumPy)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = (~np.isnan(labels)).astype(np.float32)
        else:
            mask = np.not_equal(labels, null_val).astype(np.float32)
        loss = np.square(np.subtract(preds, labels)).astype(np.float32)
        loss = np.nan_to_num(loss * mask)
        s = mask.sum()
        return float(loss.sum() / (s + 1e-8)) if s > 0 else 0.0


def masked_rmse_np(preds, labels, null_val=np.nan):
    """Masked RMSE (NumPy)."""
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))


def masked_mape_np(preds, labels, null_val=np.nan, eps=1e-5, mask_val=None):
    """Masked MAPE (NumPy). Same definition as masked_mape_torch."""
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = (~np.isnan(labels)).astype(np.float32)
        else:
            mask = np.not_equal(labels, null_val).astype(np.float32)
        if mask_val is not None:
            mask = mask * (labels >= mask_val).astype(np.float32)
        s = mask.sum()
        if s == 0:
            return 0.0
        loss = np.abs((preds - labels) / (labels + eps))
        loss = np.nan_to_num(loss * mask)
        return float(loss.sum() / (s + 1e-8))


def r2_score_np(preds, labels):
    """R² (NumPy)."""
    preds = preds.flatten()
    labels = labels.flatten()
    ss_res = np.sum((labels - preds) ** 2)
    ss_tot = np.sum((labels - labels.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def explained_variance_score_np(preds, labels):
    """Explained variance score (NumPy)."""
    preds = preds.flatten()
    labels = labels.flatten()
    diff = labels - preds
    var_res = np.var(diff)
    var_tot = np.var(labels)
    if var_tot == 0:
        return 0.0
    return float(1.0 - var_res / var_tot)
