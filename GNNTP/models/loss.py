"""
Loss functions for traffic prediction models — v3 (mask-semantic fix).

Design principles:
- Mask domain Ω = {i | labels[i] is not NaN}.  Zero is real traffic data
  (congestion = 0 mph, night-time flow = 0 vph) and MUST be included.
- Mean is computed as sum(loss · mask) / (mask.sum() + ε) — directly over
  valid samples, batch-to-batch comparable.
- mask_val is a data-filtering preprocessor, not part of the loss definition.
  When used, the evaluand distribution D → D' must be documented in the paper.
- R² / EVAR computed on-GPU with valid-only mask to avoid mixture-R² bias.
"""

import torch
import numpy as np


# ── Core mask helper ──────────────────────────────────────────────

def _build_valid_mask(labels):
    """Build a binary validity mask: NaN positions are invalid.

    IMPORTANT: Zero is NOT treated as missing. In traffic data, speed=0 is
    congestion (real signal), flow=0 is valid night-time data. Only NaN,
    originating from sensor dropouts or padding, is excluded.

    Returns a float tensor of same shape as labels: 1.0 = valid, 0.0 = NaN.
    """
    return (~torch.isnan(labels)).float()


def _reduce_masked(loss, mask, eps=1e-8):
    """Reduce a masked loss to a scalar: mean over valid samples only.

    L = (1 / |Ω|) Σ_{i∈Ω} ℓ_i,  where Ω = {i | mask[i] = 1}.

    Args:
        loss:  per-element loss tensor (already multiplied by mask).
        mask:  binary float mask. 1=valid, 0=invalid.
        eps:   guard against all-invalid edge case.
    """
    s = mask.sum()
    if s == 0:
        return loss.new_tensor(0.0)
    return (loss * mask).sum() / (s + eps)


# ── Torch losses ──────────────────────────────────────────────────

def masked_mae_loss(y_pred, y_true):
    """Training loss with zero-masking for sequence-padded batches.

    NOTE: This function retains y_true != 0 masking because training batches
    may contain zero-padding for variable-length sequences. For evaluation,
    use masked_mae_torch() which applies NaN-only masking.
    """
    mask = (y_true != 0).float()
    loss = torch.abs(y_pred - y_true)
    return _reduce_masked(loss, mask)


def masked_mae_torch(preds, labels, null_val=None, reduce=True, mask_val=None):
    """Masked MAE (torch).

    Mask domain Ω = {i | labels[i] is not NaN}. Zero values are included.

    Args:
        null_val:  Deprecated. Kept for API compatibility; NaN-only masking.
        mask_val:  Optional data-filtering threshold. When set, only samples
                   with |label| ≥ mask_val are evaluated (D → D').
                   Must be documented in the paper as a preprocessing choice.
        reduce:    If True, returns scalar mean over valid samples.
                   If False, returns per-element masked loss tensor.
    """
    mask = _build_valid_mask(labels)
    if mask_val is not None:
        mask = mask * labels.ge(mask_val).float()
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    if reduce:
        return _reduce_masked(loss, mask)
    return loss


def masked_mse_torch(preds, labels, null_val=None, mask_val=None):
    """Masked MSE (torch). Same mask semantics as masked_mae_torch."""
    mask = _build_valid_mask(labels)
    if mask_val is not None:
        mask = mask * labels.ge(mask_val).float()
    loss = torch.square(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return _reduce_masked(loss, mask)


def masked_rmse_torch(preds, labels, null_val=None, mask_val=None):
    """Masked RMSE (torch). √MSE over valid samples."""
    return torch.sqrt(masked_mse_torch(preds=preds, labels=labels,
                                       null_val=null_val, mask_val=mask_val))


def masked_mape_torch(preds, labels, null_val=None, eps=1e-5, mask_val=None):
    """Masked MAPE (torch).

    MAPE = |pred - label| / (|label| + ε), averaged over Ω = {i | valid}.

    Use mask_val to exclude small ground-truth values where MAPE is
    numerically unreliable (e.g. speed < 5 mph). This is a data-filtering
    strategy, not part of the loss definition — document it if used.
    """
    mask = _build_valid_mask(labels)
    if mask_val is not None:
        mask = mask * labels.ge(mask_val).float()
    if mask.sum() == 0:
        return torch.tensor(0.0, device=preds.device, dtype=preds.dtype)
    loss = torch.abs((preds - labels) / (labels + eps))
    return _reduce_masked(loss, mask)


def masked_smape_torch(preds, labels, null_val=None, eps=1e-5, mask_val=None):
    """Masked sMAPE (symmetric MAPE, torch).

    sMAPE = 200 · |pred − label| / (|pred| + |label| + ε)

    Bounded [0, 200]. Does not favour under/over-prediction. Does not
    explode for small |label| because denominator includes |pred|.
    """
    mask = _build_valid_mask(labels)
    if mask_val is not None:
        mask = mask * labels.ge(mask_val).float()
    if mask.sum() == 0:
        return torch.tensor(0.0, device=preds.device, dtype=preds.dtype)
    loss = 200.0 * torch.abs(preds - labels) / (torch.abs(preds) + torch.abs(labels) + eps)
    return _reduce_masked(loss, mask)


def log_cosh_loss(preds, labels):
    """Log-cosh loss: log(cosh(pred − label)). Smooth approximation of MAE."""
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
    """R² (coefficient of determination), computed on-device over valid samples.

    R² = 1 − SS_res / SS_tot, evaluated only on Ω = {i | labels[i], preds[i]
    are not NaN}.  This yields the conditional R² on the observed set, not a
    mixture R² contaminated by invalid positions.
    """
    mask = ~torch.isnan(labels) & ~torch.isnan(preds)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=preds.device, dtype=torch.float32)
    p = preds[mask].float()
    l = labels[mask].float()
    ss_res = ((l - p) ** 2).sum()
    ss_tot = ((l - l.mean()) ** 2).sum()
    if ss_tot == 0:
        return torch.tensor(0.0, device=preds.device, dtype=torch.float32)
    return 1.0 - ss_res / ss_tot


def explained_variance_score_torch(preds, labels):
    """Explained variance score, computed on-device over valid samples."""
    mask = ~torch.isnan(labels) & ~torch.isnan(preds)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=preds.device, dtype=torch.float32)
    p = preds[mask].float()
    l = labels[mask].float()
    diff = l - p
    var_res = diff.var(unbiased=False)
    var_tot = l.var(unbiased=False)
    if var_tot == 0:
        return torch.tensor(0.0, device=preds.device, dtype=torch.float32)
    return 1.0 - var_res / var_tot


# ── NumPy losses (for non-torch evaluation paths) ─────────────────

def masked_mae_np(preds, labels, null_val=None):
    """Masked MAE (NumPy). NaN-only masking; zero is valid traffic data."""
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = (~np.isnan(labels)).astype(np.float32)
        loss = np.abs(np.subtract(preds, labels)).astype(np.float32)
        loss = np.nan_to_num(loss * mask)
        s = mask.sum()
        return float(loss.sum() / (s + 1e-8)) if s > 0 else 0.0


def masked_mse_np(preds, labels, null_val=None):
    """Masked MSE (NumPy). NaN-only masking."""
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = (~np.isnan(labels)).astype(np.float32)
        loss = np.square(np.subtract(preds, labels)).astype(np.float32)
        loss = np.nan_to_num(loss * mask)
        s = mask.sum()
        return float(loss.sum() / (s + 1e-8)) if s > 0 else 0.0


def masked_rmse_np(preds, labels, null_val=None):
    """Masked RMSE (NumPy)."""
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))


def masked_mape_np(preds, labels, null_val=None, eps=1e-5, mask_val=None):
    """Masked MAPE (NumPy). Same definition as masked_mape_torch."""
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = (~np.isnan(labels)).astype(np.float32)
        if mask_val is not None:
            mask = mask * (labels >= mask_val).astype(np.float32)
        s = mask.sum()
        if s == 0:
            return 0.0
        loss = np.abs((preds - labels) / (labels + eps))
        loss = np.nan_to_num(loss * mask)
        return float(loss.sum() / (s + 1e-8))


def masked_smape_np(preds, labels, null_val=None, eps=1e-5, mask_val=None):
    """Masked sMAPE (NumPy). Same definition as masked_smape_torch."""
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = (~np.isnan(labels)).astype(np.float32)
        if mask_val is not None:
            mask = mask * (labels >= mask_val).astype(np.float32)
        s = mask.sum()
        if s == 0:
            return 0.0
        loss = 200.0 * np.abs(preds - labels) / (np.abs(preds) + np.abs(labels) + eps)
        loss = np.nan_to_num(loss * mask)
        return float(loss.sum() / (s + 1e-8))


def r2_score_np(preds, labels):
    """R² (NumPy), over valid (non-NaN) samples only."""
    mask = ~np.isnan(labels) & ~np.isnan(preds)
    if mask.sum() == 0:
        return 0.0
    p = preds[mask].flatten()
    l = labels[mask].flatten()
    ss_res = np.sum((l - p) ** 2)
    ss_tot = np.sum((l - l.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def explained_variance_score_np(preds, labels):
    """Explained variance score (NumPy), over valid samples only."""
    mask = ~np.isnan(labels) & ~np.isnan(preds)
    if mask.sum() == 0:
        return 0.0
    p = preds[mask].flatten()
    l = labels[mask].flatten()
    diff = l - p
    var_res = np.var(diff)
    var_tot = np.var(l)
    if var_tot == 0:
        return 0.0
    return float(1.0 - var_res / var_tot)
