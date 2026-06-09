"""
Loss functions for traffic prediction models — v4 (null_val support).

Design principles:
- Mask domain Ω = {i | labels[i] is NaN-free}.  By default (null_val=None)
  zero is treated as real traffic data and MUST be included in evaluation.
  Pass null_val=0 to exclude zeros (masked_* variants).
- Mean is computed as sum(loss · mask) / (mask.sum() + ε) — directly over
  valid samples, batch-to-batch comparable.
- mask_val is a data-filtering preprocessor, not part of the loss definition.
  When used, the evaluand distribution D → D' must be documented in the paper.
- R² / EVAR computed on-GPU with valid-only mask to avoid mixture-R² bias.
"""

import torch
import numpy as np


# ── Core mask helper ──────────────────────────────────────────────

def _build_valid_mask(labels, null_val=None):
    """Build a binary validity mask.

    Always excludes NaN positions. When *null_val* is a finite number
    (e.g. 0), those positions are excluded in addition to NaN.

    Args:
        labels:   (...,) tensor of ground-truth values.
        null_val: Value to treat as missing data.
                  - None :  only NaN is excluded (eval default).
                  - 0    :  NaN and zero are excluded (masked_* metrics).
                  - np.nan: same as None (NaN-only).

    Returns:
        Float tensor of same shape, 1.0 = valid, 0.0 = invalid.
    """
    mask = ~torch.isnan(labels)
    if null_val is not None:
        try:
            is_nan = np.isnan(null_val)
        except TypeError:
            is_nan = False
        if not is_nan:
            mask = mask & labels.ne(null_val)
    return mask.float()


def _reduce_masked(loss, mask, eps=1e-8):
    """Reduce a masked loss to a scalar: mean over valid samples only.

    L = (1 / |Ω|) Σ_{i∈Ω} ℓ_i,  where Ω = {i | mask[i] = 1}.

    Uses nan_to_num to guard against NaN · 0 = NaN at invalid positions
    (a floating-point edge case when loss itself is NaN).

    Args:
        loss:  per-element loss tensor (unmasked — this function applies mask).
        mask:  binary float mask. 1=valid, 0=invalid.
        eps:   guard against all-invalid edge case.
    """
    s = mask.sum()
    if s == 0:
        return loss.new_tensor(0.0)
    return torch.nan_to_num(loss * mask).sum() / (s + eps)


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
        null_val:  Value to exclude (e.g. 0 for masked_* metrics). None = NaN-only.
        mask_val:  Optional data-filtering threshold. When set, only samples
                   with |label| ≥ mask_val are evaluated (D → D').
                   Must be documented in the paper as a preprocessing choice.
        reduce:    If True, returns scalar mean over valid samples.
                   If False, returns per-element masked loss tensor.
    """
    mask = _build_valid_mask(labels, null_val)
    if mask_val is not None:
        mask = mask * labels.abs().ge(mask_val).float()
    loss = torch.abs(preds - labels)
    if reduce:
        return _reduce_masked(loss, mask)
    return loss * mask


def masked_mse_torch(preds, labels, null_val=None, mask_val=None):
    """Masked MSE (torch). Same mask semantics as masked_mae_torch."""
    mask = _build_valid_mask(labels, null_val)
    if mask_val is not None:
        mask = mask * labels.abs().ge(mask_val).float()
    loss = torch.square(preds - labels)
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
    mask = _build_valid_mask(labels, null_val)
    if mask_val is not None:
        mask = mask * labels.abs().ge(mask_val).float()
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
    mask = _build_valid_mask(labels, null_val)
    if mask_val is not None:
        mask = mask * labels.abs().ge(mask_val).float()
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


def r2_score_torch(preds, labels, null_val=None):
    """R² (coefficient of determination), computed on-device over valid samples.

    R² = 1 − SS_res / SS_tot.  Mask domain Ω follows _build_valid_mask so
    that R² is computed on the same sample set as other masked metrics.
    """
    mask = _build_valid_mask(labels, null_val) * (~torch.isnan(preds)).float()
    mask_bool = mask.bool()
    if mask_bool.sum() == 0:
        return torch.tensor(0.0, device=preds.device, dtype=torch.float32)
    p = preds[mask_bool].float()
    l = labels[mask_bool].float()
    ss_res = ((l - p) ** 2).sum()
    ss_tot = ((l - l.mean()) ** 2).sum()
    if ss_tot == 0:
        return torch.tensor(0.0, device=preds.device, dtype=torch.float32)
    return 1.0 - ss_res / ss_tot


def explained_variance_score_torch(preds, labels, null_val=None):
    """Explained variance score, computed on-device over valid samples.

    Mask domain follows _build_valid_mask for consistency with other metrics.
    """
    mask = _build_valid_mask(labels, null_val) * (~torch.isnan(preds)).float()
    mask_bool = mask.bool()
    if mask_bool.sum() == 0:
        return torch.tensor(0.0, device=preds.device, dtype=torch.float32)
    p = preds[mask_bool].float()
    l = labels[mask_bool].float()
    diff = l - p
    var_res = diff.var(unbiased=False)
    var_tot = l.var(unbiased=False)
    if var_tot == 0:
        return torch.tensor(0.0, device=preds.device, dtype=torch.float32)
    return 1.0 - var_res / var_tot


# ── NumPy losses (for non-torch evaluation paths) ─────────────────

def _build_valid_mask_np(labels, null_val=None):
    """NumPy version of _build_valid_mask."""
    mask = ~np.isnan(labels)
    if null_val is not None:
        try:
            is_nan = np.isnan(null_val)
        except TypeError:
            is_nan = False
        if not is_nan:
            mask = mask & (labels != null_val)
    return mask.astype(np.float32)


def masked_mae_np(preds, labels, null_val=None):
    """Masked MAE (NumPy)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = _build_valid_mask_np(labels, null_val)
        loss = np.abs(np.subtract(preds, labels)).astype(np.float32)
        loss = np.nan_to_num(loss * mask)
        s = mask.sum()
        return float(loss.sum() / (s + 1e-8)) if s > 0 else 0.0


def masked_mse_np(preds, labels, null_val=None):
    """Masked MSE (NumPy)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = _build_valid_mask_np(labels, null_val)
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
        mask = _build_valid_mask_np(labels, null_val)
        if mask_val is not None:
            mask = mask * (np.abs(labels) >= mask_val).astype(np.float32)
        s = mask.sum()
        if s == 0:
            return 0.0
        loss = np.abs((preds - labels) / (labels + eps))
        loss = np.nan_to_num(loss * mask)
        return float(loss.sum() / (s + 1e-8))


def masked_smape_np(preds, labels, null_val=None, eps=1e-5, mask_val=None):
    """Masked sMAPE (NumPy). Same definition as masked_smape_torch."""
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = _build_valid_mask_np(labels, null_val)
        if mask_val is not None:
            mask = mask * (np.abs(labels) >= mask_val).astype(np.float32)
        s = mask.sum()
        if s == 0:
            return 0.0
        loss = 200.0 * np.abs(preds - labels) / (np.abs(preds) + np.abs(labels) + eps)
        loss = np.nan_to_num(loss * mask)
        return float(loss.sum() / (s + 1e-8))


def r2_score_np(preds, labels, null_val=None):
    """R² (NumPy), over valid samples (consistent with _build_valid_mask_np)."""
    mask = _build_valid_mask_np(labels, null_val) * (~np.isnan(preds)).astype(np.float32)
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return 0.0
    p = preds[mask_bool].flatten()
    l = labels[mask_bool].flatten()
    ss_res = np.sum((l - p) ** 2)
    ss_tot = np.sum((l - l.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def explained_variance_score_np(preds, labels, null_val=None):
    """Explained variance score (NumPy), over valid samples (consistent with _build_valid_mask_np)."""
    mask = _build_valid_mask_np(labels, null_val) * (~np.isnan(preds)).astype(np.float32)
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return 0.0
    p = preds[mask_bool].flatten()
    l = labels[mask_bool].flatten()
    diff = l - p
    var_res = np.var(diff)
    var_tot = np.var(l)
    if var_tot == 0:
        return 0.0
    return float(1.0 - var_res / var_tot)
