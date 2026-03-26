import os
import numpy as np
import pandas as pd


def _mask_and_flat(y_pred, y_true):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    # allow broadcasting shapes; flatten after masking
    if y_pred.shape != y_true.shape:
        try:
            y_pred = np.asarray(y_pred)
            y_true = np.asarray(y_true)
        except Exception:
            pass
    mask = ~np.isnan(y_true)
    return y_pred[mask], y_true[mask]


def evaluate_model(y_pred, y_true, metrics=None, path=None):
    """Compute simple evaluation metrics and optionally save to CSV.

    This is a lightweight implementation intended for tests and simple scripts.
    It supports the common metrics used across the repository, including
    masked variants that ignore `NaN` values in the ground truth.
    """
    # 如果没有指定指标，默认使用全部指标，顺序为 MAE, MSE, RMSE, MAPE, masked_MAE, masked_MSE, masked_RMSE, masked_MAPE, R2, EVAR
    if metrics is None:
        metrics = ['MAE', 'MSE', 'RMSE', 'MAPE', 'masked_MAE', 'masked_MSE', 'masked_RMSE', 'masked_MAPE', 'R2', 'EVAR']
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    results = {}
    for metric in metrics:
        if metric == 'MAE':
            results['MAE'] = float(np.mean(np.abs(y_pred - y_true)))
        elif metric == 'MSE':
            results['MSE'] = float(np.mean((y_pred - y_true) ** 2))
        elif metric == 'RMSE':
            results['RMSE'] = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        elif metric == 'MAPE':
            denom = np.where(np.abs(y_true) < 1e-6, 1e-6, np.abs(y_true))
            results['MAPE'] = float(np.mean(np.abs((y_pred - y_true) / denom)))
        elif metric.startswith('masked_'):
            # masked_MAE etc. Mask based on NaNs in y_true
            base = metric.split('_', 1)[1]
            yp, yt = _mask_and_flat(y_pred, y_true)
            if yt.size == 0:
                val = 0.0
            else:
                if base == 'MAE':
                    val = float(np.mean(np.abs(yp - yt)))
                elif base == 'MSE':
                    val = float(np.mean((yp - yt) ** 2))
                elif base == 'RMSE':
                    val = float(np.sqrt(np.mean((yp - yt) ** 2)))
                elif base == 'MAPE':
                    denom = np.where(np.abs(yt) < 1e-6, 1e-6, np.abs(yt))
                    val = float(np.mean(np.abs((yp - yt) / denom)))
                else:
                    val = float(np.mean(np.abs(yp - yt)))
            results[metric] = val
        elif metric == 'R2':
            ypf = y_pred.flatten()
            ytf = y_true.flatten()
            ss_res = np.sum((ytf - ypf) ** 2)
            ss_tot = np.sum((ytf - np.mean(ytf)) ** 2)
            results['R2'] = float(1 - ss_res / (ss_tot + 1e-12))
        elif metric == 'EVAR':
            ypf = y_pred.flatten()
            ytf = y_true.flatten()
            var_res = np.var(ytf - ypf)
            var_true = np.var(ytf)
            results['EVAR'] = float(1 - var_res / (var_true + 1e-12))
        else:
            results[metric] = None

    if path:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        df = pd.DataFrame(list(results.items()), columns=['metric', 'value'])
        df.to_csv(path, index=False)
    return results


__all__ = ['evaluate_model']
