import numpy as np


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def R2(pred, true):
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - ss_res / ss_tot


def Peak_MAE(pred, true, peak_mask):
    peak_mask = np.asarray(peak_mask, dtype=bool)
    if peak_mask.sum() == 0:
        return np.nan
    return MAE(np.asarray(pred)[peak_mask], np.asarray(true)[peak_mask])


def Peak_MSE(pred, true, peak_mask):
    peak_mask = np.asarray(peak_mask, dtype=bool)
    if peak_mask.sum() == 0:
        return np.nan
    return MSE(np.asarray(pred)[peak_mask], np.asarray(true)[peak_mask])


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    r2 = R2(pred, true)

    return mae, mse, rmse, r2


def peak_metric(pred, true, peak_mask):
    peak_mae = Peak_MAE(pred, true, peak_mask)
    peak_mse = Peak_MSE(pred, true, peak_mask)

    return peak_mae, peak_mse


def point_metric(pred, true, peak_mask=None):
    mae, mse, rmse, r2 = metric(pred, true)
    if peak_mask is None:
        return mae, mse, rmse, r2

    peak_mae, peak_mse = peak_metric(pred, true, peak_mask)
    return mae, mse, rmse, r2, peak_mae, peak_mse


def efficiency_metric(train_time, inference_latency, params_k, memory_mb):
    return np.array([train_time, inference_latency, params_k, memory_mb], dtype=np.float64)


def AQL(quantile_pred, true, quantiles=(0.05, 0.1, 0.5, 0.9, 0.95)):
    quantiles = np.asarray(quantiles, dtype=np.float64)
    errors = np.expand_dims(true, axis=-1) - quantile_pred
    loss = np.maximum(quantiles * errors, (quantiles - 1) * errors)
    return np.mean(loss)


def PICP(lower, upper, true):
    return np.mean((true >= lower) & (true <= upper))


def PINAW(lower, upper, true):
    true_range = np.max(true) - np.min(true)
    if true_range == 0:
        return 0.0
    return np.mean(upper - lower) / true_range


def _quantile_index(quantiles, target):
    quantiles = np.asarray(quantiles, dtype=np.float64)
    matches = np.where(np.isclose(quantiles, target, atol=1e-6))[0]
    if matches.size == 0:
        raise ValueError(f"Quantile level {target} is required for probability metrics.")
    return int(matches[0])


def probability_metric(quantile_pred, true, quantiles=(0.05, 0.1, 0.5, 0.9, 0.95)):
    quantiles = list(quantiles)
    lower80_idx = _quantile_index(quantiles, 0.1)
    upper80_idx = _quantile_index(quantiles, 0.9)
    lower90_idx = _quantile_index(quantiles, 0.05)
    upper90_idx = _quantile_index(quantiles, 0.95)

    lower80 = quantile_pred[..., lower80_idx]
    upper80 = quantile_pred[..., upper80_idx]
    lower90 = quantile_pred[..., lower90_idx]
    upper90 = quantile_pred[..., upper90_idx]
    picp_80 = PICP(lower80, upper80, true)
    pinaw_80 = PINAW(lower80, upper80, true)
    picp_90 = PICP(lower90, upper90, true)
    pinaw_90 = PINAW(lower90, upper90, true)
    aql = AQL(quantile_pred, true, quantiles)

    return picp_80, pinaw_80, picp_90, pinaw_90, aql
