import torch as t
import torch.nn as nn


class AdaptivePVLoss(nn.Module):
    def __init__(
        self,
        peak_sigma=1.0,
        night_threshold=1e-3,
        eps=1e-6,
        max_weight=5.0,
        target_mean=None,
        target_std=None,
        peak_threshold=None,
        ramp_threshold=None,
        normalize_weights=True,
    ):
        super(AdaptivePVLoss, self).__init__()
        self.peak_sigma = peak_sigma
        self.night_threshold = night_threshold
        self.eps = eps
        self.max_weight = max_weight
        self.target_mean = target_mean
        self.target_std = target_std
        self.peak_threshold = peak_threshold
        self.ramp_threshold = ramp_threshold
        self.normalize_weights = normalize_weights
        self.mae = nn.L1Loss(reduction='none')

    def _physical_target(self, y_true_flat):
        if self.target_mean is None or self.target_std is None:
            return y_true_flat
        target_mean = t.as_tensor(self.target_mean, dtype=y_true_flat.dtype, device=y_true_flat.device)
        target_std = t.as_tensor(self.target_std, dtype=y_true_flat.dtype, device=y_true_flat.device)
        return y_true_flat * target_std + target_mean

    def adaptive_weights(self, y_true, mask=None):
        y_true_flat = y_true.reshape(-1)
        valid_mask = t.ones_like(y_true_flat, dtype=t.bool)
        if mask is not None:
            valid_mask = mask.reshape(-1).to(device=y_true_flat.device) > 0

        if y_true_flat.numel() == 0:
            return t.ones_like(y_true), valid_mask.reshape(y_true.shape)

        y_true_physical = self._physical_target(y_true_flat)
        stat_true = y_true_physical[valid_mask]
        if stat_true.numel() == 0:
            return t.ones_like(y_true), valid_mask.reshape(y_true.shape)

        total_samples = stat_true.numel()
        if self.peak_threshold is None:
            batch_mean = stat_true.mean()
            batch_std = stat_true.std(unbiased=False)
            peak_threshold_val = batch_mean + self.peak_sigma * batch_std
        else:
            peak_threshold_val = t.as_tensor(
                self.peak_threshold,
                dtype=y_true_physical.dtype,
                device=y_true_physical.device,
            )

        mask_peak = (y_true_physical > peak_threshold_val) & valid_mask
        mask_night = (y_true_physical <= self.night_threshold) & valid_mask

        count_peak = mask_peak.sum().float()
        count_night = mask_night.sum().float()
        weight_peak_val = total_samples / (count_peak + self.eps)
        weight_night_val = total_samples / (count_night + self.eps)
        weight_peak_val = t.clamp(weight_peak_val, max=self.max_weight)
        weight_night_val = t.clamp(weight_night_val, max=self.max_weight)

        weights = t.ones_like(y_true_flat)
        weights = t.where(mask_night, weight_night_val, weights)
        weights = t.where(mask_peak, weight_peak_val, weights)
        if self.normalize_weights:
            weights = weights / weights[valid_mask].mean().clamp_min(self.eps)

        return weights.reshape(y_true.shape), valid_mask.reshape(y_true.shape)

    def forward(self, y_pred, y_true):
        y_true_flat = y_true.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)
        if y_true_flat.numel() == 0:
            return self.mae(y_pred_flat, y_true_flat).mean()

        weights, valid_mask = self.adaptive_weights(y_true)
        valid_mask_flat = valid_mask.reshape(-1)
        if not valid_mask_flat.any():
            return self.mae(y_pred_flat, y_true_flat).mean()

        pointwise_loss = self.mae(y_pred_flat, y_true_flat) * weights.reshape(-1)
        return pointwise_loss[valid_mask_flat].mean()


def adaptive_pv_mae_loss_from_args(args):
    return AdaptivePVLoss(
        peak_sigma=getattr(args, 'adaptive_pv_peak_sigma', 1.0),
        night_threshold=getattr(args, 'adaptive_pv_night_threshold', 1e-3),
        eps=getattr(args, 'adaptive_pv_eps', 1e-6),
        max_weight=getattr(args, 'adaptive_pv_max_weight', 5.0),
        target_mean=getattr(args, 'adaptive_pv_target_mean', None),
        target_std=getattr(args, 'adaptive_pv_target_std', None),
        peak_threshold=getattr(args, 'adaptive_pv_peak_threshold', None),
        ramp_threshold=getattr(args, 'adaptive_pv_ramp_threshold', None),
        normalize_weights=getattr(args, 'adaptive_pv_normalize_weights', True),
    )


class DataDrivenSegmentBalancedPVLoss(nn.Module):
    def __init__(
        self,
        target_mean=None,
        target_std=None,
        night_threshold=0.0,
        peak_threshold=None,
        ramp_threshold=None,
        eps=1e-6,
    ):
        super(DataDrivenSegmentBalancedPVLoss, self).__init__()
        self.target_mean = target_mean
        self.target_std = target_std
        self.night_threshold = night_threshold
        self.peak_threshold = peak_threshold
        self.ramp_threshold = ramp_threshold
        self.eps = eps
        self.mae = nn.L1Loss(reduction='none')

    def _physical_target(self, y_true):
        y_true_flat = y_true.reshape(-1)
        if self.target_mean is None or self.target_std is None:
            return y_true
        target_mean = t.as_tensor(self.target_mean, dtype=y_true_flat.dtype, device=y_true_flat.device)
        target_std = t.as_tensor(self.target_std, dtype=y_true_flat.dtype, device=y_true_flat.device)
        return (y_true_flat * target_std + target_mean).reshape(y_true.shape)

    def _peak_threshold(self, y_true_physical, valid_mask):
        if self.peak_threshold is not None:
            return t.as_tensor(
                self.peak_threshold,
                dtype=y_true_physical.dtype,
                device=y_true_physical.device,
            )

        stat_true = y_true_physical[valid_mask]
        non_night = stat_true[stat_true > self.night_threshold]
        if non_night.numel() == 0:
            non_night = stat_true
        return t.quantile(non_night, 0.9)

    def _ramp_mask(self, y_true_physical, valid_mask, night_mask):
        ramp_mask = t.zeros_like(valid_mask, dtype=t.bool)
        if y_true_physical.dim() < 2 or y_true_physical.size(1) <= 1:
            return ramp_mask

        diff = t.zeros_like(y_true_physical)
        diff[:, 1:, :] = t.abs(y_true_physical[:, 1:, :] - y_true_physical[:, :-1, :])
        if self.ramp_threshold is not None:
            ramp_threshold = t.as_tensor(
                self.ramp_threshold,
                dtype=y_true_physical.dtype,
                device=y_true_physical.device,
            )
        else:
            ramp_candidates = diff[valid_mask & (~night_mask)]
            ramp_candidates = ramp_candidates[ramp_candidates > self.eps]
            if ramp_candidates.numel() == 0:
                return ramp_mask
            ramp_threshold = t.quantile(ramp_candidates, 0.8)
        if ramp_threshold <= self.eps:
            return ramp_mask
        return (diff >= ramp_threshold) & valid_mask & (~night_mask)

    def segment_weights(self, y_true, mask=None):
        valid_mask = t.ones_like(y_true, dtype=t.bool)
        if mask is not None:
            valid_mask = mask.to(device=y_true.device) > 0

        if not valid_mask.any():
            return t.ones_like(y_true), valid_mask

        y_true_physical = self._physical_target(y_true)
        peak_threshold = self._peak_threshold(y_true_physical, valid_mask)
        night_mask = (y_true_physical <= self.night_threshold) & valid_mask
        peak_mask = (y_true_physical > peak_threshold) & valid_mask
        ramp_mask = self._ramp_mask(y_true_physical, valid_mask, night_mask)

        segment_masks = [
            peak_mask,
            ramp_mask & (~peak_mask),
            night_mask & (~peak_mask),
        ]
        occupied = segment_masks[0] | segment_masks[1] | segment_masks[2]
        segment_masks.append(valid_mask & (~occupied))

        total_samples = valid_mask.sum().float()
        weights = t.ones_like(y_true)
        for segment_mask in segment_masks:
            count = segment_mask.sum().float()
            if count <= 0:
                continue
            segment_weight = t.sqrt(total_samples / (count + self.eps))
            weights = t.where(segment_mask, segment_weight, weights)

        weights = weights / weights[valid_mask].mean().clamp_min(self.eps)
        return weights, valid_mask

    def forward(self, y_pred, y_true):
        pointwise_loss = self.mae(y_pred, y_true)
        valid_mask = t.ones_like(y_true, dtype=t.bool)
        weights, valid_mask = self.segment_weights(y_true)
        if not valid_mask.any():
            return pointwise_loss.mean()

        return (pointwise_loss * weights)[valid_mask].mean()


def adaptive_segment_pv_loss_from_args(args):
    return DataDrivenSegmentBalancedPVLoss(
        target_mean=getattr(args, 'adaptive_pv_target_mean', None),
        target_std=getattr(args, 'adaptive_pv_target_std', None),
        night_threshold=getattr(args, 'adaptive_pv_night_threshold', 0.0),
        peak_threshold=getattr(args, 'adaptive_pv_peak_threshold', None),
        ramp_threshold=getattr(args, 'adaptive_pv_ramp_threshold', None),
        eps=getattr(args, 'adaptive_pv_eps', 1e-6),
    )


def point_loss_from_args(args):
    loss_name = getattr(args, 'point_train_loss', 'adaptive_segment_pv').lower()
    if loss_name in ['adaptive_segment_pv', 'adaptive_segment_balanced_pv', 'adaptive_segment_pv_loss']:
        return adaptive_segment_pv_loss_from_args(args)
    if loss_name in ['adaptivepvmae', 'adaptive_pv_mae', 'adaptivepv']:
        return adaptive_pv_mae_loss_from_args(args)
    if loss_name in ['mse', 'mseloss']:
        return nn.MSELoss()
    if loss_name in ['mae', 'maeloss', 'l1']:
        return nn.L1Loss()
    if loss_name in ['huber', 'huberloss', 'smoothl1']:
        return nn.HuberLoss(delta=getattr(args, 'huber_delta', 1.0))
    raise ValueError("Unsupported point_train_loss. Choose adaptive_segment_pv, adaptivepvmae, mse, mae, or huber.")


class AQLLoss(nn.Module):
    def __init__(self, quantiles=None):
        super(AQLLoss, self).__init__()
        self.quantiles = quantiles or [0.05, 0.1, 0.5, 0.9, 0.95]

    def forward(self, outputs, y_true):
        if not isinstance(outputs, dict):
            raise ValueError("AQLLoss requires quantile forecast outputs.")
        quantile_preds = outputs['quantiles']
        return self.aql_loss(quantile_preds, y_true)

    def aql_loss(self, quantile_preds, y_true):
        quantiles = t.tensor(
            self.quantiles,
            dtype=quantile_preds.dtype,
            device=quantile_preds.device,
        )
        errors = y_true.unsqueeze(-1) - quantile_preds
        loss = t.maximum(quantiles * errors, (quantiles - 1) * errors)
        return loss.mean()


def quantile_pv_loss_from_args(args):
    quantiles = getattr(args, 'quantiles', '0.05,0.1,0.5,0.9,0.95')
    if isinstance(quantiles, str):
        quantiles = [float(q.strip()) for q in quantiles.split(',') if q.strip()]

    return AQLLoss(quantiles=quantiles)
