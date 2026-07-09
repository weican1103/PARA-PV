import numpy as np
import torch
import torch.nn as nn
import einops

from src.PARA_PV.temporal_prior import TemporalForecastingPriorBranch
from layers.Embed import PatchEmbedding

class PhysicsAwarePatchMemoryBank:
    """Memory bank for PA-RAL: Physics-Aware Retrieval-Augmented Learner."""

    def __init__(self, max_size, patch_size, feature_dim, pred_len, device=None):
        """
        Initialize the patch memory bank.
        
        Args:
            max_size (int): Maximum number of patches to store.
            patch_size (int): Size of each patch.
            feature_dim (int): Dimensionality of each patch feature.
            device (torch.device): Device to store memory bank on (CPU/GPU).
        """
        self.max_size = max_size
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.pred_len = pred_len
        self.device = device if device is not None else torch.device('cpu')
        self.patches = torch.zeros((max_size, feature_dim), device=self.device)  # [100, d_model]
        self.trajectories = torch.zeros((max_size, pred_len), device=self.device)
        self.power_levels = torch.zeros(max_size, device=self.device)
        self.pv_states = torch.zeros(max_size, dtype=torch.long, device=self.device)
        self.day_buckets = torch.zeros(max_size, dtype=torch.long, device=self.device)
        self.ptr = 0
        self.filled = 0

    def update(self, new_patches, metadata=None):
        """
        Update the patch memory bank with new patches using circular buffer strategy.
        
        Args:
            new_patches (Tensor): New patches to add to the memory bank.
        """
        n = new_patches.size(0)
        new_patches_flat = new_patches.mean(dim=1)  # [n, d_model]
        
        if metadata is not None:
            metadata = {
                key: value.to(self.device) if value is not None else None
                for key, value in metadata.items()
            }

        remaining = new_patches_flat
        start = 0
        while start < n:
            write_count = min(self.max_size - self.ptr, n - start)
            write_slice = slice(self.ptr, self.ptr + write_count)
            read_slice = slice(start, start + write_count)
            self.patches[write_slice] = remaining[read_slice]
            if metadata is not None:
                if metadata.get('analog_trajectories') is not None:
                    self.trajectories[write_slice] = metadata['analog_trajectories'][read_slice]
                if metadata.get('power_levels') is not None:
                    self.power_levels[write_slice] = metadata['power_levels'][read_slice]
                if metadata.get('pv_states') is not None:
                    self.pv_states[write_slice] = metadata['pv_states'][read_slice].long()
                if metadata.get('day_buckets') is not None:
                    self.day_buckets[write_slice] = metadata['day_buckets'][read_slice].long()

            self.ptr = (self.ptr + write_count) % self.max_size
            self.filled = min(self.max_size, self.filled + write_count)
            start += write_count

    def retrieve(
        self,
        query_patches,
        top_k=5,
        metadata=None,
        shape_weight=1.0,
        power_weight=1.0,
        state_weight=1.0,
        bucket_weight=1.0,
        power_scale=1.0,
        day_buckets=24,
    ):
        """
        Retrieve the top-k most similar patches from the memory bank.
        
        Args:
            query_patches (Tensor): Query patches for retrieval.
            top_k (int): Number of nearest neighbors to retrieve.
        
        Returns:
            retrieved_patches (Tensor): Retrieved patches from the memory bank.
            indices (Tensor): Indices of the retrieved patches.
        """
        query_flat = query_patches.mean(dim=1)  # [B * n_vars, d_model]
        if self.filled == 0:
            empty = torch.zeros(
                query_flat.size(0),
                top_k,
                self.feature_dim,
                device=query_flat.device,
                dtype=query_flat.dtype,
            )
            empty_trajectories = torch.zeros(
                query_flat.size(0),
                top_k,
                self.pred_len,
                device=query_flat.device,
                dtype=query_flat.dtype,
            )
            empty_weights = torch.zeros(query_flat.size(0), top_k, device=query_flat.device, dtype=query_flat.dtype)
            indices = torch.zeros(query_flat.size(0), top_k, device=query_flat.device, dtype=torch.long)
            return empty, indices, empty_trajectories, empty_weights

        memory_flat = self.patches[:self.filled]  # [filled, d_model]

        query_norm = torch.nn.functional.normalize(query_flat, dim=-1)
        memory_norm = torch.nn.functional.normalize(memory_flat, dim=-1)
        similarity = shape_weight * torch.matmul(query_norm, memory_norm.T)

        if metadata is not None:
            if metadata.get('power_levels') is not None:
                query_power = metadata['power_levels'].to(query_flat.device).unsqueeze(-1)
                memory_power = self.power_levels[:self.filled].to(query_flat.device).unsqueeze(0)
                power_similarity = 1.0 / (1.0 + torch.abs(query_power - memory_power) / max(power_scale, 1e-6))
                similarity = similarity + power_weight * power_similarity

            if metadata.get('pv_states') is not None:
                query_state = metadata['pv_states'].to(query_flat.device).long().unsqueeze(-1)
                memory_state = self.pv_states[:self.filled].to(query_flat.device).unsqueeze(0)
                state_similarity = (query_state == memory_state).to(similarity.dtype)
                similarity = similarity + state_weight * state_similarity

            if metadata.get('day_buckets') is not None and day_buckets > 1:
                query_bucket = metadata['day_buckets'].to(query_flat.device).long().unsqueeze(-1)
                memory_bucket = self.day_buckets[:self.filled].to(query_flat.device).unsqueeze(0)
                raw_distance = torch.abs(query_bucket - memory_bucket)
                circular_distance = torch.minimum(raw_distance, day_buckets - raw_distance)
                bucket_similarity = 1.0 - circular_distance.to(similarity.dtype) / max(day_buckets / 2.0, 1.0)
                similarity = similarity + bucket_weight * bucket_similarity.clamp_min(0.0)

        actual_top_k = min(top_k, self.filled)
        top_scores, indices = similarity.topk(actual_top_k, dim=-1)
        retrieved_patches = self.patches[indices]
        retrieved_trajectories = self.trajectories[indices]
        retrieval_weights = torch.softmax(top_scores, dim=-1)
        if actual_top_k < top_k:
            pad_count = top_k - actual_top_k
            pad = retrieved_patches[:, -1:, :].expand(-1, pad_count, -1)
            retrieved_patches = torch.cat([retrieved_patches, pad], dim=1)
            trajectory_pad = retrieved_trajectories[:, -1:, :].expand(-1, pad_count, -1)
            retrieved_trajectories = torch.cat([retrieved_trajectories, trajectory_pad], dim=1)
            weight_pad = torch.zeros(retrieval_weights.size(0), pad_count, device=query_flat.device, dtype=query_flat.dtype)
            retrieval_weights = torch.cat([retrieval_weights, weight_pad], dim=1)
            index_pad = indices[:, -1:].expand(-1, pad_count)
            indices = torch.cat([indices, index_pad], dim=1)
        return retrieved_patches, indices, retrieved_trajectories, retrieval_weights


class PhysicsAwareDistributionShiftCorrection(nn.Module):
    """Distribution shift correction following the weather-score/timestamp design."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pred_len = config.pred_len
        self.hidden_dim = getattr(config, 'pa_dsc_hidden', 128)
        self.use_physics_clip = getattr(config, 'pa_dsc_physics_clip', True)
        self.capacity = getattr(config, 'chronos_capacity', 50.0)
        self.low_power_margin = getattr(config, 'pa_dsc_low_power_margin', 0.1)

        self.input_dim = 4  # power sequence, weather score, timestamp, day/night one-hot
        self.input_projection = nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=1)
        self.tcn_blocks = nn.ModuleList([
            self._make_tcn_block(dilation)
            for dilation in [1, 2, 4, 8]
        ])
        self.output_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(getattr(config, 'dropout', 0.1)),
            nn.Linear(self.hidden_dim, 3),
        )
        final_layer = self.output_head[-1]
        nn.init.zeros_(final_layer.weight)
        with torch.no_grad():
            final_layer.bias[0].fill_(0.0)   # tanh -> signed mean shift initialized at 0
            final_layer.bias[1].fill_(6.0)   # sigmoid -> std close to 1
            final_layer.bias[2].fill_(-6.0)  # sigmoid -> initially trust the base prediction

    def _make_tcn_block(self, dilation):
        kernel_size = 3
        padding = dilation * (kernel_size - 1) // 2
        return nn.Sequential(
            nn.Conv1d(
                self.hidden_dim,
                self.hidden_dim,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=self.hidden_dim,
            ),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(getattr(self.config, 'dropout', 0.1)),
        )

    @staticmethod
    def _fit_time_length(x, target_len):
        if x is None:
            return None
        if x.size(1) == target_len:
            return x
        if x.size(1) > target_len:
            return x[:, -target_len:, :]
        pad = x[:, -1:, :].expand(-1, target_len - x.size(1), -1)
        return torch.cat([x, pad], dim=1)

    def _power_sequence(self, x_enc, target_len):
        power = x_enc[:, :, -1:]
        return self._fit_time_length(power, target_len)

    def _topsis_weather_score(self, x_enc, target_len):
        if x_enc.size(-1) <= 1:
            return torch.zeros(
                x_enc.size(0),
                target_len,
                1,
                device=x_enc.device,
                dtype=x_enc.dtype,
            )

        weather = x_enc[:, :, :min(3, x_enc.size(-1) - 1)]
        min_v = weather.amin(dim=1, keepdim=True)
        max_v = weather.amax(dim=1, keepdim=True)
        z = (weather - min_v) / (max_v - min_v).clamp_min(1e-6)
        p = z / z.sum(dim=1, keepdim=True).clamp_min(1e-6)
        entropy = -(p * torch.log(p.clamp_min(1e-6))).sum(dim=1)
        entropy = entropy / np.log(max(weather.size(1), 2))
        weights = (1.0 - entropy)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weighted = z * weights.unsqueeze(1)
        positive_ideal = weighted.amax(dim=1, keepdim=True)
        negative_ideal = weighted.amin(dim=1, keepdim=True)
        dist_pos = torch.sqrt(((weighted - positive_ideal) ** 2).sum(dim=-1))
        dist_neg = torch.sqrt(((weighted - negative_ideal) ** 2).sum(dim=-1))
        score = dist_neg / (dist_pos + dist_neg).clamp_min(1e-6)
        score = score.unsqueeze(-1)
        score = self._fit_time_length(score, target_len)
        return score.to(dtype=x_enc.dtype)

    def _timestamp_and_daynight(self, x_mark_enc, x_mark_dec, target_prediction):
        target_len = target_prediction.size(1)
        marks = self._fit_time_length(x_mark_dec, target_len)
        if marks is None:
            marks = self._fit_time_length(x_mark_enc, target_len)
        if marks is None:
            timestamp = torch.linspace(
                0.0,
                1.0,
                target_len,
                device=target_prediction.device,
                dtype=target_prediction.dtype,
            ).view(1, target_len, 1).expand(target_prediction.size(0), -1, -1)
            daynight = torch.ones_like(timestamp)
            return timestamp, daynight

        marks = marks.to(device=target_prediction.device, dtype=target_prediction.dtype)
        day_of_year = None
        if marks.size(-1) >= 5 and marks[:, :, 1].amin() >= -0.6 and marks[:, :, 1].amax() <= 0.6:
            minute = marks[:, :, 0] + 0.5
            hour = marks[:, :, 1] + 0.5
            day_of_year = marks[:, :, 4] + 0.5
            timestamp = (day_of_year + hour / 365.0 + minute / (365.0 * 24.0)).unsqueeze(-1)
            hour_value = hour * 23.0
        elif marks.size(-1) >= 4:
            month = (marks[:, :, 0] - 1.0) / 11.0
            day = (marks[:, :, 1] - 1.0) / 30.0
            hour_value = marks[:, :, 3]
            hour = hour_value / 23.0
            day_of_year = (month * 365.0 + day * 30.0).clamp(0.0, 365.0) / 365.0
            timestamp = (month + day / 12.0 + hour / (12.0 * 31.0)).unsqueeze(-1)
        else:
            timestamp = torch.linspace(
                0.0,
                1.0,
                target_len,
                device=target_prediction.device,
                dtype=target_prediction.dtype,
            ).view(1, target_len, 1).expand(target_prediction.size(0), -1, -1)
            hour_value = torch.full_like(timestamp.squeeze(-1), 12.0)
            day_of_year = torch.full_like(hour_value, 0.5)

        daylight_half_width = 6.0 + 1.5 * torch.cos(2.0 * np.pi * (day_of_year - 172.0 / 365.0))
        sunrise = 12.0 - daylight_half_width
        sunset = 12.0 + daylight_half_width
        daynight = ((hour_value >= sunrise) & (hour_value <= sunset)).to(target_prediction.dtype).unsqueeze(-1)
        return timestamp.clamp(0.0, 1.0), daynight

    def _physics_clip(self, corrected, means, stdev, daynight, power_sequence):
        if not self.use_physics_clip or means is None or stdev is None:
            return corrected

        target_mean = means[:, :, -1:].to(corrected.device)
        target_stdev = stdev[:, :, -1:].to(corrected.device).clamp_min(1e-6)
        dataset_mean = getattr(self.config, 'adaptive_pv_target_mean', None)
        dataset_std = getattr(self.config, 'adaptive_pv_target_std', None)
        if dataset_mean is None or dataset_std is None:
            return corrected

        dataset_mean = torch.as_tensor(dataset_mean, dtype=corrected.dtype, device=corrected.device)
        dataset_std = torch.as_tensor(dataset_std, dtype=corrected.dtype, device=corrected.device).clamp_min(1e-6)
        zero_scaled = (0.0 - dataset_mean) / dataset_std
        capacity_scaled = (self.capacity - dataset_mean) / dataset_std
        zero_norm = (zero_scaled - target_mean) / target_stdev
        capacity_norm = (capacity_scaled - target_mean) / target_stdev
        lower = torch.minimum(zero_norm, capacity_norm).expand_as(corrected)
        upper = torch.maximum(zero_norm, capacity_norm).expand_as(corrected)
        corrected = torch.minimum(corrected, upper)
        timestamp_night = daynight.to(device=corrected.device, dtype=corrected.dtype) < 0.5
        low_power = power_sequence.to(device=corrected.device, dtype=corrected.dtype) <= (lower + self.low_power_margin)
        lower_mask = timestamp_night | low_power
        corrected = torch.where(lower_mask, torch.maximum(corrected, lower), corrected)
        return corrected

    def forward(self, predictions, x_enc, x_mark_enc=None, x_mark_dec=None, metadata=None, means=None, stdev=None):
        target_prediction = predictions[:, :, -1:]
        power_sequence = self._power_sequence(x_enc, target_prediction.size(1)).to(target_prediction.device)
        weather_score = self._topsis_weather_score(x_enc, target_prediction.size(1)).to(target_prediction.device)
        timestamp, daynight = self._timestamp_and_daynight(x_mark_enc, x_mark_dec, target_prediction)
        condition = torch.cat(
            [
                power_sequence,
                weather_score,
                timestamp,
                daynight,
            ],
            dim=-1,
        )
        hidden = self.input_projection(condition.transpose(1, 2))
        for block in self.tcn_blocks:
            hidden = hidden + block(hidden)
        hidden = hidden.transpose(1, 2)
        raw_mean, raw_std, raw_gate = self.output_head(hidden).chunk(3, dim=-1)
        mean = torch.tanh(raw_mean)
        std = torch.sigmoid(raw_std)
        gate = torch.sigmoid(raw_gate)
        distribution_corrected = target_prediction * std + mean
        corrected_target = target_prediction + gate * (distribution_corrected - target_prediction)
        corrected_target = self._physics_clip(
            corrected_target,
            means,
            stdev,
            daynight,
            power_sequence,
        )

        corrected = predictions.clone()
        corrected[:, :, -1:] = corrected_target
        return corrected


class Model(nn.Module):
    """
    PARA-PV model for PV forecasting with PA-RAL, Chronos residual correction, PA-DSC, and physics constraints.
    """
    def __init__(self, config, **kwargs):
        super(Model, self).__init__()
        self.config = config
        self.device = torch.device('cuda:{}'.format(self.config.gpu))
        self.use_mem_gate = config.use_mem_gate
        self.use_quantile_forecast = getattr(config, 'use_quantile_forecast', True)
        quantiles = getattr(config, 'quantiles', '0.1,0.5,0.9')
        if isinstance(quantiles, str):
            self.quantiles = [float(q.strip()) for q in quantiles.split(',') if q.strip()]
        else:
            self.quantiles = [float(q) for q in quantiles]
        self.quantiles = sorted(self.quantiles)
        self.median_quantile_idx = min(
            range(len(self.quantiles)),
            key=lambda idx: abs(self.quantiles[idx] - 0.5)
        )
        
        self.use_pa_ral = getattr(config, 'use_pa_ral', True)
        self.use_pa_dsc = getattr(config, 'use_pa_dsc', False)
        self.use_chronos_prior = getattr(config, 'use_chronos_prior', False)
        self.pa_ral_branch_name = 'PA-RAL'
        self.day_bucket_count = 24
        self.use_pa_ral_analog = getattr(config, 'use_pa_ral_analog', False)
        self.chronos_capacity = getattr(config, 'chronos_capacity', 50.0)
        self.use_chronos_adapter = getattr(config, 'use_chronos_adapter', False)
        self.chronos_adapter_hidden = getattr(config, 'chronos_adapter_hidden', config.pred_len)
        retrieval_weight_init = torch.ones(4, dtype=torch.float32)
        self.pa_ral_retrieval_logits = nn.Parameter(torch.log(retrieval_weight_init))

        # Initialize PA-RAL memory bank
        self.patch_memory_bank = PhysicsAwarePatchMemoryBank(
            max_size=config.patch_memory_size,  # e.g., 100 patches
            patch_size=config.patch_len,
            feature_dim=config.d_model,
            pred_len=config.pred_len,
            device=self.device
        )
        
        self._init_modules(config)
        self.use_chronos_prior_cache = getattr(config, 'use_chronos_prior_cache', False)
        self.temporal_prior_branch = (
            TemporalForecastingPriorBranch(config)
            if self.use_chronos_prior and not self.use_chronos_prior_cache
            else None
        )

    def _init_modules(self, config):
        self.patch_embedding = PatchEmbedding(
            config.d_model, 
            config.patch_len, 
            config.stride, 
            config.padding, 
            config.dropout
        )
        num_patches = int((config.seq_len + config.padding - config.patch_len) / config.stride + 1)
        self.head_nf = config.d_model * num_patches
        self.flatten = nn.Flatten(start_dim=-2)
        
        # Main memory prediction head
        self.memory_head = nn.Sequential(
            nn.Linear(self.head_nf, config.pred_len),
            nn.Dropout(config.dropout)
        )
        
        # Main temporal head
        self.temporal_head = nn.Sequential(
            nn.Linear(self.head_nf, config.d_model),
            nn.Dropout(config.dropout)
        )
        
        # Memory fusion gate
        if self.use_mem_gate:
            self.memory_fusion_gate = nn.Sequential(
                nn.Linear(config.d_model * 2, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, 2),
                nn.Softmax(dim=-1)
            )

        self.analog_fusion_gate = nn.Sequential(
            nn.Linear(config.pred_len * 2 + 7, config.pred_len),
            nn.GELU(),
            nn.Linear(config.pred_len, 1),
            nn.Sigmoid()
        )

        if self.use_chronos_adapter:
            adapter_dropout = getattr(config, 'chronos_adapter_dropout', config.dropout)
            adapter_hidden = max(int(self.chronos_adapter_hidden), 8)
            adapter_in_dim = config.pred_len * 4 + 7
            self.chronos_residual_adapter = nn.Sequential(
                nn.Linear(adapter_in_dim, adapter_hidden),
                nn.LayerNorm(adapter_hidden),
                nn.GELU(),
                nn.Dropout(adapter_dropout),
                nn.Linear(adapter_hidden, config.pred_len)
            )
            adapter_out = self.chronos_residual_adapter[-1]
            nn.init.zeros_(adapter_out.weight)
            nn.init.zeros_(adapter_out.bias)

        if self.use_pa_dsc:
            self.pa_dsc = PhysicsAwareDistributionShiftCorrection(config)

        # Final fusion layer(s)
        if self.use_quantile_forecast:
            self.quantile_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(config.pred_len * 2, config.pred_len),
                    nn.GELU(),
                    nn.Dropout(config.dropout)
                )
                for _ in self.quantiles
            ])
        else:
            self.fusion_layer = nn.Sequential(
                nn.Linear(config.pred_len * 2, config.pred_len),
                nn.GELU(),
                nn.Dropout(config.dropout)
            )
        
        # Memory-related modules
        self.local_memory_mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.d_model)
        )
        
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=4,
            dropout=config.dropout,
            batch_first=True
        )
        
    def _physical_power(self, x_enc):
        target_power = x_enc[:, :, -1]
        target_mean = getattr(self.config, 'adaptive_pv_target_mean', None)
        target_std = getattr(self.config, 'adaptive_pv_target_std', None)
        if target_mean is None or target_std is None:
            return target_power

        target_mean = torch.as_tensor(target_mean, dtype=target_power.dtype, device=target_power.device)
        target_std = torch.as_tensor(target_std, dtype=target_power.dtype, device=target_power.device)
        return target_power * target_std + target_mean

    def _extract_day_bucket(self, x_mark_enc):
        if x_mark_enc is None or x_mark_enc.size(-1) < 2:
            return None

        marks = x_mark_enc.detach()
        if marks.size(-1) >= 5 and marks[:, :, 1].amin() >= -0.6 and marks[:, :, 1].amax() <= 0.6:
            hours = (marks[:, :, 1] + 0.5) * 23.0
        elif marks.size(-1) >= 4:
            hours = marks[:, :, 3]
        else:
            return None

        last_hour = hours[:, -1].clamp(0, 23.999)
        bucket_width = 24.0 / max(self.day_bucket_count, 1)
        return torch.div(last_hour, bucket_width, rounding_mode='floor').long().clamp(
            0,
            max(self.day_bucket_count - 1, 0)
        )

    def _build_pa_ral_metadata(self, x_raw, x_mark_enc, n_vars):
        power = self._physical_power(x_raw)
        mean_power = power.mean(dim=1)
        max_power = power.max(dim=1).values
        ramp_strength = power.diff(dim=1).abs().max(dim=1).values if power.size(1) > 1 else torch.zeros_like(mean_power)

        night_threshold = getattr(self.config, 'adaptive_pv_night_threshold', 0.0)
        peak_threshold = getattr(self.config, 'adaptive_pv_peak_threshold', None)
        if peak_threshold is None:
            non_night = power[power > night_threshold]
            peak_threshold = non_night.quantile(0.9) if non_night.numel() > 0 else power.reshape(-1).quantile(0.9)
        peak_threshold = torch.as_tensor(peak_threshold, dtype=power.dtype, device=power.device)

        non_night_ramps = ramp_strength[mean_power > night_threshold]
        ramp_threshold = getattr(self.config, 'adaptive_pv_ramp_threshold', None)
        if ramp_threshold is None:
            ramp_threshold = non_night_ramps.quantile(0.8) if non_night_ramps.numel() > 0 else ramp_strength.quantile(0.8)
        ramp_threshold = torch.as_tensor(ramp_threshold, dtype=power.dtype, device=power.device)

        pv_states = torch.ones_like(mean_power, dtype=torch.long)
        pv_states = torch.where(mean_power <= night_threshold, torch.zeros_like(pv_states), pv_states)
        pv_states = torch.where(max_power > peak_threshold, torch.full_like(pv_states, 2), pv_states)
        ramp_mask = (ramp_strength >= ramp_threshold) & (mean_power > night_threshold) & (max_power <= peak_threshold)
        pv_states = torch.where(ramp_mask, torch.full_like(pv_states, 3), pv_states)

        day_buckets = self._extract_day_bucket(x_mark_enc)
        if day_buckets is None:
            day_buckets = torch.zeros_like(pv_states)

        return {
            'power_levels': mean_power.repeat_interleave(n_vars).detach(),
            'pv_states': pv_states.repeat_interleave(n_vars).detach(),
            'day_buckets': day_buckets.to(x_raw.device).repeat_interleave(n_vars).detach(),
        }

    def _build_analog_trajectories(self, x_enc, n_vars):
        if x_enc.size(1) >= self.config.pred_len:
            trajectories = x_enc[:, -self.config.pred_len:, :]
        else:
            pad_len = self.config.pred_len - x_enc.size(1)
            pad = x_enc[:, -1:, :].expand(-1, pad_len, -1)
            trajectories = torch.cat([x_enc, pad], dim=1)
        return einops.rearrange(trajectories, 'b t n -> (b n) t', n=n_vars).detach()

    def _metadata_gate_features(self, metadata, B, n_vars):
        if metadata is None:
            return torch.zeros(B, n_vars, 6, device=self.device)

        pv_states = metadata['pv_states'].reshape(B, n_vars).long().clamp(0, 3)
        state_one_hot = torch.nn.functional.one_hot(pv_states, num_classes=4).to(dtype=torch.float32)
        power_levels = metadata['power_levels'].reshape(B, n_vars, 1).to(dtype=torch.float32)
        positive_power = power_levels.detach()[power_levels.detach() > 0]
        if positive_power.numel() > 0:
            power_scale = positive_power.quantile(0.75).clamp_min(1.0)
        else:
            power_scale = power_levels.detach().abs().mean().clamp_min(1.0)
        power_levels = (power_levels / power_scale).clamp(0.0, 5.0)
        day_buckets = metadata['day_buckets'].reshape(B, n_vars, 1).to(dtype=torch.float32)
        day_buckets = day_buckets / max(self.day_bucket_count - 1, 1)
        return torch.cat([state_one_hot, power_levels, day_buckets], dim=-1).to(self.device)

    def _chronos_night_mask(self, metadata, B, n_vars):
        if metadata is None:
            return torch.zeros(B, n_vars, 1, dtype=torch.bool, device=self.device)

        pv_states = metadata['pv_states'].reshape(B, n_vars).long().clamp(0, 3)
        return pv_states.eq(0).unsqueeze(-1).to(self.device)

    def _clip_chronos_prior(self, chronos_prior, means, stdev, metadata, B, n_vars, target_only=False):
        if means is None or stdev is None:
            return chronos_prior

        target_mean = means[:, :, -1:].to(chronos_prior.device)
        target_stdev = stdev[:, :, -1:].to(chronos_prior.device).clamp_min(1e-6)
        zero_norm = (0.0 - target_mean) / target_stdev
        capacity_norm = (self.chronos_capacity - target_mean) / target_stdev

        clipped = chronos_prior.clamp(
            min=zero_norm.expand(-1, chronos_prior.size(1), -1),
            max=capacity_norm.expand(-1, chronos_prior.size(1), -1),
        )
        if metadata is not None:
            pv_states = metadata['pv_states'].reshape(B, n_vars).long()
            if target_only or chronos_prior.size(1) == 1:
                target_states = pv_states[:, -1:].to(chronos_prior.device)
                night_mask = target_states.eq(0).view(B, 1, 1)
            else:
                target_states = pv_states.to(chronos_prior.device)
                night_mask = target_states.eq(0).view(B, chronos_prior.size(1), 1)
            clipped = torch.where(
                night_mask,
                zero_norm.expand(-1, chronos_prior.size(1), -1),
                clipped
            )
        return clipped

    def _chronos_residual_delta(
        self,
        chronos_prior,
        base_prediction,
        x_enc,
        metadata,
        B,
        n_vars,
    ):
        if not self.use_chronos_adapter:
            return torch.zeros_like(base_prediction)

        residual_prior = chronos_prior - base_prediction.detach()
        state_features = self._metadata_gate_features(metadata, B, n_vars)[:, -1:, :]
        recent_power = x_enc[:, -1:, -1:].permute(0, 2, 1).expand_as(residual_prior).detach()
        recent_window = x_enc[:, -min(16, x_enc.size(1)):, -1:]
        recent_std = recent_window.std(dim=1, unbiased=False, keepdim=False)
        recent_std = recent_std.unsqueeze(1).detach()
        adapter_input = torch.cat(
            [
                base_prediction.detach(),
                chronos_prior.detach(),
                residual_prior,
                recent_power,
                state_features,
                recent_std,
            ],
            dim=-1,
        )
        delta = self.chronos_residual_adapter(adapter_input)
        night_mask = self._chronos_night_mask(metadata, B, n_vars)[:, -1:, :]
        return delta.masked_fill(night_mask, 0.0)

    def _compute_local_memory(self, patches, metadata=None):
        """Compute local memory by retrieving and fusing similar patches"""
        # Retrieve similar patches from memory bank
        retrieval_metadata = metadata if self.use_pa_ral else None
        retrieval_weights_adaptive = torch.softmax(self.pa_ral_retrieval_logits, dim=0)
        power_scale = 1.0
        if retrieval_metadata is not None and retrieval_metadata.get('power_levels') is not None:
            power_levels = retrieval_metadata['power_levels'].to(patches.device)
            power_scale = power_levels.detach().std(unbiased=False).clamp_min(1.0).item()
        retrieved_patches, _, retrieved_trajectories, retrieval_weights = self.patch_memory_bank.retrieve(
            patches,
            top_k=self.config.top_k,
            metadata=retrieval_metadata,
            shape_weight=retrieval_weights_adaptive[0],
            power_weight=retrieval_weights_adaptive[1],
            state_weight=retrieval_weights_adaptive[2],
            bucket_weight=retrieval_weights_adaptive[3],
            power_scale=power_scale,
            day_buckets=self.day_bucket_count,
        )
        
        # Process retrieved patches with local MLP
        local_memory = self.local_memory_mlp(retrieved_patches)
        
        # Average over retrieved patches
        local_memory = local_memory.mean(dim=1, keepdim=True)
        
        # Residual connection with original patches
        local_memory = local_memory + patches

        analog_prior = (retrieved_trajectories * retrieval_weights.unsqueeze(-1)).sum(dim=1)
        if retrieval_weights.size(1) > 1:
            uniform_confidence = 1.0 / retrieval_weights.size(1)
            max_confidence = retrieval_weights.max(dim=1, keepdim=True).values
            analog_reliability = (max_confidence - uniform_confidence) / (1.0 - uniform_confidence)
            analog_reliability = analog_reliability.clamp(0.0, 1.0)
        else:
            analog_reliability = retrieval_weights.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        return local_memory, analog_prior, analog_reliability

    def _compute_global_memory(self, patches, metadata=None):
        """Compute global memory by aggregating information across all patches"""
        # Self-attention to capture global dependencies
        attn_output, _ = self.memory_attention(
            query=patches,
            key=patches,
            value=patches
        )
        
        # Update patch memory bank with current patches
        update_metadata = metadata if self.use_pa_ral else None
        self.patch_memory_bank.update(patches.detach(), update_metadata)
        
        if self.use_mem_gate:
            return attn_output  # Return full attention output for advanced gating
        else:
            # Return global context for simple gating (original behavior)
            return attn_output.mean(dim=1, keepdim=True)

    def forward_prediction(
        self,
        x_enc,
        pa_ral_metadata=None,
        chronos_prior=None,
        means=None,
        stdev=None,
        x_mark_enc=None,
        x_mark_dec=None,
    ):
        B, L, n_vars = x_enc.shape
        
        # 1. Process temporal features
        patches, _ = self.patch_embedding(x_enc.transpose(1, 2))  # [B * n_vars, n_patches, d_model]
        
        # 2. Compute local and global memory
        local_memory, analog_prior, analog_reliability = self._compute_local_memory(patches, pa_ral_metadata)  # [B * n_vars, n_patches, d_model]
        global_memory = self._compute_global_memory(patches, pa_ral_metadata)  # [B * n_vars, n_patches, d_model] or [B * n_vars, 1, d_model]
        
        # 3. Combine local and global memory
        if self.use_mem_gate:
            # Advanced memory fusion with gating
            combined_features = torch.cat([local_memory, global_memory], dim=-1)  # [B * n_vars, n_patches, d_model*2]
            gate_weights = self.memory_fusion_gate(combined_features)  # [B * n_vars, n_patches, 2]
            
            # Weighted fusion
            memory_features = (
                gate_weights[:, :, 0:1] * local_memory +
                gate_weights[:, :, 1:2] * global_memory
            )  # [B * n_vars, n_patches, d_model]
        else:
            # Simple addition (original behavior)
            memory_features = local_memory + global_memory  # [B * n_vars, n_patches, d_model]

        # 4. Get temporal predictions
        memory_features = self.flatten(memory_features)  # [B * n_vars, head_nf]
        temporal_features = self.temporal_head(memory_features)  # [B, n_vars, d_model]
        memory_features = self.memory_head(memory_features)  # [B * n_vars, pred_len]
        temporal_features = einops.rearrange(temporal_features, '(b n) d -> b n d', b=B, n=n_vars)  # [B, n_vars, d_model]
        memory_features = einops.rearrange(memory_features, '(b n) d -> b n d', b=B, n=n_vars)  # [B, n_vars, pred_len]

        if self.use_pa_ral_analog:
            analog_prior = einops.rearrange(analog_prior, '(b n) t -> b n t', b=B, n=n_vars)
            analog_reliability = einops.rearrange(analog_reliability, '(b n) r -> b n r', b=B, n=n_vars)
            current_anchor = x_enc[:, -1:, :].permute(0, 2, 1)
            analog_prior = current_anchor + (analog_prior - analog_prior[:, :, :1])
            analog_prior = analog_prior.clamp(-5.0, 5.0)

            analog_context = torch.cat(
                [
                    memory_features,
                    analog_prior,
                    analog_reliability,
                    self._metadata_gate_features(pa_ral_metadata, B, n_vars),
                ],
                dim=-1,
            )
            analog_mix = self.analog_fusion_gate(analog_context) * analog_reliability
            memory_features = (
                (1.0 - analog_mix) * memory_features +
                analog_mix * analog_prior
            )

        if chronos_prior is not None:
            chronos_prior = self._clip_chronos_prior(
                chronos_prior,
                means,
                stdev,
                pa_ral_metadata,
                B,
                n_vars,
                target_only=True,
            )
            chronos_prior = chronos_prior.permute(0, 2, 1).contiguous()

        # PA-RAL is the base temporal prediction; Chronos adapter learns only the residual.
        predictions = memory_features  # [B, n_vars, pred_len]
        if chronos_prior is not None:
            if chronos_prior.size(1) == 1 and n_vars > 1:
                base_target = predictions[:, -1:, :]
                chrono_delta = self._chronos_residual_delta(
                    chronos_prior,
                    base_target,
                    x_enc,
                    pa_ral_metadata,
                    B,
                    n_vars,
                )
                corrected_target = base_target + chrono_delta
                predictions = predictions.clone()
                predictions[:, -1:, :] = corrected_target
            else:
                if chronos_prior.size(1) != n_vars:
                    raise ValueError(
                        f"Chronos prior variable dimension {chronos_prior.size(1)} "
                        f"does not match model variables {n_vars}."
                    )
                base_target = predictions[:, -1:, :]
                chronos_target = chronos_prior[:, -1:, :]
                chrono_delta = self._chronos_residual_delta(
                    chronos_target,
                    base_target,
                    x_enc,
                    pa_ral_metadata,
                    B,
                    n_vars,
                )
                predictions = predictions.clone()
                predictions[:, -1:, :] = base_target + chrono_delta
        predictions = predictions.permute(0, 2, 1)  # [B, pred_len, n_vars]
        if self.use_pa_dsc:
            predictions = self.pa_dsc(
                predictions,
                x_enc,
                x_mark_enc=x_mark_enc,
                x_mark_dec=x_mark_dec,
                metadata=pa_ral_metadata,
                means=means,
                stdev=stdev,
            )
        if self.use_quantile_forecast:
            corrected_center = predictions.permute(0, 2, 1).contiguous()  # [B, n_vars, pred_len]
            quantile_context = torch.cat([corrected_center, memory_features], dim=-1)
            quantile_outputs = [
                head(quantile_context) + corrected_center for head in self.quantile_heads
            ]
            quantile_outputs = torch.stack(quantile_outputs, dim=-1)  # [B, n_vars, pred_len, n_quantiles]
            quantile_outputs, _ = torch.sort(quantile_outputs, dim=-1)
            quantile_outputs = quantile_outputs.permute(0, 2, 1, 3).contiguous()
            point_prediction = quantile_outputs[..., self.median_quantile_idx]
            return {
                'point': point_prediction,
                'quantiles': quantile_outputs,
                'quantile_levels': self.quantiles,
            }
        return predictions

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None, chronos_prior=None):
        B, L, D = x_enc.shape
        x_enc = x_enc.to(self.device)
        x_mark_enc = x_mark_enc.to(self.device) if x_mark_enc is not None else None
        pa_ral_metadata = self._build_pa_ral_metadata(x_enc, x_mark_enc, D)
        
        # Normalize input
        x_enc, means, stdev = self._normalize_input(x_enc)
        pa_ral_metadata['analog_trajectories'] = self._build_analog_trajectories(x_enc, D)
        
        if chronos_prior is not None:
            chronos_prior = chronos_prior.to(self.device)
        elif self.temporal_prior_branch is not None:
            chronos_prior = self.temporal_prior_branch(x_enc, self.config.pred_len)
        
        # Main prediction branch
        predictions = self.forward_prediction(
            x_enc,
            pa_ral_metadata,
            chronos_prior=chronos_prior,
            means=means,
            stdev=stdev,
            x_mark_enc=x_mark_enc,
            x_mark_dec=x_mark_dec,
        )
        
        # Denormalize output
        y = self._denormalize_output(predictions, means, stdev)
        return y

    def _normalize_input(self, x):
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        stdev /= self.config.norm_const
        x = x / stdev
        return x, means, stdev

    def _denormalize_output(self, y, means, stdev):
        if isinstance(y, dict):
            point = y['point'] * (stdev.repeat(1, self.config.pred_len, 1))
            point = point + (means.repeat(1, self.config.pred_len, 1))
            quantiles = y['quantiles'] * stdev.repeat(1, self.config.pred_len, 1).unsqueeze(-1)
            quantiles = quantiles + means.repeat(1, self.config.pred_len, 1).unsqueeze(-1)
            return {
                'point': point,
                'quantiles': quantiles,
                'quantile_levels': y['quantile_levels'],
            }
        y = y * (stdev.repeat(1, self.config.pred_len, 1))
        y = y + (means.repeat(1, self.config.pred_len, 1))
        return y
