import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class TemporalForecastingPriorBranch(nn.Module):
    """Frozen Chronos2 forecasting prior branch."""

    def __init__(self, config):
        super().__init__()
        self.pred_len = config.pred_len
        self.model_path = getattr(
            config,
            'chronos_model_path',
            os.path.abspath(os.path.join(os.getcwd(), 'pretrained_models', 'chronos-2')),
        )
        self.device_map = getattr(config, 'chronos_device_map', None)
        if self.device_map is None:
            self.device_map = f"cuda:{config.gpu}" if getattr(config, 'use_gpu', True) else "cpu"
        self.dtype = self._resolve_dtype(getattr(config, 'chronos_dtype', 'float32'))
        self.freq = getattr(config, 'chronos_freq', '15min')
        self.quantile = str(getattr(config, 'chronos_quantile', '0.5'))

        try:
            from chronos import Chronos2Pipeline
        except ImportError as exc:
            raise ImportError(
                "Chronos prior requires the chronos-forecasting package. "
                "Install it with: pip install 'chronos-forecasting>=2.0'"
            ) from exc

        try:
            self.pipeline = Chronos2Pipeline.from_pretrained(
                self.model_path,
                device_map=self.device_map,
                dtype=self.dtype,
            )
        except TypeError as exc:
            if "dtype" not in str(exc):
                raise
            self.pipeline = Chronos2Pipeline.from_pretrained(
                self.model_path,
                device_map=self.device_map,
                torch_dtype=self.dtype,
            )
        model = getattr(self.pipeline, 'model', None)
        if model is not None:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

    @staticmethod
    def _resolve_dtype(dtype_name):
        if isinstance(dtype_name, torch.dtype):
            return dtype_name
        dtype_name = str(dtype_name).lower()
        if dtype_name in {'float16', 'fp16', 'half'}:
            return torch.float16
        if dtype_name in {'bfloat16', 'bf16'}:
            return torch.bfloat16
        return torch.float32

    def _build_context_df(self, values):
        rows = []
        bsz, seq_len, n_vars = values.shape
        timestamps = pd.date_range("2000-01-01", periods=seq_len, freq=self.freq)
        for batch_idx in range(bsz):
            series_id = f"b{batch_idx}"
            target_series = values[batch_idx, :, -1]
            covariates = values[batch_idx, :, :-1]
            for step_idx in range(seq_len):
                row = {
                    "id": series_id,
                    "timestamp": timestamps[step_idx],
                    "target": float(target_series[step_idx]),
                }
                for cov_idx in range(max(n_vars - 1, 0)):
                    row[f"cov_{cov_idx}"] = float(covariates[step_idx, cov_idx])
                rows.append(row)
        return pd.DataFrame(rows)

    def _extract_point_forecast(self, pred_df, bsz, pred_len):
        candidate_columns = [
            self.quantile,
            float(self.quantile) if self.quantile.replace('.', '', 1).isdigit() else self.quantile,
            "0.5",
            0.5,
            "mean",
            "target",
            "prediction",
        ]
        value_column = next((col for col in candidate_columns if col in pred_df.columns), None)
        if value_column is None:
            numeric_columns = [
                col for col in pred_df.columns
                if col not in {"id", "timestamp"} and np.issubdtype(pred_df[col].dtype, np.number)
            ]
            if not numeric_columns:
                raise ValueError(f"Cannot find a numeric Chronos forecast column in {list(pred_df.columns)}")
            value_column = numeric_columns[len(numeric_columns) // 2]

        prior = np.zeros((bsz, pred_len, 1), dtype=np.float32)
        for batch_idx in range(bsz):
            series_id = f"b{batch_idx}"
            series_pred = pred_df[pred_df["id"] == series_id].sort_values("timestamp")
            values = series_pred[value_column].to_numpy(dtype=np.float32)
            if values.size < pred_len:
                pad_value = values[-1] if values.size else 0.0
                values = np.pad(values, (0, pred_len - values.size), constant_values=pad_value)
            prior[batch_idx, :, 0] = values[:pred_len]
        return prior

    @torch.no_grad()
    def forward(self, x_enc, pred_len=None):
        pred_len = pred_len or self.pred_len
        source_device = x_enc.device
        values = x_enc.detach().float().cpu().numpy()
        bsz, _, _ = values.shape

        context_df = self._build_context_df(values)
        pred_df = self.pipeline.predict_df(
            context_df,
            prediction_length=pred_len,
            quantile_levels=[float(self.quantile)],
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )
        prior = self._extract_point_forecast(pred_df, bsz, pred_len)
        return torch.as_tensor(prior, dtype=x_enc.dtype, device=source_device)
