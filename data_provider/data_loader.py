import os
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils.augmentation import run_augmentation_single
from utils.timefeatures import time_features

warnings.filterwarnings('ignore')


PV_DATA_FILE = 'Solarstation1-50MW-2019.xlsx'


def read_pv_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    raise ValueError(f"Unsupported custom PV data file: {file_path}. Expected an Excel file.")


def normalize_pv_columns(df_raw):
    pv_column_map = {
        'Time(year-month-day h:m:s)': 'date',
        'Total solar irradiance (W/m2)': 'total_solar_irradiance',
        'Direct normal irradiance (W/m2)': 'direct_normal_irradiance',
        'Global horizontal irradiance (W/m2)': 'global_horizontal_irradiance',
        'Air temperature  (degC) ': 'air_temperature',
        'Air temperature  (degC)': 'air_temperature',
        'Air temperature  (°C) ': 'air_temperature',
        'Air temperature  (°C)': 'air_temperature',
        'Atmosphere (hpa)': 'atmosphere_pressure',
        'Relative humidity (%)': 'relative_humidity',
        'Power (MW)': 'power',
    }
    df_raw = df_raw.rename(columns={col: pv_column_map.get(col, col) for col in df_raw.columns})
    df_raw.columns = [col.strip() if isinstance(col, str) else col for col in df_raw.columns]
    return df_raw


class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='MS', data_path=PV_DATA_FILE,
                 target='power', scale=True, timeenc=0, freq='15min'):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'test', 'val']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]
        self.args = args
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        file_path = os.path.join(self.root_path, self.data_path)
        df_raw = normalize_pv_columns(read_pv_file(file_path))

        if 'date' not in df_raw.columns:
            raise ValueError(
                "PV custom data requires a date column. Expected 'date' or "
                "'Time(year-month-day h:m:s)'."
            )
        if self.target not in df_raw.columns:
            raise ValueError(
                f"Target column '{self.target}' not found in {self.data_path}. "
                f"Available columns: {list(df_raw.columns)}"
            )

        cols = [col for col in df_raw.columns if col not in ['date', self.target]]
        df_raw = df_raw[['date'] + cols + [self.target]]
        df_raw['date'] = pd.to_datetime(df_raw['date'])

        value_cols = cols + [self.target]
        for col in value_cols:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
        df_raw = df_raw.replace(-99, np.nan)
        df_raw[value_cols] = df_raw[value_cols].interpolate(
            method='linear', limit_direction='both'
        ).fillna(0)

        num_train = int(len(df_raw) * 0.8)
        num_test = int(len(df_raw) * 0.1)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features in ['M', 'MS']:
            df_data = df_raw[df_raw.columns[1:]]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        else:
            raise ValueError(f"Unsupported feature mode for PV custom data: {self.features}")

        target_col_idx = list(df_data.columns).index(self.target)
        train_target_raw = df_data[self.target].iloc[border1s[0]:border2s[0]].to_numpy(dtype=np.float64)
        train_target_raw = train_target_raw[np.isfinite(train_target_raw)]
        if train_target_raw.size == 0:
            train_target_raw = np.array([0.0], dtype=np.float64)

        positive_power = train_target_raw[train_target_raw > 0]
        if positive_power.size > 0:
            low_power_floor = np.quantile(positive_power, 0.01)
            self.adaptive_pv_night_threshold = float(max(low_power_floor, np.finfo(np.float32).eps))
            non_night_power = train_target_raw[train_target_raw > self.adaptive_pv_night_threshold]
            if non_night_power.size == 0:
                non_night_power = positive_power
        else:
            self.adaptive_pv_night_threshold = 0.0
            non_night_power = train_target_raw

        self.adaptive_pv_peak_threshold = float(np.quantile(non_night_power, 0.9))
        train_ramp_raw = np.abs(np.diff(train_target_raw))
        train_ramp_raw = train_ramp_raw[np.isfinite(train_ramp_raw)]
        train_ramp_raw = train_ramp_raw[train_ramp_raw > 0]
        if train_ramp_raw.size > 0:
            self.adaptive_pv_ramp_threshold = float(np.quantile(train_ramp_raw, 0.8))
        else:
            self.adaptive_pv_ramp_threshold = 0.0

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            self.target_mean = float(self.scaler.mean_[target_col_idx])
            self.target_std = float(self.scaler.scale_[target_col_idx])
        else:
            data = df_data.values
            self.target_mean = 0.0
            self.target_std = 1.0

        df_stamp = df_raw[['date']][border1:border2].copy()
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else:
            raise ValueError(f"Unsupported time encoding mode: {self.timeenc}")

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, _ = run_augmentation_single(
                self.data_x, self.data_y, self.args
            )

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if getattr(self.args, 'use_chronos_prior_cache', False):
            return seq_x, seq_y, seq_x_mark, seq_y_mark, index
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
