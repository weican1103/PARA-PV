from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import point_metric, probability_metric, efficiency_metric
from utils.losses import point_loss_from_args, quantile_pv_loss_from_args
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.augmentation import run_augmentation,run_augmentation_single

warnings.filterwarnings('ignore')

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.train_epoch_times = []
        self.avg_train_epoch_time = np.nan
        self.chronos_prior_cache = {}

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _forward_model(self, batch_x, batch_x_mark, dec_inp, batch_y_mark, flag=None, batch_indices=None):
        chronos_prior = self._batch_chronos_prior(flag, batch_indices) if flag is not None else None
        if chronos_prior is None:
            return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        return self.model(
            batch_x,
            batch_x_mark,
            dec_inp,
            batch_y_mark,
            chronos_prior=chronos_prior,
        )

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if getattr(self.args, 'forecast_mode', 'point') == 'prob':
            return quantile_pv_loss_from_args(self.args)
        return point_loss_from_args(self.args)

    def _set_adaptive_pv_loss_stats(self, train_data):
        if not hasattr(train_data, 'target_mean'):
            return
        self.args.adaptive_pv_target_mean = train_data.target_mean
        self.args.adaptive_pv_target_std = train_data.target_std
        self.args.adaptive_pv_night_threshold = train_data.adaptive_pv_night_threshold
        self.args.adaptive_pv_peak_threshold = train_data.adaptive_pv_peak_threshold
        self.args.adaptive_pv_ramp_threshold = train_data.adaptive_pv_ramp_threshold

    def _unpack_batch(self, batch):
        if len(batch) == 5:
            batch_x, batch_y, batch_x_mark, batch_y_mark, batch_indices = batch
        else:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            batch_indices = None
        return batch_x, batch_y, batch_x_mark, batch_y_mark, batch_indices

    @staticmethod
    def _flag_from_dataset(data_set):
        return {0: 'train', 1: 'val', 2: 'test'}[data_set.set_type]

    def _load_chronos_prior_cache(self, flag):
        if not getattr(self.args, 'use_chronos_prior_cache', False):
            return None
        if flag not in self.chronos_prior_cache:
            cache_path = os.path.join(
                self.args.chronos_prior_cache_dir,
                f'chronos_prior_{flag}.npy',
            )
            if not os.path.exists(cache_path):
                raise FileNotFoundError(
                    f"Chronos prior cache not found for {flag}: {cache_path}. "
                    "Run scripts/precompute_chronos_priors.py first."
                )
            self.chronos_prior_cache[flag] = np.load(cache_path, mmap_mode='r')
        return self.chronos_prior_cache[flag]

    def _batch_chronos_prior(self, flag, batch_indices):
        cache = self._load_chronos_prior_cache(flag)
        if cache is None:
            return None
        if batch_indices is None:
            raise ValueError("Chronos prior cache requires dataset indices in each batch.")
        indices = batch_indices.detach().cpu().numpy().astype(np.int64)
        prior = np.asarray(cache[indices], dtype=np.float32)
        return torch.as_tensor(prior, dtype=torch.float32, device=self.device)

    def _slice_model_outputs(self, outputs, f_dim):
        if isinstance(outputs, dict):
            return {
                'point': outputs['point'][:, -self.args.pred_len:, f_dim:],
                'quantiles': outputs['quantiles'][:, -self.args.pred_len:, f_dim:, :],
                'quantile_levels': outputs['quantile_levels'],
            }
        return outputs[:, -self.args.pred_len:, f_dim:]

    @staticmethod
    def _point_output(outputs):
        if isinstance(outputs, dict):
            return outputs['point']
        return outputs

    def _peak_mask(self, trues, data_set):
        if getattr(data_set, 'scale', False) and not getattr(self.args, 'inverse', False):
            trues_physical = trues * data_set.target_std + data_set.target_mean
        else:
            trues_physical = trues
        peak_threshold = getattr(data_set, 'adaptive_pv_peak_threshold', None)
        if peak_threshold is None:
            non_night = trues_physical[
                trues_physical > getattr(self.args, 'adaptive_pv_night_threshold', 0.0)
            ]
            if non_night.size == 0:
                non_night = trues_physical.reshape(-1)
            peak_threshold = np.quantile(non_night, 0.9)
        return trues_physical > peak_threshold, float(peak_threshold)

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        flag = self._flag_from_dataset(vali_data)
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_indices = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._forward_model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark, flag, batch_indices
                        )
                else:
                    outputs = self._forward_model(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark, flag, batch_indices
                    )
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = self._slice_model_outputs(outputs, f_dim)
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = self._point_output(outputs).detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(outputs, batch_y)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        self._set_adaptive_pv_loss_stats(train_data)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_indices = self._unpack_batch(batch)
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._forward_model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark, 'train', batch_indices
                        )

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = self._slice_model_outputs(outputs, f_dim)
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self._forward_model(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark, 'train', batch_indices
                    )

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = self._slice_model_outputs(outputs, f_dim)
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            epoch_cost = time.time() - epoch_time
            self.train_epoch_times.append(epoch_cost)
            self.avg_train_epoch_time = float(np.mean(self.train_epoch_times))
            print("Epoch: {} cost time: {}".format(epoch + 1, epoch_cost))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location=self.device))

        preds = []
        trues = []
        quantile_preds = []
        quantile_levels = None
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        inference_time = 0.0
        inference_steps = 0
        if self.args.use_gpu and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_indices = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_gpu and torch.cuda.is_available():
                    torch.cuda.synchronize(self.device)
                infer_start = time.time()
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._forward_model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark, 'test', batch_indices
                        )
                else:
                    outputs = self._forward_model(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark, 'test', batch_indices
                    )
                if self.args.use_gpu and torch.cuda.is_available():
                    torch.cuda.synchronize(self.device)
                inference_time += time.time() - infer_start
                inference_steps += batch_x.shape[0] * self.args.pred_len

                f_dim = -1 if self.args.features == 'MS' else 0
                if isinstance(outputs, dict):
                    quantile_levels = outputs['quantile_levels']
                    point_outputs = outputs['point'][:, -self.args.pred_len:, :]
                    quantile_outputs = outputs['quantiles'][:, -self.args.pred_len:, :, :]
                else:
                    point_outputs = outputs[:, -self.args.pred_len:, :]
                    quantile_outputs = None
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                point_outputs = point_outputs.detach().cpu().numpy()
                if quantile_outputs is not None:
                    quantile_outputs = quantile_outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = point_outputs.shape
                    point_outputs = test_data.inverse_transform(point_outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    if quantile_outputs is not None:
                        q_shape = quantile_outputs.shape
                        quantile_outputs = quantile_outputs.transpose(0, 1, 3, 2).reshape(-1, q_shape[2])
                        quantile_outputs = test_data.inverse_transform(quantile_outputs)
                        quantile_outputs = quantile_outputs.reshape(q_shape[0], q_shape[1], q_shape[3], q_shape[2]).transpose(0, 1, 3, 2)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
        
                outputs = point_outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]
                if quantile_outputs is not None:
                    quantile_outputs = quantile_outputs[:, :, f_dim:, :]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if quantile_outputs is not None:
                    quantile_preds.append(quantile_outputs)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        if quantile_preds:
            quantile_preds = np.concatenate(quantile_preds, axis=0)
        else:
            quantile_preds = None
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        if quantile_preds is not None:
            quantile_preds = quantile_preds.reshape(
                -1, quantile_preds.shape[-3], quantile_preds.shape[-2], quantile_preds.shape[-1]
            )
        print('test shape:', preds.shape, trues.shape)

        # result save
        result_setting = setting
        folder_path = './results/' + result_setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        train_time = float(getattr(self, 'avg_train_epoch_time', np.nan))
        inference_latency = inference_time / inference_steps if inference_steps > 0 else np.nan
        learnable_params = getattr(self, 'learnable_params', None)
        if learnable_params is None:
            learnable_params = self._count_learnable_parameters(self.model)
        params_k = float(learnable_params) / 1000.0
        if self.args.use_gpu and torch.cuda.is_available():
            memory_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        else:
            memory_mb = np.nan
        efficiency_metrics = efficiency_metric(train_time, inference_latency, params_k, memory_mb)
        print(
            't_train: {:.4f}s/epoch, t_inf: {:.8f}s/step, params: {:.2f}K, mem: {:.2f}MB'.format(
                train_time, inference_latency, params_k, memory_mb
            )
        )

        forecast_mode = getattr(self.args, 'forecast_mode', 'point')
        point_metrics = None
        peak_metrics = None
        prob_metrics = None
        if forecast_mode == 'prob':
            if quantile_preds is None:
                raise ValueError("Probability forecasting requires quantile predictions.")
            picp_80, pinaw_80, picp_90, pinaw_90, aql = probability_metric(
                quantile_preds, trues, quantile_levels
            )
            prob_metrics = np.array([picp_80, pinaw_80, picp_90, pinaw_90, aql])
            print('picp_80: {:.4f}, pinaw_80: {:.4f}, picp_90: {:.4f}, pinaw_90: {:.4f}, aql: {:.4f}'.format(
                picp_80, pinaw_80, picp_90, pinaw_90, aql
            ))
        else:
            peak_mask, _ = self._peak_mask(trues, test_data)
            mae, mse, rmse, r2, peak_mae, peak_mse = point_metric(preds, trues, peak_mask)
            point_metrics = np.array([mae, mse, rmse, r2])
            peak_metrics = np.array([peak_mae, peak_mse])
            print('mse: {:.4f}, mae: {:.4f}, rmse: {:.4f}, r2: {:.4f}'.format(mse, mae, rmse, r2))
            print('peak_mse: {:.4f}, peak_mae: {:.4f}'.format(
                peak_mse, peak_mae
            ))

        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        if forecast_mode == 'prob':
            f.write('picp_80: {:.4f}, pinaw_80: {:.4f}, picp_90: {:.4f}, pinaw_90: {:.4f}, aql: {:.4f}'.format(
                prob_metrics[0], prob_metrics[1], prob_metrics[2], prob_metrics[3], prob_metrics[4]
            ))
        else:
            f.write('mse: {:.4f}, mae: {:.4f}, rmse: {:.4f}, r2: {:.4f}'.format(
                point_metrics[1], point_metrics[0], point_metrics[2], point_metrics[3]
            ))
            f.write('\n')
            f.write('peak_mse: {:.4f}, peak_mae: {:.4f}'.format(
                peak_metrics[1], peak_metrics[0]
            ))
        f.write('\n')
        f.write('t_train: {:.4f}s/epoch, t_inf: {:.8f}s/step, params: {:.2f}K, mem: {:.2f}MB'.format(
            efficiency_metrics[0], efficiency_metrics[1], efficiency_metrics[2], efficiency_metrics[3]
        ))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'efficiency_metrics.npy', efficiency_metrics)
        if forecast_mode == 'prob':
            np.save(folder_path + 'metrics.npy', prob_metrics)
        else:
            np.save(folder_path + 'metrics.npy', point_metrics)
            np.save(folder_path + 'peak_metrics.npy', peak_metrics)
        if prob_metrics is not None:
            np.save(folder_path + 'prob_metrics.npy', prob_metrics)
            np.save(folder_path + 'quantile_pred.npy', quantile_preds)
            np.save(folder_path + 'quantile_levels.npy', np.array(quantile_levels))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
