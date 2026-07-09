import argparse
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.print_args import print_args
import random
import numpy as np

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
if __name__ == '__main__':
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='PARA-PV')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast', help='task name')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='Our model', help='model id')
    parser.add_argument('--model', type=str, default='PARA-PV', choices=['PARA-PV'], help='model name')

    # data loader
    parser.add_argument('--data', type=str, default='custom', help='dataset type, fixed to custom PV data')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='Solarstation1-50MW-2019.xlsx', help='custom PV Excel data file')
    parser.add_argument('--features', type=str, default='MS', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='power', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='15min', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=192, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=96, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # PARA-PV model
    parser.add_argument('--top_k', type=int, default=5, help='number of PA-RAL patches to retrieve')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--patch_memory_size', type=int, default=100, choices=[100, 512, 1024, 2048, 4096, 8192], help='PA-RAL patch memory bank size')
    parser.add_argument('--norm_const', type=float, default=0.4, help='input normalization constant')
    parser.add_argument('--stride', type=int, default=8, help='patch stride')
    parser.add_argument('--padding', type=int, default=8, help='patch padding')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--use_mem_gate', type=str2bool, default=False, help='use learned local/global memory fusion gate')
    parser.add_argument('--use_pa_ral', type=str2bool, default=True, help='use PA-RAL physics-aware retrieval branch')
    parser.add_argument('--use_pa_ral_analog', type=str2bool, default=False, help='use retrieved historical analog trajectories in PA-RAL')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--forecast_mode', type=str, default='point', choices=['point', 'prob'], help='point or probabilistic forecasting mode')
    parser.add_argument('--point_train_loss', type=str, default='adaptive_segment_pv', choices=['adaptive_segment_pv', 'adaptivepvmae', 'mse', 'mae', 'huber'], help='point-mode training loss')
    parser.add_argument('--huber_delta', type=float, default=1.0, help='delta for HuberLoss when --point_train_loss huber')
    parser.add_argument('--quantiles', type=str, default='0.05,0.1,0.5,0.9,0.95', help='comma-separated quantile levels for probability mode with fixed AQL loss; must include 0.05,0.1,0.5,0.9,0.95')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    
    # Frozen Chronos prior and physics-aware distribution shift correction
    parser.add_argument('--use_chronos_prior', type=str2bool, default=False, help='use frozen Chronos2 zero-shot temporal forecasting prior branch')
    parser.add_argument('--chronos_model_path', type=str, default='./pretrained_models/chronos-2', help='local path of Chronos2 pretrained model')
    parser.add_argument('--chronos_device_map', type=str, default=None, help='device_map passed to Chronos2Pipeline.from_pretrained')
    parser.add_argument('--chronos_dtype', type=str, default='float32', choices=['float32', 'float16', 'bfloat16'], help='dtype for Chronos2 prior inference')
    parser.add_argument('--chronos_freq', type=str, default='15min', help='synthetic timestamp frequency for Chronos2 predict_df input')
    parser.add_argument('--chronos_quantile', type=str, default='0.5', help='Chronos2 quantile level used as point prior')
    parser.add_argument('--chronos_capacity', type=float, default=50.0, help='PV capacity upper bound in MW for clipping Chronos prior')
    parser.add_argument('--use_chronos_adapter', type=str2bool, default=False, help='freeze Chronos2 and train a small PV calibration adapter on top of its prior')
    parser.add_argument('--chronos_adapter_hidden', type=int, default=96, help='hidden dimension of the Chronos prior calibration adapter')
    parser.add_argument('--chronos_adapter_dropout', type=float, default=0.1, help='dropout used inside the Chronos prior calibration adapter')
    parser.add_argument('--use_chronos_prior_cache', type=str2bool, default=False, help='load precomputed Chronos2 priors from disk instead of running Chronos2 in the training environment')
    parser.add_argument('--chronos_prior_cache_dir', type=str, default='./pretrained_models/chronos-2-priors', help='directory containing chronos_prior_{train,val,test}.npy')
    parser.add_argument('--use_pa_dsc', type=str2bool, default=False, help='use Physics-aware Distribution Shift Correction after Chronos residual adapter')
    parser.add_argument('--pa_dsc_hidden', type=int, default=128, help='hidden dimension for PA-DSC correction module')
    parser.add_argument('--pa_dsc_physics_clip', type=str2bool, default=True, help='clip PA-DSC target prediction to PV physical bounds in normalized space')
    parser.add_argument('--pa_dsc_low_power_margin', type=float, default=0.1, help='margin above normalized zero-power bound for PA-DSC low-power risk clipping')

    # GPU
    parser.add_argument('--gpu', type=int, default=1, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device IDs for multi-GPU training')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2024, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true", help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true", help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")
    
    args = parser.parse_args()
    if args.data != 'custom':
        raise ValueError('This project is configured to run only the custom PV dataset.')
    allowed_data_paths = {
        'Solarstation1-50MW-2019.xlsx',
        'Solarstation6-35MW-20192020.xlsx',
    }
    if args.data_path not in allowed_data_paths:
        raise ValueError(
            'This project is configured to run only the allowed PV datasets: '
            f'{sorted(allowed_data_paths)}.'
        )
    args.use_quantile_forecast = args.forecast_mode == 'prob'
    if args.forecast_mode == 'prob':
        quantile_levels = [float(q.strip()) for q in args.quantiles.split(',') if q.strip()]
        required_quantiles = [0.05, 0.1, 0.5, 0.9, 0.95]
        missing = [
            q for q in required_quantiles
            if not any(abs(q - existing) < 1e-6 for existing in quantile_levels)
        ]
        if missing:
            raise ValueError(
                '--forecast_mode prob requires quantiles 0.05,0.1,0.5,0.9,0.95 '
                f'for PICP/PINAW 80 and 90 metrics. Missing: {missing}'
            )
    args.use_gpu = True if torch.cuda.is_available() else False

    # set gpu id
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # print arguments
    print_args(args)

    if args.task_name != 'long_term_forecast':
        raise ValueError('This PV-LLM setup is configured to run only long_term_forecast.')
    Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        exp = Exp(args)
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            ii)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        
        torch.cuda.empty_cache()
