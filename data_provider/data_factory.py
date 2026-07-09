from torch.utils.data import DataLoader

from data_provider.data_loader import Dataset_Custom, PV_DATA_FILE


ALLOWED_PV_DATA_FILES = {
    PV_DATA_FILE,
    'Solarstation6-35MW-20192020.xlsx',
}


def data_provider(args, flag):
    if args.data != 'custom':
        raise ValueError("This PV-LLM project is configured to run only the custom PV dataset.")
    if args.data_path not in ALLOWED_PV_DATA_FILES:
        raise ValueError(
            'This PV-LLM project is configured to run only the allowed PV datasets: '
            f'{sorted(ALLOWED_PV_DATA_FILES)}.'
        )

    timeenc = 0 if args.embed != 'timeF' else 1
    shuffle_flag = False if flag in ['test', 'TEST'] else True

    data_set = Dataset_Custom(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=args.freq,
    )

    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=False,
    )
    return data_set, data_loader
