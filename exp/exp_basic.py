import torch


class LazyModelRegistry(dict):
    def __init__(self):
        super().__init__()
        self.module_map = {
            'PARA-PV': 'src.PARA_PV.model',
        }

    def __getitem__(self, model_name):
        if model_name not in self.module_map:
            raise KeyError(f"Unknown model '{model_name}'. Available models: {list(self.module_map.keys())}")
        if model_name not in self:
            module_path = self.module_map[model_name]
            module = __import__(module_path, fromlist=[''])
            self[model_name] = module
        return dict.__getitem__(self, model_name)


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = LazyModelRegistry()

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.learnable_params = self._count_learnable_parameters(self.model)
        self.total_params = self._count_total_parameters(self.model)
        
        if args.is_training:
            self._log_model_parameters()
        
        
    @staticmethod
    def _count_learnable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def _count_total_parameters(model):
        return sum(p.numel() for p in model.parameters())

    def _log_model_parameters(self):
        """
        打印模型参数。
        """
        learable_params = self.learnable_params
        total_params = self.total_params
        print(f"Learnable model parameters: {learable_params:,}")
        print(f"Total model parameters: {total_params:,}")
        

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
