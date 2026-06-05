import os
import json
import inspect
import torch
from GNNTP.models.locator import get_model_metadata, get_model_resource_path, has_model_resource
from GNNTP.data.registry import get_dataset_class
from GNNTP.common.registry_evaluator import get_evaluator_class
from GNNTP.common.registry_executor import get_executor_class
from GNNTP.utils.paths import PROJECT_ROOT, RESOURCE_DATA_ROOT


class ConfigParser(object):
    """
    use to parse the user defined parameters and use these to modify the
    pipeline's parameter setting.
    值得注意的是，目前各阶段的参数是放置于同一个 dict 中的，因此需要编程时保证命名空间不冲突。
    config 优先级：命令行 > config file > default config
    """

    def __init__(self, task, model, dataset, config_file=None,
                 saved_model=True, train=True, other_args=None, hyper_config_dict=None):
        """
        Args:
            task, model, dataset (str): 用户在命令行必须指明的三个参数
            config_file (str): 配置文件的文件名，将在项目根目录下进行搜索
            other_args (dict): 通过命令行传入的其他参数
        """
        self.config = {}
        self._parse_external_config(task, model, dataset, saved_model, train, other_args, hyper_config_dict)
        self._parse_config_file(config_file)
        self._load_default_config()
        self._init_device()

    def _parse_external_config(self, task, model, dataset,
                               saved_model=True, train=True, other_args=None, hyper_config_dict=None):
        if task is None:
            raise ValueError('the parameter task should not be None!')
        if model is None:
            raise ValueError('the parameter model should not be None!')
        if dataset is None:
            raise ValueError('the parameter dataset should not be None!')
        # 目前暂定这三个参数必须由用户指定
        self.config['task'] = task
        self.config['model'] = model
        self.config['dataset'] = dataset
        self.config['saved_model'] = saved_model
        self.config['train'] = False if task == 'map_matching' else train
        if other_args is not None:
            # TODO: 这里可以设计加入参数检查，哪些参数是允许用户通过命令行修改的
            for key in other_args:
                self.config[key] = other_args[key]
        if hyper_config_dict is not None:
            # 超参数调整时传入的待调整的参数，优先级低于命令行参数
            if hyper_config_dict is not None:
                for key, value in hyper_config_dict.items():
                    if other_args is None or key not in other_args:
                        self.config[key] = value



    def _parse_config_file(self, config_file):
        if config_file is not None:
            # TODO: 对 config file 的格式进行检查
            config_path = PROJECT_ROOT / f"{config_file}.json"
            if config_path.exists():
                with config_path.open('r', encoding="utf-8") as f:
                    x = json.load(f)
                    for key in x:
                        if key not in self.config:
                            self.config[key] = x[key]
            else:
                raise FileNotFoundError(
                    'Config file {}.json is not found. Please ensure \
                    the config file is in the root dir and is a JSON \
                    file.'.format(config_file))

    def _load_default_config(self):
        model_metadata = get_model_metadata(self.config['task'], self.config['model'])
        if 'dataset_class' not in self.config:
            self.config['dataset_class'] = model_metadata['dataset_class']
        if 'executor' not in self.config:
            self.config['executor'] = model_metadata['executor']
        if 'evaluator' not in self.config:
            self.config['evaluator'] = model_metadata['evaluator']
        if self.config['model'].upper() in ['LSTM', 'GRU', 'RNN']:
            self.config['rnn_type'] = self.config['model']
            self.config['model'] = 'RNN'
        # 接着加载每个阶段的 default config
        default_file_list = [
            self._get_model_default_config_path(),
            self._get_dataset_default_config_path(),
            self._get_executor_default_config_path(),
            self._get_evaluator_default_config_path(),
        ]
        # 加载所有默认配置
        for file_name in default_file_list:
            with open(file_name, 'r') as f:
                x = json.load(f)
                for key in x:
                    if key not in self.config:
                        self.config[key] = x[key]
        # 加载数据集config.json
        # with open('./resource_data/{}/config.json'.format(self.config['dataset']), 'r') as f:
        # 不要使用硬编码，而是通过项目根目录下的raw_data
        dataset_config_path = RESOURCE_DATA_ROOT / self.config['dataset'] / 'config.json'
        with dataset_config_path.open('r', encoding="utf-8") as f:
            x = json.load(f)
            for key in x:
                if key == 'info':
                    for ik in x[key]:
                        if ik not in self.config:
                            self.config[ik] = x[key][ik]
                else:
                    if key not in self.config:
                        self.config[key] = x[key]

    def _init_device(self):
        import logging

        logger = logging.getLogger()

        # 检测 DDP 环境（torchrun 自动设置 LOCAL_RANK）
        local_rank_env = os.environ.get('LOCAL_RANK', None)
        is_distributed = local_rank_env is not None

        if is_distributed:
            # ── DDP 模式 ──
            local_rank = int(local_rank_env)
            world_size = int(os.environ.get('WORLD_SIZE', '1'))
            rank = int(os.environ.get('RANK', '0'))
            dist_backend = self.config.get('dist_backend', 'nccl')

            if not torch.cuda.is_available():
                raise RuntimeError("DDP requires CUDA but CUDA is not available")

            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")

            self.config['local_rank'] = local_rank
            self.config['world_size'] = world_size
            self.config['rank'] = rank
            self.config['dist_backend'] = dist_backend
            self.config['is_distributed'] = True

            logger.info(
                "DDP mode: rank=%d/%d, local_rank=%d, device=%s",
                rank, world_size, local_rank, device,
            )

            # 线性缩放学习率
            if self.config.get('scale_lr', True) and 'learning_rate' in self.config:
                orig_lr = self.config['learning_rate']
                scaled_lr = orig_lr * world_size
                self.config['learning_rate'] = scaled_lr
                logger.info("LR scaled: %s → %s (world_size=%d)", orig_lr, scaled_lr, world_size)
        else:
            # ── 单卡 / CPU 模式（向后兼容）──
            self.config['is_distributed'] = False
            self.config['world_size'] = 1
            self.config['rank'] = 0

            use_gpu = self.config.get('gpu', True)
            gpu_id = self.config.get('gpu_id', 0)

            if use_gpu and torch.cuda.is_available():
                if gpu_id >= torch.cuda.device_count():
                    raise ValueError(
                        f"gpu_id {gpu_id} is invalid, only {torch.cuda.device_count()} GPUs available"
                    )
                device = torch.device(f"cuda:{gpu_id}")
                logger.info(f"Using GPU {gpu_id}")
            else:
                if use_gpu:
                    logger.warning("GPU requested but not available, using CPU instead.")
                device = torch.device("cpu")

        self.config['device'] = device

    def _get_model_default_config_path(self):
        if not has_model_resource(self.config['task'], self.config['model'], 'config.json'):
            raise FileNotFoundError(
                'Model config.json is required in model directory for task={}, model={}.'.format(
                    self.config['task'], self.config['model']
                )
            )
        return get_model_resource_path(self.config['task'], self.config['model'], 'config.json')

    def _get_dataset_default_config_path(self):
        # 使用模型 manifest 中声明的 dataset_class 定位默认配置文件，
        # 而非用户通过 --dataset_class 覆盖的值（后者仅用于运行时数据集类选择）。
        model_metadata = get_model_metadata(self.config['task'], self.config['model'])
        manifest_dataset_class = model_metadata['dataset_class']
        dataset_class = get_dataset_class(
            manifest_dataset_class, task=self.config['task'], model_name=self.config['model']
        )
        dataset_module_dir = os.path.dirname(inspect.getfile(dataset_class))
        return os.path.join(dataset_module_dir, '{}.json'.format(manifest_dataset_class))

    def _get_executor_default_config_path(self):
        if has_model_resource(self.config['task'], self.config['model'], 'executor.json'):
            return get_model_resource_path(self.config['task'], self.config['model'], 'executor.json')
        executor_class = get_executor_class(
            self.config['executor'], task=self.config['task'], model_name=self.config['model']
        )
        executor_module_dir = os.path.dirname(inspect.getfile(executor_class))
        return os.path.join(executor_module_dir, '{}.json'.format(self.config['executor']))

    def _get_evaluator_default_config_path(self):
        evaluator_class = get_evaluator_class(
            self.config['evaluator'], task=self.config['task'], model_name=self.config['model']
        )
        evaluator_module_dir = os.path.dirname(inspect.getfile(evaluator_class))
        return os.path.join(evaluator_module_dir, '{}.json'.format(self.config['evaluator']))

    def get(self, key, default=None):
        return self.config.get(key, default)

    def __getitem__(self, key):
        if key in self.config:
            return self.config[key]
        else:
            raise KeyError('{} is not in the config'.format(key))

    def __setitem__(self, key, value):
        self.config[key] = value

    def __contains__(self, key):
        return key in self.config

    # 支持迭代操作
    def __iter__(self):
        return self.config.__iter__()
