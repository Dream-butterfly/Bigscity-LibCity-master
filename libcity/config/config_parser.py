import os
import json
import torch


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
            for key in hyper_config_dict:
                self.config[key] = hyper_config_dict[key]

    def _parse_config_file(self, config_file):
        if config_file is not None:
            # TODO: 对 config file 的格式进行检查
            if os.path.exists('./{}.json'.format(config_file)):
                with open('./{}.json'.format(config_file), 'r') as f:
                    x = json.load(f)
                    for key in x:
                        if key not in self.config:
                            self.config[key] = x[key]
            else:
                raise FileNotFoundError(
                    'Config file {}.json is not found. Please ensure \
                    the config file is in the root dir and is a JSON \
                    file.'.format(config_file))

    def _load_json_file(self, relative_path):
        with open(os.path.join('./libcity/config', relative_path), 'r') as f:
            return json.load(f)

    def _load_stage_defaults(self, relative_path):
        candidates = []
        if relative_path.startswith('model/'):
            _, task_name, _ = relative_path.split('/', 2)
            candidates.extend([
                'model/_base.json',
                f'model/{task_name}/_base.json',
                relative_path,
            ])
        elif relative_path.startswith('data/'):
            candidates.extend([
                'data/_base.json',
                relative_path,
            ])
        elif relative_path.startswith('executor/'):
            candidates.extend([
                'executor/_base.json',
                relative_path,
            ])
        elif relative_path.startswith('evaluator/'):
            candidates.extend([
                'evaluator/_base.json',
                relative_path,
            ])
        else:
            candidates.append(relative_path)

        merged = {}
        for candidate in candidates:
            candidate_path = os.path.join('./libcity/config', candidate)
            if os.path.exists(candidate_path):
                merged.update(self._load_json_file(candidate))
        return merged

    def _load_default_config(self):
        # 首先加载 task config
        with open('./libcity/config/task_config.json', 'r') as f:
            task_config = json.load(f)
            if self.config['task'] not in task_config:
                raise ValueError(
                    'task {} is not supported.'.format(self.config['task']))
            task_entry = task_config[self.config['task']]
            # check model and dataset
            model = self.config['model']
            allowed_models = task_entry.get('allowed_model', [])
            if model not in allowed_models:
                # Don't fail hard if model is not listed; allow registration-based discovery.
                # Log a warning and try to infer dataset/executor/evaluator later.
                # Note: some workflows may still prefer strict checking; this relaxes that.
                # We import logging lazily to avoid heavy imports here.
                try:
                    import logging
                    logging.getLogger(__name__).warning(
                        "Model %s is not listed in task_config for task %s; will try auto-discovery.",
                        model, self.config['task'])
                except Exception:
                    pass
            # Try to set dataset_class/executor/evaluator from task config mapping if present
            model_mapping = task_entry.get(model, {}) if isinstance(task_entry, dict) else {}
            # dataset_class
            if 'dataset_class' not in self.config:
                if 'dataset_class' in model_mapping:
                    self.config['dataset_class'] = model_mapping['dataset_class']
                else:
                    # fallback: try sensible defaults or discover from registry
                    try:
                        from libcity.data.registry import DATASET_REGISTRY
                        # prefer TrafficStatePointDataset if present
                        if 'TrafficStatePointDataset' in DATASET_REGISTRY.items():
                            self.config['dataset_class'] = 'TrafficStatePointDataset'
                        else:
                            # pick first registered dataset if exists
                            items = DATASET_REGISTRY.items()
                            if items:
                                self.config['dataset_class'] = list(items.keys())[0]
                            else:
                                # fallback to existing value in config or empty
                                self.config.setdefault('dataset_class', '')
                    except Exception:
                        self.config.setdefault('dataset_class', '')
            # task-specific encoder defaults
            if self.config['task'] == 'traj_loc_pred' and 'traj_encoder' not in self.config:
                if 'traj_encoder' in model_mapping:
                    self.config['traj_encoder'] = model_mapping['traj_encoder']
            if self.config['task'] == 'eta' and 'eta_encoder' not in self.config:
                if 'eta_encoder' in model_mapping:
                    self.config['eta_encoder'] = model_mapping['eta_encoder']
            # executor
            if 'executor' not in self.config:
                if 'executor' in model_mapping:
                    self.config['executor'] = model_mapping['executor']
                else:
                    # try model-specific executor name else default
                    try:
                        from libcity.executor.registry import EXECUTOR_REGISTRY
                        candidate = model + 'Executor'
                        if candidate in EXECUTOR_REGISTRY.items():
                            self.config['executor'] = candidate
                        else:
                            # pick TrafficStateExecutor if present
                            if 'TrafficStateExecutor' in EXECUTOR_REGISTRY.items():
                                self.config['executor'] = 'TrafficStateExecutor'
                            else:
                                # pick first registered executor
                                items = EXECUTOR_REGISTRY.items()
                                if items:
                                    self.config['executor'] = list(items.keys())[0]
                                else:
                                    self.config.setdefault('executor', '')
                    except Exception:
                        self.config.setdefault('executor', '')
            # evaluator
            if 'evaluator' not in self.config:
                if 'evaluator' in model_mapping:
                    self.config['evaluator'] = model_mapping['evaluator']
                else:
                    try:
                        from libcity.evaluator.registry import EVALUATOR_REGISTRY
                        candidate = model + 'Evaluator'
                        if candidate in EVALUATOR_REGISTRY.items():
                            self.config['evaluator'] = candidate
                        else:
                            if 'TrafficStateEvaluator' in EVALUATOR_REGISTRY.items():
                                self.config['evaluator'] = 'TrafficStateEvaluator'
                            else:
                                items = EVALUATOR_REGISTRY.items()
                                if items:
                                    self.config['evaluator'] = list(items.keys())[0]
                                else:
                                    self.config.setdefault('evaluator', '')
                    except Exception:
                        self.config.setdefault('evaluator', '')
            # 对于 LSTM RNN GRU 使用的都是同一个类，只是 RNN 模块不一样而已，这里做一下修改
            if self.config['model'].upper() in ['LSTM', 'GRU', 'RNN']:
                self.config['rnn_type'] = self.config['model']
                self.config['model'] = 'RNN'
            # if self.config['dataset'] not in task_config['allowed_dataset']:
            #     raise ValueError('task {} do not support dataset {}'.format(
            #         self.config['task'], self.config['dataset']))
        # 接着加载每个阶段的 default config
        default_file_list = []
        # model
        default_file_list.append('model/{}/{}.json'.format(self.config['task'], self.config['model']))
        # dataset
        default_file_list.append('data/{}.json'.format(self.config['dataset_class']))
        # executor
        default_file_list.append('executor/{}.json'.format(self.config['executor']))
        # evaluator
        default_file_list.append('evaluator/{}.json'.format(self.config['evaluator']))
        # 加载所有默认配置
        for file_name in default_file_list:
            defaults = self._load_stage_defaults(file_name)
            for key in defaults:
                if key not in self.config:
                    self.config[key] = defaults[key]
        # 加载数据集config.json
        with open('./raw_data/{}/config.json'.format(self.config['dataset']), 'r') as f:
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
        use_gpu = self.config.get('gpu', True)
        gpu_id = self.config.get('gpu_id', 0)
        if use_gpu:
            torch.cuda.set_device(gpu_id)
        self.config['device'] = torch.device(
            "cuda:%d" % gpu_id if torch.cuda.is_available() and use_gpu else "cpu")

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
