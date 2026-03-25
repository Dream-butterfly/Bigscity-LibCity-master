import ast
from logging import getLogger

import optuna
from optuna.samplers import GridSampler, RandomSampler, TPESampler


class HyperTuning:
    """
    自动调参

    Note:
        HyperTuning is based on Optuna (https://optuna.org/)
    """

    def __init__(self, objective_function, space=None, params_file=None, algo='grid_search',
                 max_evals=100, task=None, model_name=None, dataset_name=None, config_file=None,
                 saved_model=True, train=True, other_args=None):
        self.task = task
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.config_file = config_file
        self.saved_model = saved_model
        self.train = train
        self.other_args = other_args or {}
        self._logger = getLogger()

        self.best_score = None
        self.best_params = None
        self.best_test_result = None
        self.params2result = {}

        self.objective_function = objective_function
        self.max_evals = max_evals
        if space:
            self.space = self._normalize_space(space)
        elif params_file:
            self.space = self._build_space_from_file(params_file)
        else:
            raise ValueError('at least one of `space` and `params_file` is provided')

        supported_algos = {'grid_search', 'tpe', 'random_search'}
        if algo == 'atpe':
            raise ValueError('Illegal hyper algorithm type [atpe], use [tpe] instead')
        if algo not in supported_algos:
            raise ValueError('Illegal hyper algorithm type [{}]'.format(algo))
        self.algo = algo

        if self.algo == 'grid_search':
            self.max_evals = min(self.max_evals, self._spacesize())

    @staticmethod
    def _coerce_value(value):
        if isinstance(value, (int, float, str, bool)):
            return value
        raise TypeError('Unsupported parameter value type [{}]'.format(type(value).__name__))

    @classmethod
    def _normalize_space(cls, space):
        normalized = {}
        for name, spec in space.items():
            if isinstance(spec, dict):
                normalized[name] = spec
                continue
            if isinstance(spec, (list, tuple)):
                normalized[name] = {'type': 'choice', 'values': [cls._coerce_value(v) for v in spec]}
                continue
            raise TypeError('Unsupported space spec for parameter [{}]'.format(name))
        return normalized

    @staticmethod
    def _build_space_from_file(file):
        space = {}
        with open(file, 'r') as fp:
            for line in fp:
                para_list = line.strip().split(' ')
                if len(para_list) < 3:
                    continue
                para_name, para_type, para_value = para_list[0], para_list[1], "".join(para_list[2:])
                if para_type == 'choice':
                    values = ast.literal_eval(para_value)
                    space[para_name] = {'type': 'choice', 'values': values}
                elif para_type == 'uniform':
                    low, high = para_value.strip().split(',')
                    space[para_name] = {'type': 'uniform', 'low': float(low), 'high': float(high)}
                elif para_type == 'quniform':
                    low, high, q = para_value.strip().split(',')
                    space[para_name] = {
                        'type': 'quniform',
                        'low': float(low),
                        'high': float(high),
                        'step': float(q)
                    }
                elif para_type == 'loguniform':
                    low, high = para_value.strip().split(',')
                    space[para_name] = {'type': 'loguniform', 'low': float(low), 'high': float(high)}
                else:
                    raise ValueError('Illegal parameter type [{}]'.format(para_type))
        return space

    @staticmethod
    def params2str(params):
        params_str = ''
        for param_name in params:
            params_str += param_name + ':' + str(params[param_name]) + ', '
        return params_str[:-2]

    def _spacesize(self):
        total = 1
        for spec in self.space.values():
            if spec['type'] != 'choice':
                raise ValueError('Grid search only supports choice parameters')
            total *= len(spec['values'])
        return total

    def _build_grid_space(self):
        return {
            name: [self._coerce_value(value) for value in spec['values']]
            for name, spec in self.space.items()
        }

    def save_result(self, filename=None):
        with open(filename, 'w') as fp:
            fp.write('best params: ' + str(self.best_params) + '\n')
            fp.write('best_valid_score: \n')
            fp.write(str(self.params2result[self.params2str(self.best_params)]['best_valid_score']) + '\n')
            fp.write('best_test_result: \n')
            fp.write(str(self.params2result[self.params2str(self.best_params)]['test_result']) + '\n')
            fp.write('----------------------------------------------------------------------------\n')
            fp.write('All parameters tune and result: \n')
            for params in self.params2result:
                fp.write(params + '\n')
                fp.write('Test result:\n' + str(self.params2result[params]['test_result']) + '\n')
        self._logger.info('hyper-tuning result is saved at {}'.format(filename))

    def _suggest_parameter(self, trial, name, spec):
        para_type = spec['type']
        if para_type == 'choice':
            return trial.suggest_categorical(name, spec['values'])
        if para_type == 'uniform':
            return trial.suggest_float(name, spec['low'], spec['high'])
        if para_type == 'quniform':
            low = spec['low']
            high = spec['high']
            step = spec['step']
            if float(low).is_integer() and float(high).is_integer() and float(step).is_integer():
                return trial.suggest_int(name, int(low), int(high), step=int(step))
            return trial.suggest_float(name, low, high, step=step)
        if para_type == 'loguniform':
            return trial.suggest_float(name, spec['low'], spec['high'], log=True)
        raise ValueError('Illegal parameter type [{}]'.format(para_type))

    def fn(self, params):
        hyper_config_dict = params.copy()
        params_str = self.params2str(params)
        self._logger.info('running parameters:')
        self._logger.info(str(hyper_config_dict))
        result_dict = self.objective_function(
            task=self.task, model_name=self.model_name, dataset_name=self.dataset_name,
            config_file=self.config_file, saved_model=self.saved_model, train=self.train,
            other_args=self.other_args, hyper_config_dict=hyper_config_dict)
        self.params2result[params_str] = result_dict

        score = result_dict['best_valid_score']
        if self.best_score is None or score < self.best_score:
            self.best_score = score
            self.best_params = params.copy()
            self.best_test_result = result_dict.get('test_result')
        self._logger.info('current parameters:')
        self._logger.info(str(hyper_config_dict))
        self._logger.info('current best valid score: %.4f' % result_dict['best_valid_score'])
        self._logger.info('current test result:')
        self._logger.info(result_dict['test_result'])
        return score

    def _objective(self, trial):
        params = {}
        for name, spec in self.space.items():
            params[name] = self._suggest_parameter(trial, name, spec)
        return self.fn(params)

    def _run_grid_search(self):
        if not self.space:
            return
        study = optuna.create_study(
            direction='minimize',
            sampler=GridSampler(self._build_grid_space())
        )
        study.optimize(self._objective, n_trials=self.max_evals)

    def _run_sampler_search(self, sampler):
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(self._objective, n_trials=self.max_evals)

    def start(self):
        if self.algo == 'grid_search':
            self._run_grid_search()
        elif self.algo == 'random_search':
            seed = self.other_args.get('seed')
            self._run_sampler_search(RandomSampler(seed=seed))
        elif self.algo == 'tpe':
            seed = self.other_args.get('seed')
            self._run_sampler_search(TPESampler(seed=seed))
