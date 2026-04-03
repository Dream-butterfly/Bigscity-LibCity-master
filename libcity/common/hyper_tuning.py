import ast
import math
from logging import getLogger

import optuna


class HyperTuning:
    """
    自动调参

    Note:
        HyperTuning is based on Optuna (https://optuna.org/)
    """

    def __init__(
        self,
        objective_function,
        space=None,
        params_file=None,
        algo='grid_search',
        max_evals=100,
        task=None,
        model_name=None,
        dataset_name=None,
        config_file=None,
        saved_model=True,
        train=True,
        other_args=None,
    ):
        self.task = task
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.config_file = config_file
        self.saved_model = saved_model
        self.train = train
        self.other_args = other_args
        self._logger = getLogger()

        self.best_score = None
        self.best_params = None
        self.best_test_result = None
        self.params2result = {}
        self.objective_function = objective_function
        self.study = None

        if space is not None:
            raise ValueError('custom `space` is no longer supported, please use `params_file`')
        if not params_file:
            raise ValueError('`params_file` is required')

        self.space = self._build_space_from_file(params_file)
        self.algo = algo
        self.max_evals = max_evals
        self.sampler = self._build_sampler(algo)

        if algo == 'grid_search':
            self.max_evals = self._grid_space_size()

    @staticmethod
    def _build_space_from_file(file):
        space = {}
        with open(file, 'r', encoding='utf-8') as fp:
            for raw_line in fp:
                line = raw_line.strip()
                if not line or line.startswith('#'):
                    continue
                para_list = line.split(' ')
                if len(para_list) < 3:
                    continue
                para_name = para_list[0]
                para_type = para_list[1]
                para_value = ''.join(para_list[2:])
                if para_type == 'choice':
                    values = ast.literal_eval(para_value)
                    if not isinstance(values, (list, tuple)) or len(values) == 0:
                        raise ValueError(f'Illegal choice parameter values for [{para_name}]')
                    space[para_name] = {
                        'type': 'choice',
                        'values': list(values),
                    }
                elif para_type == 'uniform':
                    low, high = para_value.split(',')
                    space[para_name] = {
                        'type': 'uniform',
                        'low': float(low),
                        'high': float(high),
                    }
                elif para_type == 'quniform':
                    low, high, q = para_value.split(',')
                    space[para_name] = {
                        'type': 'quniform',
                        'low': float(low),
                        'high': float(high),
                        'q': float(q),
                    }
                elif para_type == 'loguniform':
                    low, high = para_value.split(',')
                    space[para_name] = {
                        'type': 'loguniform',
                        'low': float(low),
                        'high': float(high),
                    }
                else:
                    raise ValueError(f'Illegal parameter type [{para_type}]')
        return space

    def _build_sampler(self, algo):
        if algo == 'grid_search':
            return optuna.samplers.GridSampler(self._build_grid_space())
        if algo == 'random_search':
            return optuna.samplers.RandomSampler()
        if algo == 'tpe':
            return optuna.samplers.TPESampler()
        if algo == 'atpe':
            self._logger.warning('Optuna does not provide ATPE; fallback to TPESampler.')
            return optuna.samplers.TPESampler()
        raise ValueError(f'Illegal hyper algorithm type [{algo}]')

    def _build_grid_space(self):
        grid_space = {}
        for name, spec in self.space.items():
            if spec['type'] == 'choice':
                grid_space[name] = list(spec['values'])
            elif spec['type'] == 'quniform':
                grid_space[name] = self._expand_quniform_values(spec['low'], spec['high'], spec['q'])
            else:
                raise ValueError(
                    f'grid_search only supports choice/quniform, but got [{spec["type"]}] for [{name}]'
                )
        return grid_space

    def _grid_space_size(self):
        size = 1
        for values in self._build_grid_space().values():
            size *= len(values)
        return size

    @staticmethod
    def _expand_quniform_values(low, high, q):
        if q <= 0:
            raise ValueError('quniform requires q > 0')
        steps = int(round((high - low) / q))
        values = []
        for index in range(steps + 1):
            value = low + index * q
            if value > high + 1e-12:
                break
            values.append(HyperTuning._normalize_number(value))
        if not values:
            raise ValueError('quniform produced no values')
        return values

    @staticmethod
    def _normalize_number(value):
        if isinstance(value, float) and math.isclose(value, round(value), rel_tol=0.0, abs_tol=1e-12):
            return int(round(value))
        return value

    @staticmethod
    def params2str(params):
        params_str = ''
        for param_name in params:
            params_str += param_name + ':' + str(params[param_name]) + ', '
        return params_str[:-2]

    def save_result(self, filename=None):
        with open(filename, 'w', encoding='utf-8') as fp:
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

    def _suggest_params(self, trial):
        params = {}
        for name, spec in self.space.items():
            if spec['type'] == 'choice':
                params[name] = trial.suggest_categorical(name, spec['values'])
            elif spec['type'] == 'uniform':
                params[name] = trial.suggest_float(name, spec['low'], spec['high'])
            elif spec['type'] == 'quniform':
                value = trial.suggest_float(name, spec['low'], spec['high'], step=spec['q'])
                params[name] = self._normalize_number(value)
            elif spec['type'] == 'loguniform':
                params[name] = trial.suggest_float(name, spec['low'], spec['high'], log=True)
            else:
                raise ValueError(f'Illegal parameter type [{spec["type"]}]')
        return params

    def fn(self, trial):
        hyper_config_dict = self._suggest_params(trial)
        params_str = self.params2str(hyper_config_dict)
        self._logger.info('running parameters:')
        self._logger.info(str(hyper_config_dict))
        result_dict = self.objective_function(
            task=self.task,
            model_name=self.model_name,
            dataset_name=self.dataset_name,
            config_file=self.config_file,
            saved_model=self.saved_model,
            train=self.train,
            other_args=self.other_args,
            hyper_config_dict=hyper_config_dict,
        )
        self.params2result[params_str] = result_dict

        score = result_dict['best_valid_score']
        if self.best_score is None or score < self.best_score:
            self.best_score = score
            self.best_params = hyper_config_dict.copy()
            self.best_test_result = result_dict.get('test_result')

        self._logger.info('current parameters:')
        self._logger.info(str(hyper_config_dict))
        self._logger.info('current best valid score: %.4f' % result_dict['best_valid_score'])
        self._logger.info('current test result:')
        self._logger.info(result_dict['test_result'])
        return score

    def start(self):
        self.study = optuna.create_study(direction='minimize', sampler=self.sampler)
        self.study.optimize(self.fn, n_trials=self.max_evals)
