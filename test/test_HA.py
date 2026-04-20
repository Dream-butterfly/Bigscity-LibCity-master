import numpy as np
import sys
import os
from baseline_utils import get_project_root, load_dataset_3d

root_path = get_project_root(__file__)
sys.path.append(root_path)
from GNNTP.common.evaluator_utils import evaluate_model


config = {
    'model': 'HA',
    'lag': [24 * 7 * 12],
    'weight': [1],
    'dataset': 'METR_LA',
    'train_rate': 0.7,
    'eval_rate': 0.1,
    'input_window': 12,
    'output_window': 3,
    'null_value': 0,
    'metrics': ['MAE', 'MAPE', 'MSE', 'RMSE', 'masked_MAE',
                'masked_MAPE', 'masked_MSE', 'masked_RMSE', 'R2', 'EVAR']
}


def get_data(dataset):
    config['dataset'] = dataset
    return load_dataset_3d(config)


def historical_average(data):
    t, n, f = data.shape
    train_rate = config.get('train_rate', 0.7)
    eval_rate = config.get('eval_rate', 0.1)
    output_window = config.get('output_window', 3)
    lag = config.get('lag', [7 * 24 * 12])
    weight = config.get('weight', 1.0)
    null_value = config.get('null_value', 0)

    if isinstance(lag, int):
        lag = [lag]
    if isinstance(weight, int) or isinstance(weight, float):
        weight = [weight]
    assert len(lag) == len(weight)
    assert np.isclose(sum(weight), 1.0)

    y_true = []
    y_pred = []
    for i in range(int(t * (train_rate + eval_rate)), t):
        # y_true
        y_true.append(data[i, :, :])  # (N, F)
        # y_pred
        y_pred_i = 0
        for j in range(len(lag)):
            # 隔lag[j]时间步在整个训练集采样, 得到(n_sample, N, F)取平均值得到(N, F), 最后用weight[j]加权
            inds = [j for j in range(i % lag[j], int(t * (train_rate + eval_rate)), lag[j])]
            history = data[inds, :, :].copy()
            # 对得到的history数据去除空值后求平均
            null_mask = (history == null_value)
            history[null_mask] = np.nan
            y_pred_i += weight[j] * np.nanmean(history, axis=0)
            y_pred_i[np.isnan(y_pred_i)] = 0
        y_pred.append(y_pred_i)  # (N, F)

    y_pred = np.array(y_pred)  # (test_size, N, F)
    y_true = np.array(y_true)  # (test_size, N, F)
    y_pred = np.expand_dims(y_pred, axis=1)  # (test_size, 1, N, F)
    y_true = np.expand_dims(y_true, axis=1)  # (test_size, 1, N, F)
    y_pred = np.repeat(y_pred, output_window, axis=1)  # (test_size, out, N, F)
    y_true = np.repeat(y_true, output_window, axis=1)  # (test_size, out, N, F)
    return y_pred, y_true


def main():
    print(config)
    data = get_data(config.get('dataset', ''))
    y_pred, y_true = historical_average(data)
    # y_pred = y_pred[:, :, :, 0]
    # y_true = y_true[:, :, :, 0]
    evaluate_model(y_pred=y_pred, y_true=y_true, metrics=config['metrics'],
                   path=config['model'] + '_' + config['dataset'] + '_metrics.csv')


if __name__ == '__main__':
    main()
