import numpy as np
import sys
import os
from statsmodels.tsa.api import VAR
from baseline_utils import get_project_root, load_dataset_3d

root_path = get_project_root(__file__)
sys.path.append(root_path)
from libcity.utils import StandardScaler
from libcity.common.evaluator_utils import evaluate_model
from libcity.utils import preprocess_data


config = {
    'model': 'VAR',
    'maxlags': 1,
    'dataset': 'METR_LA',
    'train_rate': 0.7,
    'eval_rate': 0.1,
    'input_window': 12,
    'output_window': 3,
    'metrics': ['MAE', 'MAPE', 'MSE', 'RMSE', 'masked_MAE',
                'masked_MAPE', 'masked_MSE', 'masked_RMSE', 'R2', 'EVAR']
}


def get_data(dataset):
    config['dataset'] = dataset
    return load_dataset_3d(config)


def run_VAR(data, inputs):
    ts, points, f = data.shape
    train_rate = config.get('train_rate', 0.7)
    eval_rate = config.get('eval_rate', 0.1)
    output_window = config.get('output_window', 3)
    maxlags = config.get("maxlags", 1)

    data = data.reshape(ts, -1)[:int(ts * (train_rate + eval_rate))]  # (train_size, N * F)
    scaler = StandardScaler(data.mean(), data.std())
    data = scaler.transform(data)

    model = VAR(data)
    results = model.fit(maxlags=maxlags, ic='aic')

    inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], -1)  # (num_samples, out, N * F)
    y_pred = []  # (num_samples, out, N, F)
    for sample in inputs:  # (out, N * F)
        sample = scaler.transform(sample[-maxlags:])  # (T, N, F)
        out = results.forecast(sample, output_window)  # (out, N * F)
        out = scaler.inverse_transform(out)  # (out, N * F)
        y_pred.append(out.reshape(output_window, points, f))
    y_pred = np.array(y_pred)  # (num_samples, out, N, F)
    return y_pred


def main():
    print(config)
    data = get_data(config.get('dataset', ''))
    trainx, trainy, testx, testy = preprocess_data(data, config)
    y_pred = run_VAR(data, testx)
    evaluate_model(y_pred=y_pred, y_true=testy, metrics=config['metrics'],
                   path=config['model']+'_'+config['dataset']+'_metrics.csv')


if __name__ == '__main__':
    main()
