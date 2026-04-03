import json
import warnings
import os
import sys
import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from tqdm import tqdm
from statsmodels.tsa.arima.model import ARIMA
from baseline_utils import get_project_root, load_dataset_3d

root_path = get_project_root(__file__)
sys.path.append(root_path)
from libcity.common.evaluator_utils import evaluate_model
from libcity.utils import preprocess_data


config = {
    'model': 'ARIMA',
    'p_range': [0, 4],
    'd_range': [0, 3],
    'q_range': [0, 4],
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


# Try to find the best (p,d,q) parameters for ARIMA
def order_select_pred(data):
    # data: (T, F)
    res = ARIMA(data, order=(0, 0, 0)).fit()
    bic = res.bic
    p_range = config.get('p_range', [0, 4])
    d_range = config.get('d_range', [0, 3])
    q_range = config.get('q_range', [0, 4])
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=ConvergenceWarning)
        warnings.simplefilter("error", category=RuntimeWarning)
        for p in range(p_range[0], p_range[1]):
            for d in range(d_range[0], d_range[1]):
                for q in range(q_range[0], q_range[1]):
                    try:
                        cur_res = ARIMA(data, order=(p, d, q)).fit()
                    except:
                        continue
                    if cur_res.bic < bic:
                        bic = cur_res.bic
                        res = cur_res
    return res


def arima(data):
    output_window = config.get('output_window', 3)
    y_pred = []  # (num_samples, N, out, F)
    data = data.swapaxes(1, 2)  # (num_samples, N, out, F)
    for time_slot in tqdm(data, 'ts'):  # (N, out, F)
        y_pred_ele = []  # (N, out, F)
        # Different nodes should be predict by different ARIMA models instance.
        for seq in time_slot:  # (out, F)
            pred = order_select_pred(seq).forecast(steps=output_window)
            pred = pred.reshape((-1, seq.shape[1]))  # (out, F)
            y_pred_ele.append(pred)
        y_pred.append(y_pred_ele)
    return np.array(y_pred).swapaxes(1, 2)  # (num_samples, out, N, F)


def main():
    print(config)
    data = get_data(config.get('dataset', ''))
    trainx, trainy, testx, testy = preprocess_data(data, config)
    y_pred = arima(testx)
    evaluate_model(y_pred=y_pred, y_true=testy, metrics=config['metrics'],
                   path=config['model']+'_'+config['dataset']+'_metrics.csv')


if __name__ == '__main__':
    main()
