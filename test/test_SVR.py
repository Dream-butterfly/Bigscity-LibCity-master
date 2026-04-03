import sys
import os
import numpy as np
from tqdm import tqdm
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from baseline_utils import get_project_root, load_dataset_3d

root_path = get_project_root(__file__)
sys.path.append(root_path)
from libcity.common.evaluator_utils import evaluate_model
from libcity.utils import preprocess_data


config = {
    'model': 'SVR',
    'kernel': 'rbf',
    'dataset': 'METR_LA',
    'train_rate': 0.7,
    'eval_rate': 0.1,
    'input_window': 12,
    'output_window': 3,
    'metrics': ['MAE', 'MAPE', 'MSE', 'RMSE', 'masked_MAE',
                'masked_MAPE', 'masked_MSE', 'masked_RMSE', 'R2', 'EVAR']}


def get_data(dataset):
    config['dataset'] = dataset
    return load_dataset_3d(config)


def run_SVR(data):
    ts, num_nodes, f = data.shape
    output_window = config.get("output_window", 3)
    kernel = config.get('kernel', 'rbf')

    y_pred = []
    y_true = []
    for i in tqdm(range(num_nodes), 'num_nodes'):
        trainx, trainy, testx, testy = preprocess_data(data[:, i, :], config)  # (T, F)
        # (train_size, in/out, F), (test_size, in/out, F)
        trainx = np.reshape(trainx, (trainx.shape[0], -1))  # (train_size, in * F)
        trainy = np.reshape(trainy, (trainy.shape[0], -1))  # (train_size, out * F)
        testx = np.reshape(testx, (testx.shape[0], -1))  # (test_size, in * F)

        svr_model = MultiOutputRegressor(SVR(kernel=kernel))
        svr_model.fit(trainx, trainy)
        pre = svr_model.predict(testx)  # (test_size, out * F)
        y_pred.append(pre.reshape(pre.shape[0], output_window, f))
        y_true.append(testy)

    y_pred = np.array(y_pred)  # (N, test_size, out, F)
    y_true = np.array(y_true)  # (N, test_size, out, F)
    y_pred = y_pred.transpose((1, 2, 0, 3))  # (test_size, out, N, F)
    y_true = y_true.transpose((1, 2, 0, 3))  # (test_size, out, N, F)
    return y_pred, y_true


def main():
    print(config)
    data = get_data(config.get('dataset', ''))
    y_pred, y_true = run_SVR(data)
    evaluate_model(y_pred=y_pred, y_true=y_true, metrics=config['metrics'],
                   path=config['model']+'_'+config['dataset']+'_metrics.csv')


if __name__ == '__main__':
    main()
