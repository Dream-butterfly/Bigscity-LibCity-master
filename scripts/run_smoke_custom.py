"""
Custom smoke test that allows passing extra config options (gpu, num_workers, etc.).
Runs a single forward pass using the project's ConfigParser/get_dataset/get_model.
"""
import argparse
from libcity.tasks.config_parser import ConfigParser
from libcity.data import get_dataset
from libcity.utils import get_model, set_random_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='traffic_state_pred')
    parser.add_argument('--model', type=str, default='DCRNN')
    parser.add_argument('--dataset', type=str, default='PEMSD4')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gpu', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    other_args = {'batch_size': args.batch_size, 'seed': args.seed, 'gpu': args.gpu, 'num_workers': args.num_workers}
    config = ConfigParser(task=args.task, model=args.model, dataset=args.dataset, config_file=None, saved_model=False, train=False, other_args=other_args)
    print('Config device:', config['device'])
    set_random_seed(config.get('seed', 0))

    dataset = get_dataset(config)
    train_data, _, _ = dataset.get_data()
    data_feature = dataset.get_data_feature()
    batch = next(iter(train_data))

    model = get_model(config, data_feature).to(config['device'])
    print('Model instantiated:', model.__class__.__name__)
    batch.to_tensor(config['device'])
    output = model.predict(batch)
    print('Output shape:', getattr(output, 'shape', str(type(output))))


if __name__ == '__main__':
    main()

