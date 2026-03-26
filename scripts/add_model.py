"""
Add a new model to the LibCity project.

This helper creates:
 - libcity/model/<task>/<ModelName>.py (template with @register_model)
 - libcity/config/model/<task>/<ModelName>.json (default config)
 - Updates libcity/config/task_config.json: adds model to allowed_model and a mapping

Usage (interactive):
    python scripts/add_model.py

Usage (non-interactive):
    python scripts/add_model.py --model MyModel --task traffic_state_pred --dataset_class TrafficStatePointDataset \
        --executor TrafficStateExecutor --evaluator TrafficStateEvaluator --yes

Options:
  --model / -m         Model name (class name, e.g. MyModel)
  --task / -t          Task name (default: traffic_state_pred)
  --dataset_class      Dataset class name (default: TrafficStatePointDataset)
  --executor           Executor class name (default: TrafficStateExecutor)
  --evaluator          Evaluator class name (default: TrafficStateEvaluator)
  --create_config      Whether to create a default config JSON (default: True)
  --yes / -y           Run non-interactive and overwrite without prompt
  --overwrite          Overwrite existing files (default: False)

The script is conservative: it will not overwrite files unless --overwrite is specified
or --yes is used to confirm.
"""
from __future__ import annotations
import argparse
import json
import os
import re
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]

MODEL_TEMPLATE = """"""  # placeholder

def camel_to_snake(name: str) -> str:
    s1 = re.sub('(.)([A-Z][a-z]+)', r"\1_\2", name)
    return re.sub('([a-z0-9])([A-Z])', r"\1_\2", s1).lower()


def model_template(model_name: str, task: str) -> str:
    return dedent(f'''
    """
    Minimal template for {model_name} (task={task}).

    Implement your model by editing this file. Keep heavy imports inside the class
    if you want to avoid import-time side-effects.
    """
    import torch
    import torch.nn as nn
    from logging import getLogger
    from libcity.model.registry import register_model
    from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
    from libcity.model import loss


    @register_model("{task}")
    class {model_name}(AbstractTrafficStateModel):
        """A minimal example model. Customize layers and forward/predict/loss."""
        def __init__(self, config, data_feature):
            super().__init__(config, data_feature)
            self._logger = getLogger()
            self.num_nodes = data_feature.get('num_nodes', 1)
            self.input_window = config.get('input_window', 12)
            self.output_window = config.get('output_window', 12)
            self.feature_dim = data_feature.get('feature_dim', 1)
            self.output_dim = data_feature.get('output_dim', 1)
            self.device = config.get('device', torch.device('cpu'))

            # A tiny feed-forward example: project flattened input to output
            hidden = max(16, self.num_nodes * self.feature_dim)
            self.backbone = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.input_window * self.num_nodes * self.feature_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, self.output_window * self.num_nodes * self.output_dim),
            )

        def forward(self, batch, batches_seen=None):
            x = batch['X']  # (B, input_window, N, F)
            B = x.shape[0]
            x = x.to(self.device)
            out = self.backbone(x)
            out = out.view(B, self.output_window, self.num_nodes, self.output_dim)
            return out

        def predict(self, batch, batches_seen=None):
            return self.forward(batch, batches_seen)

        def calculate_loss(self, batch, batches_seen=None):
            y_true = batch['y']
            y_pred = self.predict(batch, batches_seen)
            if self.data_feature.get('scaler') is not None:
                y_true = self.data_feature['scaler'].inverse_transform(y_true[..., :self.output_dim])
                y_pred = self.data_feature['scaler'].inverse_transform(y_pred[..., :self.output_dim])
            return loss.masked_mae_torch(y_pred, y_true, 0)
    ''')


DEFAULT_CONFIG_TEMPLATE = {
    "max_epoch": 50,
    "learning_rate": 0.001,
    "batch_size": 64,
    "input_window": 12,
    "output_window": 12,
    "scaler": "standard"
}


def safe_write(path: Path, content: str, overwrite: bool = False) -> bool:
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
    return True


def update_task_config(task_config_path: Path, task: str, model_name: str, dataset_class: str, executor: str, evaluator: str) -> bool:
    # load
    with task_config_path.open('r', encoding='utf-8') as f:
        cfg = json.load(f)

    if task not in cfg:
        raise ValueError(f"Task '{task}' not found in {task_config_path}")

    task_entry = cfg[task]
    # ensure allowed_model includes model_name
    allowed = set(task_entry.get('allowed_model', []))
    if model_name not in allowed:
        allowed.add(model_name)
        task_entry['allowed_model'] = sorted(list(allowed))

    # add mapping for this model
    if model_name not in task_entry:
        task_entry[model_name] = {
            'dataset_class': dataset_class,
            'executor': executor,
            'evaluator': evaluator
        }
    else:
        # don't override existing mapping, but ensure required keys exist
        for k, v in [('dataset_class', dataset_class), ('executor', executor), ('evaluator', evaluator)]:
            if k not in task_entry[model_name]:
                task_entry[model_name][k] = v

    # write back
    with task_config_path.open('w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return True


def main():
    parser = argparse.ArgumentParser(description='Add a new model to LibCity')
    parser.add_argument('--model', '-m', help='Model class name (CamelCase), e.g. MyModel')
    parser.add_argument('--task', '-t', default='traffic_state_pred', help='Task name (default: traffic_state_pred)')
    parser.add_argument('--dataset_class', default='TrafficStatePointDataset')
    parser.add_argument('--executor', default='TrafficStateExecutor')
    parser.add_argument('--evaluator', default='TrafficStateEvaluator')
    parser.add_argument('--create_config', action='store_true', default=True, help='Create default model config JSON')
    parser.add_argument('--no-config', action='store_false', dest='create_config', help='Do not create config JSON')
    parser.add_argument('--yes', '-y', action='store_true', help='Assume yes for prompts')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    args = parser.parse_args()

    # interactive if model not provided
    if not args.model:
        args.model = input('Model class name (CamelCase, e.g. MyModel): ').strip()
    model_name = args.model.strip()
    if not re.match(r'^[A-Z][A-Za-z0-9_]*$', model_name):
        print('Model name must be CamelCase starting with uppercase letter.')
        return

    task = args.task
    dataset_class = args.dataset_class
    executor = args.executor
    evaluator = args.evaluator

    print(f"Adding model {model_name} for task {task}")

    # Paths
    model_dir = ROOT / 'libcity' / 'model' / task
    model_file = model_dir / (model_name + '.py')
    config_dir = ROOT / 'libcity' / 'config' / 'model' / task
    config_file = config_dir / (model_name + '.json')
    task_config_path = ROOT / 'libcity' / 'config' / 'task_config.json'

    if model_file.exists() and not args.overwrite:
        print(f"Model file {model_file} already exists. Use --overwrite to replace.")
        if not args.yes:
            return

    # create model file
    created_model = safe_write(model_file, model_template(model_name, task), overwrite=args.overwrite)
    if created_model:
        print(f"Created model file: {model_file}")
    else:
        print(f"Skipped creating model file (exists): {model_file}")

    # create config json if requested
    if args.create_config:
        if config_file.exists() and not args.overwrite:
            print(f"Config file {config_file} already exists. Use --overwrite to replace.")
        else:
            config_dir.mkdir(parents=True, exist_ok=True)
            with config_file.open('w', encoding='utf-8') as f:
                json.dump(DEFAULT_CONFIG_TEMPLATE, f, indent=2)
            print(f"Created config JSON: {config_file}")

    # update task_config.json
    try:
        update_task_config(task_config_path, task, model_name, dataset_class, executor, evaluator)
        print(f"Updated task_config.json: added {model_name} mapping under task {task}")
    except Exception as e:
        print('Failed to update task_config.json:', e)
        return

    print('\nNext steps:')
    print(f"- Edit {model_file} to implement your model logic.")
    if args.create_config:
        print(f"- Edit {config_file} to tune default hyperparameters.")
    print("- Run a smoke test: python scripts/run_smoke_custom.py --task {} --model {} --dataset PEMSD4 --batch_size 1 --gpu False --num_workers 0".format(task, model_name))
    print('- Run training: python run_model.py --task {} --model {} --dataset PEMSD4 --max_epoch 1 --gpu False --num_workers 0'.format(task, model_name))


if __name__ == '__main__':
    main()

