from libcity.utils import get_evaluator, get_run_subdir
from libcity.executor.abstract_executor import AbstractExecutor


class GensimExecutor(AbstractExecutor):
    def __init__(self, config, model, data_feature):
        self.evaluator = get_evaluator(config)
        self.config = config
        self.model = model
        self.exp_id = config.get('exp_id', None)

        self.cache_dir = get_run_subdir(self.exp_id, 'model_cache')
        self.evaluate_res_dir = get_run_subdir(self.exp_id, 'evaluate_cache')

    def evaluate(self, test_dataloader):
        """
        use model to test data
        """
        self.evaluator.evaluate()

    def train(self, train_dataloader, eval_dataloader):
        """
        use data to train model with config
        """
        self.model.run()

    def load_model(self, cache_name):
        pass

    def save_model(self, cache_name):
        pass
