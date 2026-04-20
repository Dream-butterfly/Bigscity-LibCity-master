import os
from logging import getLogger

import numpy as np

from GNNTP.data.dataset import AbstractDataset
from GNNTP.data.dataset.traffic_state_dataset_mixins import (
    TrafficStateExternalFeatureMixin,
    TrafficStateGraphMixin,
    TrafficStatePipelineMixin,
    TrafficStateResourceMixin,
    TrafficStateTemporalLoaderMixin,
)
from GNNTP.utils import ensure_dir, get_dataset_cache_dir


class TrafficStateDataset(
    TrafficStateResourceMixin,
    TrafficStateGraphMixin,
    TrafficStateTemporalLoaderMixin,
    TrafficStateExternalFeatureMixin,
    TrafficStatePipelineMixin,
    AbstractDataset,
):
    """
    交通状态预测数据集的基类。
    默认使用`input_window`的数据预测`output_window`对应的数据，即一个X，一个y。
    一般将外部数据融合到X中共同进行预测，因此数据为[X, y]。
    默认使用`train_rate`和`eval_rate`在样本数量(num_samples)维度上直接切分训练集、测试集、验证集。
    """

    def __init__(self, config):
        self.config = config
        self.dataset = self.config.get("dataset", "")
        self.batch_size = self.config.get("batch_size", 64)
        self.cache_dataset = self.config.get("cache_dataset", True)
        self.num_workers = self.config.get("num_workers", 0)
        self.pad_with_last_sample = self.config.get("pad_with_last_sample", True)
        self.train_rate = self.config.get("train_rate", 0.7)
        self.eval_rate = self.config.get("eval_rate", 0.1)
        self.scaler_type = self.config.get("scaler", "none")
        self.ext_scaler_type = self.config.get("ext_scaler", "none")
        self.load_external = self.config.get("load_external", False)
        self.normal_external = self.config.get("normal_external", False)
        self.add_time_in_day = self.config.get("add_time_in_day", False)
        self.add_day_in_week = self.config.get("add_day_in_week", False)
        self.input_window = self.config.get("input_window", 12)
        self.output_window = self.config.get("output_window", 12)
        self.robustness_test = self.config.get("robustness_test", False)
        self.disturb_rate = self.config.get("disturb_rate", 0.5)
        self.noise_type = self.config.get("noise_type", "none")
        self.noise_mean = self.config.get("noise_mean", [0])
        self.noise_SD = self.config.get("noise_SD", [0])
        self.parameters_str = (
            str(self.dataset)
            + "_"
            + str(self.input_window)
            + "_"
            + str(self.output_window)
            + "_"
            + str(self.train_rate)
            + "_"
            + str(self.eval_rate)
            + "_"
            + str(self.scaler_type)
            + "_"
            + str(self.batch_size)
            + "_"
            + str(self.load_external)
            + "_"
            + str(self.add_time_in_day)
            + "_"
            + str(self.add_day_in_week)
            + "_"
            + str(self.pad_with_last_sample)
        )
        self.cache_file_name = os.path.join(
            get_dataset_cache_dir(), "traffic_state_{}.npz".format(self.parameters_str)
        )
        self.cache_file_folder = get_dataset_cache_dir()
        ensure_dir(self.cache_file_folder)
        self.data_path = "./resource_data/" + self.dataset + "/"
        if not os.path.exists(self.data_path):
            raise ValueError(
                "Dataset {} not exist! Please ensure the path "
                "'./resource_data/{}/' exist!".format(self.dataset, self.dataset)
            )
        # 加载数据集的config.json文件
        self.weight_col = self.config.get("weight_col", "")
        self.data_col = self.config.get("data_col", "")
        self.ext_col = self.config.get("ext_col", "")
        self.geo_file = self.config.get("geo_file", self.dataset)
        self.rel_file = self.config.get("rel_file", self.dataset)
        self.data_files = self.config.get("data_files", self.dataset)
        self.ext_file = self.config.get("ext_file", self.dataset)
        self.output_dim = self.config.get("output_dim", 1)
        self.time_intervals = self.config.get("time_intervals", 300)  # s
        self.init_weight_inf_or_zero = self.config.get("init_weight_inf_or_zero", "inf")
        self.set_weight_link_or_dist = self.config.get("set_weight_link_or_dist", "dist")
        self.bidir_adj_mx = self.config.get("bidir_adj_mx", False)
        self.calculate_weight_adj = self.config.get("calculate_weight_adj", False)
        self.weight_adj_epsilon = self.config.get("weight_adj_epsilon", 0.1)
        self.distance_inverse = self.config.get("distance_inverse", False)

        # 初始化
        self.data = None
        self.feature_name = {"X": "float", "y": "float"}  # 此类的输入只有X和y
        self.adj_mx = None
        self.scaler = None
        self.ext_scaler = None
        self.feature_dim = 0
        self.ext_dim = 0
        self.num_nodes = 0
        self.num_batches = 0
        self._logger = getLogger()
        self._normalize_dataset_resource_files()
        geo_path = os.path.join(self.data_path, self.geo_file + ".geo")
        if os.path.exists(geo_path):
            self._load_geo()
        else:
            raise ValueError("Not found .geo file: {}".format(geo_path))
        rel_path = os.path.join(self.data_path, self.rel_file + ".rel")
        if os.path.exists(rel_path):  # .rel file is not necessary
            self._load_rel()
        else:
            self.adj_mx = np.zeros((len(self.geo_ids), len(self.geo_ids)), dtype=np.float32)

    def _load_dyna(self, filename):
        """
        加载数据文件(.dyna/.grid/.od/.gridod)，子类必须实现这个方法来指定如何加载数据文件，返回对应的多维数据,
        提供5个实现好的方法加载上述几类文件，并转换成不同形状的数组:
        `_load_dyna_3d`/`_load_grid_3d`/`_load_grid_4d`/`_load_grid_od_4d`/`_load_grid_od_6d`

        Args:
            filename(str): 数据文件名，不包含后缀

        Returns:
            np.ndarray: 数据数组
        """
        raise NotImplementedError("Please implement the function `_load_dyna()`.")

    def _add_external_information(self, df, ext_data=None):
        """
        将外部数据和原始交通状态数据结合到高维数组中，子类必须实现这个方法来指定如何融合外部数据和交通状态数据,
        如果不想加外部数据，可以把交通状态数据`df`直接返回,
        提供3个实现好的方法适用于不同形状的交通状态数据跟外部数据结合:
        `_add_external_information_3d`/`_add_external_information_4d`/`_add_external_information_6d`

        Args:
            df(np.ndarray): 交通状态数据多维数组
            ext_data(np.ndarray): 外部数据

        Returns:
            np.ndarray: 融合后的外部数据和交通状态数据
        """
        raise NotImplementedError("Please implement the function `_add_external_information()`.")

    def get_data_feature(self):
        """
        返回数据集特征，子类必须实现这个函数，返回必要的特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        raise NotImplementedError("Please implement the function `get_data_feature()`.")
