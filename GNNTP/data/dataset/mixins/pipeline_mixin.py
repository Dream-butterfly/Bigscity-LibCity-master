import os

import numpy as np

from GNNTP.data.dataloader import generate_dataloader
from GNNTP.utils import (
    LogScaler,
    MinMax01Scaler,
    MinMax11Scaler,
    NoneScaler,
    NormalScaler,
    StandardScaler,
    ensure_dir,
    gaussian_noise,
    zero_noise,
)


class TrafficStatePipelineMixin:
    def _load_or_prepare_raw_splits(self):
        if self.data is None:
            if self.cache_dataset and os.path.exists(self.cache_file_name):
                x_train, y_train, x_val, y_val, x_test, y_test = self._load_cache_train_val_test()
            else:
                x_train, y_train, x_val, y_val, x_test, y_test = self._generate_train_val_test()
            self.data = {
                "x_train": np.array(x_train, copy=True),
                "y_train": np.array(y_train, copy=True),
                "x_val": np.array(x_val, copy=True),
                "y_val": np.array(y_val, copy=True),
                "x_test": np.array(x_test, copy=True),
                "y_test": np.array(y_test, copy=True),
            }
        return (
            np.array(self.data["x_train"], copy=True),
            np.array(self.data["y_train"], copy=True),
            np.array(self.data["x_val"], copy=True),
            np.array(self.data["y_val"], copy=True),
            np.array(self.data["x_test"], copy=True),
            np.array(self.data["y_test"], copy=True),
        )

    def _generate_input_data(self, df):
        num_samples = df.shape[0]
        input_w = self.input_window
        output_w = self.output_window
        total_w = input_w + output_w

        # -------- 边界检查 --------
        num_windows = num_samples - total_w + 1
        if num_windows <= 0:
            raise ValueError(
                f"Input data length ({num_samples}) is smaller than "
                f"input_window + output_window ({total_w})"
            )

        # -------- 核心：stride window --------
        windows = np.lib.stride_tricks.sliding_window_view(df, window_shape=total_w, axis=0)

        # windows shape:
        # (num_samples - total_w + 1, ..., total_w, feature_dim)

        # -------- 调整维度顺序（关键优化点）--------
        # 把时间维提前，避免后续切片产生非连续内存
        windows = np.moveaxis(windows, -1, 1)

        # 截断有效窗口（理论上已对齐，但保持严谨）
        windows = windows[:num_windows]

        # -------- 切分 x / y --------
        x = windows[:, :input_w, ...]
        y = windows[:, input_w:, ...]

        # -------- 关键：一次性 contiguous copy --------
        # 比 x.copy() 更可控（尤其对 PyTorch 友好）
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)

        return x, y

    def _generate_data(self):
        """
        加载数据文件(.dyna/.grid/.od/.gridod)和外部数据(.ext)，并生成 (x, y)

        Returns:
            x: (num_samples, input_length, ..., feature_dim)
            y: (num_samples, output_length, ..., feature_dim)
        """

        # -------- 统一 data_files（避免多余 copy）--------
        data_files = self.data_files if isinstance(self.data_files, list) else [self.data_files]

        # -------- 外部数据（只加载一次）--------
        ext_data = None
        if self.load_external:
            ext_path = os.path.join(self.data_path, self.ext_file + ".ext")
            if os.path.exists(ext_path):
                ext_data = self._load_ext()

        # -------- 主流程 --------
        x_list = []
        y_list = []

        for filename in data_files:
            # 1. 加载主数据
            df = self._load_dyna(filename)

            # 2. 融合时间特征(tod/dow)与外部数据
            # _add_external_information 内部已处理 ext_data=None 的情况
            df = self._add_external_information(df, ext_data)

            # 3. 滑窗生成
            x, y = self._generate_input_data(df)

            x_list.append(x)
            y_list.append(y)

        # -------- concat 优化 --------
        if len(x_list) == 1:
            x, y = x_list[0], y_list[0]
        else:
            # 一次性拼接（避免链式 concat）
            x = np.concatenate(x_list, axis=0)
            y = np.concatenate(y_list, axis=0)

        # -------- logging（避免字符串拼接开销）--------
        self._logger.info("Dataset created")
        self._logger.info("x shape: %s, y shape: %s", x.shape, y.shape)

        return x, y

    def _split_train_val_test(self, x, y):
        """
        划分训练集、测试集、验证集，并缓存数据集

        Args:
            x(np.ndarray): 输入数据 (num_samples, input_length, ..., feature_dim)
            y(np.ndarray): 输出数据 (num_samples, input_length, ..., feature_dim)

        Returns:
            tuple: tuple contains:
                x_train: (num_samples, input_length, ..., feature_dim) \n
                y_train: (num_samples, input_length, ..., feature_dim) \n
                x_val: (num_samples, input_length, ..., feature_dim) \n
                y_val: (num_samples, input_length, ..., feature_dim) \n
                x_test: (num_samples, input_length, ..., feature_dim) \n
                y_test: (num_samples, input_length, ..., feature_dim)
        """
        test_rate = 1 - self.train_rate - self.eval_rate

        num_samples = x.shape[0]
        num_test = round(num_samples * test_rate)
        num_train = round(num_samples * self.train_rate)
        num_val = num_samples - num_test - num_train

        # train
        x_train, y_train = x[:num_train], y[:num_train]
        # val
        x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
        # test
        x_test, y_test = x[-num_test:], y[-num_test:]
        self._logger.info("train\t" + "x: " + str(x_train.shape) + ", y: " + str(y_train.shape))
        self._logger.info("eval\t" + "x: " + str(x_val.shape) + ", y: " + str(y_val.shape))
        self._logger.info("test\t" + "x: " + str(x_test.shape) + ", y: " + str(y_test.shape))

        if self.cache_dataset:
            ensure_dir(self.cache_file_folder)
            np.savez_compressed(
                self.cache_file_name,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                x_val=x_val,
                y_val=y_val,
            )
            self._logger.info("Saved at " + self.cache_file_name)
        return x_train, y_train, x_val, y_val, x_test, y_test

    def _generate_train_val_test(self):
        """
        加载数据集，并划分训练集、测试集、验证集，并缓存数据集

        Returns:
            tuple: tuple contains:
                x_train: (num_samples, input_length, ..., feature_dim) \n
                y_train: (num_samples, input_length, ..., feature_dim) \n
                x_val: (num_samples, input_length, ..., feature_dim) \n
                y_val: (num_samples, input_length, ..., feature_dim) \n
                x_test: (num_samples, input_length, ..., feature_dim) \n
                y_test: (num_samples, input_length, ..., feature_dim)
        """
        x, y = self._generate_data()
        return self._split_train_val_test(x, y)

    def _load_cache_train_val_test(self):
        """
        加载之前缓存好的训练集、测试集、验证集

        Returns:
            tuple: tuple contains:
                x_train: (num_samples, input_length, ..., feature_dim) \n
                y_train: (num_samples, input_length, ..., feature_dim) \n
                x_val: (num_samples, input_length, ..., feature_dim) \n
                y_val: (num_samples, input_length, ..., feature_dim) \n
                x_test: (num_samples, input_length, ..., feature_dim) \n
                y_test: (num_samples, input_length, ..., feature_dim)
        """
        self._logger.info("Loading " + self.cache_file_name)
        cat_data = np.load(self.cache_file_name)
        x_train = cat_data["x_train"]
        y_train = cat_data["y_train"]
        x_test = cat_data["x_test"]
        y_test = cat_data["y_test"]
        x_val = cat_data["x_val"]
        y_val = cat_data["y_val"]
        self._logger.info("train\t" + "x: " + str(x_train.shape) + ", y: " + str(y_train.shape))
        self._logger.info("eval\t" + "x: " + str(x_val.shape) + ", y: " + str(y_val.shape))
        self._logger.info("test\t" + "x: " + str(x_test.shape) + ", y: " + str(y_test.shape))
        return x_train, y_train, x_val, y_val, x_test, y_test

    def _get_scalar(self, scaler_type, x_train, y_train):
        """
        根据全局参数`scaler_type`选择数据归一化方法

        Args:
            x_train: 训练数据X
            y_train: 训练数据y

        Returns:
            Scaler: 归一化对象
        """
        if scaler_type == "normal":
            scaler = NormalScaler(maxx=max(x_train.max(), y_train.max()))
            self._logger.info("NormalScaler max: " + str(scaler.max))
        elif scaler_type == "standard":
            scaler = StandardScaler(mean=x_train.mean(), std=x_train.std())
            self._logger.info("StandardScaler mean: " + str(scaler.mean) + ", std: " + str(scaler.std))
        elif scaler_type == "minmax01":
            scaler = MinMax01Scaler(maxx=max(x_train.max(), y_train.max()), minn=min(x_train.min(), y_train.min()))
            self._logger.info("MinMax01Scaler max: " + str(scaler.max) + ", min: " + str(scaler.min))
        elif scaler_type == "minmax11":
            scaler = MinMax11Scaler(maxx=max(x_train.max(), y_train.max()), minn=min(x_train.min(), y_train.min()))
            self._logger.info("MinMax11Scaler max: " + str(scaler.max) + ", min: " + str(scaler.min))
        elif scaler_type == "log":
            scaler = LogScaler()
            self._logger.info("LogScaler")
        elif scaler_type == "none":
            scaler = NoneScaler()
            self._logger.info("NoneScaler")
        else:
            raise ValueError("Scaler type error!")
        return scaler

    def _add_noise(self, x_test):
        """
        根据全局参数`noise_type`选择噪声类型，并随机加入到测试数据X

        Args:
            x_test: 测试数据X

        Returns:
            x_test_disturbed: 添加噪声后的测试数据X
        """
        if self.noise_type == "zero":
            self._logger.info("zero noise \trate: " + str(self.disturb_rate))
            x_test_disturbed = zero_noise(x_test, self.disturb_rate, self.output_dim)
        elif self.noise_type == "gaussian":
            self._logger.info(
                "gaussian noise \trate: "
                + str(self.disturb_rate)
                + ", mean: "
                + str(self.noise_mean)
                + ", std: "
                + str(self.noise_SD)
            )
            x_test_disturbed = gaussian_noise(
                x_test, self.disturb_rate, self.noise_mean, self.noise_SD, self.output_dim
            )
        else:
            raise ValueError("noise type error!")
        return x_test_disturbed

    def get_data(self):
        """
        返回数据的DataLoader，包括训练数据、测试数据、验证数据

        Returns:
            tuple: tuple contains:
                train_dataloader: Dataloader composed of Batch (class) \n
                eval_dataloader: Dataloader composed of Batch (class) \n
                test_dataloader: Dataloader composed of Batch (class)
        """
        # 加载数据集
        x_train, y_train, x_val, y_val, x_test, y_test = self._load_or_prepare_raw_splits()
        # 在测试集上添加随机扰动
        if self.robustness_test:
            x_test = self._add_noise(x_test)
        # 数据归一化
        self.feature_dim = x_train.shape[-1]
        if self.output_dim > self.feature_dim:
            self._logger.warning(
                "Configured output_dim=%d is larger than feature_dim=%d, fallback to feature_dim.",
                self.output_dim,
                self.feature_dim,
            )
            self.output_dim = self.feature_dim
        self.ext_dim = self.feature_dim - self.output_dim
        self.scaler = self._get_scalar(
            self.scaler_type, x_train[..., : self.output_dim], y_train[..., : self.output_dim]
        )
        self.ext_scaler = self._get_scalar(
            self.ext_scaler_type, x_train[..., self.output_dim:], y_train[..., self.output_dim:]
        )
        x_train[..., : self.output_dim] = self.scaler.transform(x_train[..., : self.output_dim])
        y_train[..., : self.output_dim] = self.scaler.transform(y_train[..., : self.output_dim])
        x_val[..., : self.output_dim] = self.scaler.transform(x_val[..., : self.output_dim])
        y_val[..., : self.output_dim] = self.scaler.transform(y_val[..., : self.output_dim])
        x_test[..., : self.output_dim] = self.scaler.transform(x_test[..., : self.output_dim])
        y_test[..., : self.output_dim] = self.scaler.transform(y_test[..., : self.output_dim])
        if self.normal_external:
            x_train[..., self.output_dim:] = self.ext_scaler.transform(x_train[..., self.output_dim:])
            y_train[..., self.output_dim:] = self.ext_scaler.transform(y_train[..., self.output_dim:])
            x_val[..., self.output_dim:] = self.ext_scaler.transform(x_val[..., self.output_dim:])
            y_val[..., self.output_dim:] = self.ext_scaler.transform(y_val[..., self.output_dim:])
            x_test[..., self.output_dim:] = self.ext_scaler.transform(x_test[..., self.output_dim:])
            y_test[..., self.output_dim:] = self.ext_scaler.transform(y_test[..., self.output_dim:])
        # 把训练集的X和y聚合在一起成为list，测试集验证集同理
        # x_train/y_train: (num_samples, input_length, ..., feature_dim)
        # train_data(list): train_data[i]是一个元组，由x_train[i]和y_train[i]组成
        train_data = (x_train, y_train)
        eval_data = (x_val, y_val)
        test_data = (x_test, y_test)

        # DDP 分布式采样器
        train_sampler = eval_sampler = test_sampler = None
        if self.config.get('is_distributed', False):
            from GNNTP.data.runtime import _make_ddp_samplers
            train_sampler, eval_sampler, test_sampler = _make_ddp_samplers(
                train_data, eval_data, test_data,
                self.config['world_size'], self.config['rank'],
            )

        # 转Dataloader
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = generate_dataloader(
            train_data,
            eval_data,
            test_data,
            self.feature_name,
            self.batch_size,
            self.num_workers,
            pad_with_last_sample=(
                False if self.config.get('is_distributed', False) else self.pad_with_last_sample
            ),
            train_sampler=train_sampler,
            eval_sampler=eval_sampler,
            test_sampler=test_sampler,
        )
        self.num_batches = len(self.train_dataloader)
        return self.train_dataloader, self.eval_dataloader, self.test_dataloader
