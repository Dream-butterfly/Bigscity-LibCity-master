import os
import sys
import numpy as np

from GNNTP.data.dataset import TrafficStatePointDataset
from GNNTP.utils import get_dataset_cache_dir


class ASTGCNDataset(TrafficStatePointDataset):
    """ASTGCN 专用数据集，实现 closeness/period/trend 三段 CPT 采样策略。

    与标准滑动窗口不同，ASTGCN 从三个时间尺度采样:
      - closeness: 前 len_closeness 个最近的 output_window 段
      - period: 前 len_period 天同一时刻的 output_window 段
      - trend: 前 len_trend 周同一时刻的 output_window 段

    Migrated from Bigscity-LibCity-master/libcity/data/dataset/dataset_subclass/astgcn_dataset.py
    """

    def __init__(self, config):
        self.len_closeness = config.get('len_closeness', 3)
        self.len_period = config.get('len_period', 4)
        self.len_trend = config.get('len_trend', 0)
        self.interval_period = config.get('interval_period', 1)  # period 的间隔/天
        self.interval_trend = config.get('interval_trend', 7)    # trend 的间隔/天
        assert (self.len_closeness + self.len_period + self.len_trend > 0)

        super().__init__(config)

        self.points_per_hour = 3600 // self.time_intervals
        self.feature_name = {'X': 'float', 'y': 'float'}
        self.parameters_str = (
            str(self.dataset) + '_' + str(self.len_closeness)
            + '_' + str(self.len_period) + '_' + str(self.len_trend)
            + '_' + str(self.interval_period) + '_' + str(self.interval_trend)
            + '_' + str(self.output_window) + '_' + str(self.train_rate)
            + '_' + str(self.eval_rate) + '_' + str(self.scaler_type)
            + '_' + str(self.batch_size) + '_' + str(self.add_time_in_day)
            + '_' + str(self.add_day_in_week) + '_' + str(self.pad_with_last_sample)
        )
        self.cache_file_name = os.path.join(
            get_dataset_cache_dir(),
            'astgcn_point_based_{}.npz'.format(self.parameters_str)
        )

    def _search_data(self, sequence_length, label_start_idx, num_for_predict, num_of_depend, units):
        """根据 CPT 参数在时间序列中定位数据索引。

        Args:
            sequence_length: 历史数据的总长度
            label_start_idx: 预测开始的时间片索引
            num_for_predict: 预测的时间片序列长度
            num_of_depend: len_trend/len_period/len_closeness
            units: trend/period/closeness 的长度(以小时为单位)

        Returns:
            list[(start_idx, end_idx)] 或 None
        """
        if self.points_per_hour <= 0:
            raise ValueError("points_per_hour should be greater than 0!")
        if label_start_idx + num_for_predict > sequence_length:
            return None
        x_idx = []
        for i in range(1, num_of_depend + 1):
            start_idx = label_start_idx - self.points_per_hour * units * i
            end_idx = start_idx + num_for_predict
            if start_idx >= 0:
                x_idx.append((start_idx, end_idx))
            else:
                return None
        if len(x_idx) != num_of_depend:
            return None
        return x_idx[::-1]  # 从左至右

    def _get_sample_indices(self, data_sequence, label_start_idx):
        """获取 CPT 三段采样的索引。

        Returns:
            trend_sample, period_sample, closeness_sample, target
            或 (None, None, None, None)
        """
        trend_sample, period_sample, closeness_sample = None, None, None
        if label_start_idx + self.output_window > data_sequence.shape[0]:
            return trend_sample, period_sample, closeness_sample, None

        if self.len_trend > 0:
            trend_indices = self._search_data(
                data_sequence.shape[0], label_start_idx, self.output_window,
                self.len_trend, self.interval_trend * 24,
            )
            if not trend_indices:
                return None, None, None, None
            trend_sample = np.concatenate(
                [data_sequence[i: j] for i, j in trend_indices], axis=0)

        if self.len_period > 0:
            period_indices = self._search_data(
                data_sequence.shape[0], label_start_idx, self.output_window,
                self.len_period, self.interval_period * 24,
            )
            if not period_indices:
                return None, None, None, None
            period_sample = np.concatenate(
                [data_sequence[i: j] for i, j in period_indices], axis=0)

        if self.len_closeness > 0:
            closeness_indices = self._search_data(
                data_sequence.shape[0], label_start_idx, self.output_window,
                self.len_closeness, 1,
            )
            if not closeness_indices:
                return None, None, None, None
            closeness_sample = np.concatenate(
                [data_sequence[i: j] for i, j in closeness_indices], axis=0)

        target = data_sequence[label_start_idx: label_start_idx + self.output_window]
        return trend_sample, period_sample, closeness_sample, target

    def _generate_input_data(self, df):
        """使用 CPT 采样策略生成输入数据。

        覆盖父类的滑动窗口方法。

        Args:
            df: 输入数据, shape: (len_time, ..., feature_dim)

        Returns:
            sources: (num_samples, Tw+Td+Th, ..., feature_dim)
            targets: (num_samples, Tp, ..., feature_dim)
        """
        trend_samples, period_samples, closeness_samples, targets = [], [], [], []
        flag = 0
        for idx in range(df.shape[0]):
            sample = self._get_sample_indices(df, idx)
            if (sample[0] is None) and (sample[1] is None) and (sample[2] is None):
                continue
            flag = 1
            trend_sample, period_sample, closeness_sample, target = sample
            if self.len_trend > 0:
                trend_sample = np.expand_dims(trend_sample, axis=0)
                trend_samples.append(trend_sample)
            if self.len_period > 0:
                period_sample = np.expand_dims(period_sample, axis=0)
                period_samples.append(period_sample)
            if self.len_closeness > 0:
                closeness_sample = np.expand_dims(closeness_sample, axis=0)
                closeness_samples.append(closeness_sample)
            target = np.expand_dims(target, axis=0)
            targets.append(target)

        if flag == 0:
            self._logger.warning(
                'Parameter len_closeness/len_period/len_trend is too large '
                'for the time range of the data!')
            sys.exit()

        sources = []
        if len(closeness_samples) > 0:
            closeness_samples = np.concatenate(closeness_samples, axis=0)
            sources.append(closeness_samples)
            self._logger.info('closeness: ' + str(closeness_samples.shape))
        if len(period_samples) > 0:
            period_samples = np.concatenate(period_samples, axis=0)
            sources.append(period_samples)
            self._logger.info('period: ' + str(period_samples.shape))
        if len(trend_samples) > 0:
            trend_samples = np.concatenate(trend_samples, axis=0)
            sources.append(trend_samples)
            self._logger.info('trend: ' + str(trend_samples.shape))
        sources = np.concatenate(sources, axis=1)  # (num_samples, Tw+Td+Th, N, F)
        targets = np.concatenate(targets, axis=0)  # (num_samples, Tp, N, F)
        return sources, targets

    def get_data_feature(self):
        """返回数据集特征，包含 CPT 三段长度。"""
        return {
            "scaler": self.scaler,
            "adj_mx": self.adj_mx,
            "num_nodes": self.num_nodes,
            "feature_dim": self.feature_dim,
            "output_dim": self.output_dim,
            "ext_dim": self.ext_dim,
            "len_closeness": self.len_closeness * self.output_window,
            "len_period": self.len_period * self.output_window,
            "len_trend": self.len_trend * self.output_window,
            "num_batches": self.num_batches,
        }
