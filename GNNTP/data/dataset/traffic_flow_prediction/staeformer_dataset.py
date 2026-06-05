import datetime
import numpy as np
import pandas as pd

from GNNTP.data.dataset import TrafficStatePointDataset


class STAEformerDataset(TrafficStatePointDataset):
    """STAEformer 专用数据集，提供非 one-hot 的 TOD/DOW 编码。

    与标准 TrafficStatePointDataset 不同:
      - TOD: float (0-1) 而非 one-hot/multi-hot
      - DOW: int (0-6) 而非 one-hot (7-dim)

    STAEformer 使用 nn.Embedding 将 TOD/DOW 直接映射到嵌入空间，
    因此需要整数索引而非 one-hot 向量。

    Migrated from Bigscity-LibCity-master/libcity/data/dataset/dataset_subclass/staeformer_dataset.py
    """

    def __init__(self, config):
        super().__init__(config)

    def _add_external_information(self, df, ext_data=None):
        return self._add_external_information_3d(df, ext_data)

    def _add_external_information_3d(self, df, ext_data=None):
        """增加外部信息，TOD 为 float (0-1)，DOW 为 int (0-6)。"""
        num_samples, num_nodes, feature_dim = df.shape
        is_time_nan = pd.DatetimeIndex(self.timesolts).isnull().any()
        data_list = [df]
        if self.add_time_in_day and not is_time_nan:
            time_ind = (self.timesolts - self.timesolts.astype("datetime64[D]")) / np.timedelta64(1, "D")
            time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
            data_list.append(time_in_day)
        if self.add_day_in_week and not is_time_nan:
            dayofweek = []
            for day in self.timesolts.astype("datetime64[D]"):
                dayofweek.append(datetime.datetime.strptime(str(day), '%Y-%m-%d').weekday())
            day_in_week = np.tile(dayofweek, [1, num_nodes, 1]).transpose((2, 1, 0))
            data_list.append(day_in_week)
        # 外部数据集
        if ext_data is not None:
            if not is_time_nan:
                indexs = []
                for ts in self.timesolts:
                    ts_index = self.idx_of_ext_timesolts[ts]
                    indexs.append(ts_index)
                select_data = ext_data[indexs]  # T * ext_dim
                for i in range(select_data.shape[1]):
                    data_ind = select_data[:, i]
                    data_ind = np.tile(data_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
                    data_list.append(data_ind)
            else:
                if ext_data.shape[0] == df.shape[0]:
                    select_data = ext_data  # T * ext_dim
                    for i in range(select_data.shape[1]):
                        data_ind = select_data[:, i]
                        data_ind = np.tile(data_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
                        data_list.append(data_ind)
        data = np.concatenate(data_list, axis=-1)
        return data
