import numpy as np
import pandas as pd


class TrafficStateTemporalLoaderMixin:
    def _load_dyna_3d(self, filename):
        """
        加载.dyna文件，格式[dyna_id, type, time, entity_id, properties(若干列)],
        .geo文件中的id顺序应该跟.dyna中一致,
        其中全局参数`data_col`用于指定需要加载的数据的列，不设置则默认全部加载

        Args:
            filename(str): 数据文件名，不包含后缀

        Returns:
            np.ndarray: 数据数组, 3d-array: (len_time, num_nodes, feature_dim)
        """
        # 加载数据集
        self._logger.info("Loading file " + filename + ".dyna")
        dynafile = pd.read_csv(self.data_path + filename + ".dyna")
        if self.data_col != "":  # 根据指定的列加载数据集
            if isinstance(self.data_col, list):
                data_col = self.data_col.copy()
            else:  # str
                data_col = [self.data_col].copy()
            data_col.insert(0, "time")
            data_col.insert(1, "entity_id")
            missing_cols = [col for col in data_col if col not in dynafile.columns]
            if len(missing_cols) > 0:
                self._logger.warning(
                    "Configured data_col contains columns not in %s.dyna: %s. Fallback to all feature columns.",
                    filename,
                    missing_cols,
                )
                dynafile = dynafile[dynafile.columns[2:]]  # 从time列开始所有列
            else:
                dynafile = dynafile[data_col]
        else:  # 不指定则加载所有列
            dynafile = dynafile[dynafile.columns[2:]]  # 从time列开始所有列
        # 求时间序列
        self.timesolts = list(dynafile["time"][: int(dynafile.shape[0] / len(self.geo_ids))])
        self.idx_of_timesolts = dict()
        if not dynafile["time"].isna().any():  # 时间没有空值
            self.timesolts = list(map(lambda x: x.replace("T", " ").replace("Z", ""), self.timesolts))
            self.timesolts = np.array(self.timesolts, dtype="datetime64[ns]")
            for idx, _ts in enumerate(self.timesolts):
                self.idx_of_timesolts[_ts] = idx
        # 转3-d数组
        feature_dim = len(dynafile.columns) - 2
        df = dynafile[dynafile.columns[-feature_dim:]]
        len_time = len(self.timesolts)
        data = []
        for i in range(0, df.shape[0], len_time):
            data.append(df[i: i + len_time].values)
        data = np.array(data, dtype=np.float32)  # (len(self.geo_ids), len_time, feature_dim)
        data = data.swapaxes(0, 1)  # (len_time, len(self.geo_ids), feature_dim)
        self._logger.info("Loaded file " + filename + ".dyna" + ", shape=" + str(data.shape))
        return data

    def _load_grid_3d(self, filename):
        """
        加载.grid文件，格式[dyna_id, type, time, row_id, column_id, properties(若干列)],
        .geo文件中的id顺序应该跟.dyna中一致,
        其中全局参数`data_col`用于指定需要加载的数据的列，不设置则默认全部加载,

        Args:
            filename(str): 数据文件名，不包含后缀

        Returns:
            np.ndarray: 数据数组, 3d-array: (len_time, num_grids, feature_dim)
        """
        # 加载数据集
        self._logger.info("Loading file " + filename + ".grid")
        gridfile = pd.read_csv(self.data_path + filename + ".grid")
        if self.data_col != "":  # 根据指定的列加载数据集
            if isinstance(self.data_col, list):
                data_col = self.data_col.copy()
            else:  # str
                data_col = [self.data_col].copy()
            data_col.insert(0, "time")
            data_col.insert(1, "row_id")
            data_col.insert(2, "column_id")
            gridfile = gridfile[data_col]
        else:  # 不指定则加载所有列
            gridfile = gridfile[gridfile.columns[2:]]  # 从time列开始所有列
        # 求时间序列
        self.timesolts = list(gridfile["time"][: int(gridfile.shape[0] / len(self.geo_ids))])
        self.idx_of_timesolts = dict()
        if not gridfile["time"].isna().any():  # 时间没有空值
            self.timesolts = list(map(lambda x: x.replace("T", " ").replace("Z", ""), self.timesolts))
            self.timesolts = np.array(self.timesolts, dtype="datetime64[ns]")
            for idx, _ts in enumerate(self.timesolts):
                self.idx_of_timesolts[_ts] = idx
        # 转3-d数组
        feature_dim = len(gridfile.columns) - 3
        df = gridfile[gridfile.columns[-feature_dim:]]
        len_time = len(self.timesolts)
        data = []
        for i in range(0, df.shape[0], len_time):
            data.append(df[i: i + len_time].values)
        data = np.array(data, dtype=np.float32)  # (len(self.geo_ids), len_time, feature_dim)
        data = data.swapaxes(0, 1)  # (len_time, len(self.geo_ids), feature_dim)
        self._logger.info("Loaded file " + filename + ".grid" + ", shape=" + str(data.shape))
        return data

    def _load_grid_4d(self, filename):
        """
        加载.grid文件，格式[dyna_id, type, time, row_id, column_id, properties(若干列)],
        .geo文件中的id顺序应该跟.dyna中一致,
        其中全局参数`data_col`用于指定需要加载的数据的列，不设置则默认全部加载

        Args:
            filename(str): 数据文件名，不包含后缀

        Returns:
            np.ndarray: 数据数组, 4d-array: (len_time, len_row, len_column, feature_dim)
        """
        # 加载数据集
        self._logger.info("Loading file " + filename + ".grid")
        gridfile = pd.read_csv(self.data_path + filename + ".grid")
        if self.data_col != "":  # 根据指定的列加载数据集
            if isinstance(self.data_col, list):
                data_col = self.data_col.copy()
            else:  # str
                data_col = [self.data_col].copy()
            data_col.insert(0, "time")
            data_col.insert(1, "row_id")
            data_col.insert(2, "column_id")
            gridfile = gridfile[data_col]
        else:  # 不指定则加载所有列
            gridfile = gridfile[gridfile.columns[2:]]  # 从time列开始所有列
        # 求时间序列
        self.timesolts = list(gridfile["time"][: int(gridfile.shape[0] / len(self.geo_ids))])
        self.idx_of_timesolts = dict()
        if not gridfile["time"].isna().any():  # 时间没有空值
            self.timesolts = list(map(lambda x: x.replace("T", " ").replace("Z", ""), self.timesolts))
            self.timesolts = np.array(self.timesolts, dtype="datetime64[ns]")
            for idx, _ts in enumerate(self.timesolts):
                self.idx_of_timesolts[_ts] = idx
        # 转4-d数组
        feature_dim = len(gridfile.columns) - 3
        df = gridfile[gridfile.columns[-feature_dim:]]
        len_time = len(self.timesolts)
        data = []
        for i in range(self.len_row):
            tmp = []
            for j in range(self.len_column):
                index = (i * self.len_column + j) * len_time
                tmp.append(df[index: index + len_time].values)
            data.append(tmp)
        data = np.array(data, dtype=np.float32)  # (len_row, len_column, len_time, feature_dim)
        data = data.swapaxes(2, 0).swapaxes(1, 2)  # (len_time, len_row, len_column, feature_dim)
        self._logger.info("Loaded file " + filename + ".grid" + ", shape=" + str(data.shape))
        return data

    def _load_od_4d(self, filename):
        """
        加载.od文件，格式[dyna_id, type, time, origin_id, destination_id properties(若干列)],
        .geo文件中的id顺序应该跟.dyna中一致,
        其中全局参数`data_col`用于指定需要加载的数据的列，不设置则默认全部加载

        Args:
            filename(str): 数据文件名，不包含后缀

        Returns:
            np.ndarray: 数据数组, 4d-array: (len_time, len_row, len_column, feature_dim)
        """
        self._logger.info("Loading file " + filename + ".od")
        odfile = pd.read_csv(self.data_path + filename + ".od")
        if self.data_col != "":  # 根据指定的列加载数据集
            if isinstance(self.data_col, list):
                data_col = self.data_col.copy()
            else:  # str
                data_col = [self.data_col].copy()
            data_col.insert(0, "time")
            data_col.insert(1, "origin_id")
            data_col.insert(2, "destination_id")
            odfile = odfile[data_col]
        else:  # 不指定则加载所有列
            odfile = odfile[odfile.columns[2:]]  # 从time列开始所有列
        # 求时间序列
        self.timesolts = list(odfile["time"][: int(odfile.shape[0] / self.num_nodes / self.num_nodes)])
        self.idx_of_timesolts = dict()
        if not odfile["time"].isna().any():  # 时间没有空值
            self.timesolts = list(map(lambda x: x.replace("T", " ").replace("Z", ""), self.timesolts))
            self.timesolts = np.array(self.timesolts, dtype="datetime64[ns]")
            for idx, _ts in enumerate(self.timesolts):
                self.idx_of_timesolts[_ts] = idx

        feature_dim = len(odfile.columns) - 3
        df = odfile[odfile.columns[-feature_dim:]]
        len_time = len(self.timesolts)
        data = np.zeros((self.num_nodes, self.num_nodes, len_time, feature_dim))
        for i in range(self.num_nodes):
            origin_index = i * len_time * self.num_nodes  # 每个起点占据len_t*n行
            for j in range(self.num_nodes):
                destination_index = j * len_time  # 每个终点占据len_t行
                index = origin_index + destination_index
                data[i][j] = df[index: index + len_time].values
        data = data.transpose((2, 0, 1, 3))  # (len_time, num_nodes, num_nodes, feature_dim)
        self._logger.info("Loaded file " + filename + ".od" + ", shape=" + str(data.shape))
        return data

    def _load_grid_od_4d(self, filename):
        """
        加载.gridod文件，格式[dyna_id, type, time, origin_row_id, origin_column_id,
        destination_row_id, destination_column_id, properties(若干列)],
        .geo文件中的id顺序应该跟.dyna中一致,
        其中全局参数`data_col`用于指定需要加载的数据的列，不设置则默认全部加载

        Args:
            filename(str): 数据文件名，不包含后缀

        Returns:
            np.ndarray: 数据数组, 4d-array: (len_time, num_grids, num_grids, feature_dim)
        """
        # 加载数据集
        self._logger.info("Loading file " + filename + ".gridod")
        gridodfile = pd.read_csv(self.data_path + filename + ".gridod")
        if self.data_col != "":  # 根据指定的列加载数据集
            if isinstance(self.data_col, list):
                data_col = self.data_col.copy()
            else:  # str
                data_col = [self.data_col].copy()
            data_col.insert(0, "time")
            data_col.insert(1, "origin_row_id")
            data_col.insert(2, "origin_column_id")
            data_col.insert(3, "destination_row_id")
            data_col.insert(4, "destination_column_id")
            gridodfile = gridodfile[data_col]
        else:  # 不指定则加载所有列
            gridodfile = gridodfile[gridodfile.columns[2:]]  # 从time列开始所有列
        # 求时间序列
        self.timesolts = list(
            gridodfile["time"][: int(gridodfile.shape[0] / len(self.geo_ids) / len(self.geo_ids))]
        )
        self.idx_of_timesolts = dict()
        if not gridodfile["time"].isna().any():  # 时间没有空值
            self.timesolts = list(map(lambda x: x.replace("T", " ").replace("Z", ""), self.timesolts))
            self.timesolts = np.array(self.timesolts, dtype="datetime64[ns]")
            for idx, _ts in enumerate(self.timesolts):
                self.idx_of_timesolts[_ts] = idx
        # 转4-d数组
        feature_dim = len(gridodfile.columns) - 5
        df = gridodfile[gridodfile.columns[-feature_dim:]]
        len_time = len(self.timesolts)
        data = np.zeros((len(self.geo_ids), len(self.geo_ids), len_time, feature_dim))
        for oi in range(self.len_row):
            for oj in range(self.len_column):
                origin_index = (oi * self.len_column + oj) * len_time * len(self.geo_ids)  # 每个起点占据len_t*n行
                for di in range(self.len_row):
                    for dj in range(self.len_column):
                        destination_index = (di * self.len_column + dj) * len_time  # 每个终点占据len_t行
                        index = origin_index + destination_index
                        data[oi * self.len_column + oj][di * self.len_column + dj] = df[
                            index: index + len_time
                        ].values
        data = data.transpose((2, 0, 1, 3))  # (len_time, num_grids, num_grids, feature_dim)
        self._logger.info("Loaded file " + filename + ".gridod" + ", shape=" + str(data.shape))
        return data

    def _load_grid_od_6d(self, filename):
        """
        加载.gridod文件，格式[dyna_id, type, time, origin_row_id, origin_column_id,
        destination_row_id, destination_column_id, properties(若干列)],
        .geo文件中的id顺序应该跟.dyna中一致,
        其中全局参数`data_col`用于指定需要加载的数据的列，不设置则默认全部加载

        Args:
            filename(str): 数据文件名，不包含后缀

        Returns:
            np.ndarray: 数据数组, 6d-array: (len_time, len_row, len_column, len_row, len_column, feature_dim)
        """
        # 加载数据集
        self._logger.info("Loading file " + filename + ".gridod")
        gridodfile = pd.read_csv(self.data_path + filename + ".gridod")
        if self.data_col != "":  # 根据指定的列加载数据集
            if isinstance(self.data_col, list):
                data_col = self.data_col.copy()
            else:  # str
                data_col = [self.data_col].copy()
            data_col.insert(0, "time")
            data_col.insert(1, "origin_row_id")
            data_col.insert(2, "origin_column_id")
            data_col.insert(3, "destination_row_id")
            data_col.insert(4, "destination_column_id")
            gridodfile = gridodfile[data_col]
        else:  # 不指定则加载所有列
            gridodfile = gridodfile[gridodfile.columns[2:]]  # 从time列开始所有列
        # 求时间序列
        self.timesolts = list(
            gridodfile["time"][: int(gridodfile.shape[0] / len(self.geo_ids) / len(self.geo_ids))]
        )
        self.idx_of_timesolts = dict()
        if not gridodfile["time"].isna().any():  # 时间没有空值
            self.timesolts = list(map(lambda x: x.replace("T", " ").replace("Z", ""), self.timesolts))
            self.timesolts = np.array(self.timesolts, dtype="datetime64[ns]")
            for idx, _ts in enumerate(self.timesolts):
                self.idx_of_timesolts[_ts] = idx
        # 转6-d数组
        feature_dim = len(gridodfile.columns) - 5
        df = gridodfile[gridodfile.columns[-feature_dim:]]
        len_time = len(self.timesolts)
        data = np.zeros((self.len_row, self.len_column, self.len_row, self.len_column, len_time, feature_dim))
        for oi in range(self.len_row):
            for oj in range(self.len_column):
                origin_index = (oi * self.len_column + oj) * len_time * len(self.geo_ids)  # 每个起点占据len_t*n行
                for di in range(self.len_row):
                    for dj in range(self.len_column):
                        destination_index = (di * self.len_column + dj) * len_time  # 每个终点占据len_t行
                        index = origin_index + destination_index
                        data[oi][oj][di][dj] = df[index: index + len_time].values
        data = data.transpose((4, 0, 1, 2, 3, 5))  # (len_time, len_row, len_column, len_row, len_column, feature_dim)
        self._logger.info("Loaded file " + filename + ".gridod" + ", shape=" + str(data.shape))
        return data

    def _load_ext(self):
        """
        加载.ext文件，格式[ext_id, time, properties(若干列)],
        其中全局参数`ext_col`用于指定需要加载的数据的列，不设置则默认全部加载

        Returns:
            np.ndarray: 外部数据数组，shape: (timeslots, ext_dim)
        """
        # 加载数据集
        extfile = pd.read_csv(self.data_path + self.ext_file + ".ext")
        if self.ext_col != "":  # 根据指定的列加载数据集
            if isinstance(self.ext_col, list):
                ext_col = self.ext_col.copy()
            else:  # str
                ext_col = [self.ext_col].copy()
            ext_col.insert(0, "time")
            missing_cols = [col for col in ext_col if col not in extfile.columns]
            if len(missing_cols) > 0:
                self._logger.warning(
                    "Configured ext_col contains columns not in %s.ext: %s. Fallback to all ext columns.",
                    self.ext_file,
                    missing_cols,
                )
                extfile = extfile[extfile.columns[1:]]  # 从time列开始所有列
            else:
                extfile = extfile[ext_col]
        else:  # 不指定则加载所有列
            extfile = extfile[extfile.columns[1:]]  # 从time列开始所有列
        # 求时间序列
        self.ext_timesolts = extfile["time"]
        self.idx_of_ext_timesolts = dict()
        if not extfile["time"].isna().any():  # 时间没有空值
            self.ext_timesolts = list(map(lambda x: x.replace("T", " ").replace("Z", ""), self.ext_timesolts))
            self.ext_timesolts = np.array(self.ext_timesolts, dtype="datetime64[ns]")
            for idx, _ts in enumerate(self.ext_timesolts):
                self.idx_of_ext_timesolts[_ts] = idx
        # 求外部特征数组
        feature_dim = len(extfile.columns) - 1
        df = extfile[extfile.columns[-feature_dim:]].values
        self._logger.info("Loaded file " + self.ext_file + ".ext" + ", shape=" + str(df.shape))
        return df
