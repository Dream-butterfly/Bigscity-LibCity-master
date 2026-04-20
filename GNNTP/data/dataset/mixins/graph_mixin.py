import numpy as np
import pandas as pd


class TrafficStateGraphMixin:
    def _load_geo(self):
        """
        加载.geo文件，格式[geo_id, type, coordinates, properties(若干列)]
        """
        geofile = pd.read_csv(self.data_path + self.geo_file + ".geo")
        self.geo_ids = list(geofile["geo_id"])
        self.num_nodes = len(self.geo_ids)
        self.geo_to_ind = {}
        self.ind_to_geo = {}
        for index, idx in enumerate(self.geo_ids):
            self.geo_to_ind[idx] = index
            self.ind_to_geo[index] = idx
        self._logger.info("Loaded file " + self.geo_file + ".geo" + ", num_nodes=" + str(len(self.geo_ids)))

    def _load_grid_geo(self):
        """
        加载.geo文件，格式[geo_id, type, coordinates, row_id, column_id, properties(若干列)]
        """
        geofile = pd.read_csv(self.data_path + self.geo_file + ".geo")
        self.geo_ids = list(geofile["geo_id"])
        self.num_nodes = len(self.geo_ids)
        self.geo_to_ind = {}
        self.geo_to_rc = {}
        for index, idx in enumerate(self.geo_ids):
            self.geo_to_ind[idx] = index
        for i in range(geofile.shape[0]):
            self.geo_to_rc[geofile["geo_id"][i]] = [geofile["row_id"][i], geofile["column_id"][i]]
        self.len_row = max(list(geofile["row_id"])) + 1
        self.len_column = max(list(geofile["column_id"])) + 1
        self._logger.info(
            "Loaded file "
            + self.geo_file
            + ".geo"
            + ", num_grids="
            + str(len(self.geo_ids))
            + ", grid_size="
            + str((self.len_row, self.len_column))
        )

    def _load_rel(self):
        """
        加载.rel文件，格式[rel_id, type, origin_id, destination_id, properties(若干列)],
        生成N*N的邻接矩阵，计算逻辑如下：
        (1) 权重所对应的列名用全局参数`weight_col`来指定, \
        (2) 若没有指定该参数, \
            (2.1) rel只有4列，则认为rel中的每一行代表一条邻接边，权重为1。其余边权重为0，代表不邻接。 \
            (2.2) rel只有5列，则默认最后一列为`weight_col` \
            (2.3) 否则报错 \
        (3) 根据得到的权重列`weight_col`计算邻接矩阵 \
            (3.1) 参数`bidir_adj_mx`=True代表构造无向图，=False为有向图 \
            (3.2) 参数`set_weight_link_or_dist`为`link`代表构造01矩阵，为`dist`代表构造权重矩阵（非01） \
            (3.3) 参数`init_weight_inf_or_zero`为`zero`代表矩阵初始化为全0，`inf`代表矩阵初始化成全inf，初始化值也就是rel文件中不存在的边的权值 \
            (3.4) 参数`calculate_weight_adj`=True表示对权重矩阵应用带阈值的高斯核函数进行稀疏化，对01矩阵不做处理，=False不进行稀疏化，
            修改函数self._calculate_adjacency_matrix()可以构造其他方法替换全阈值高斯核的稀疏化方法 \

        Returns:
            np.ndarray: self.adj_mx, N*N的邻接矩阵
        """
        relfile = pd.read_csv(self.data_path + self.rel_file + ".rel")
        self._logger.info("set_weight_link_or_dist: {}".format(self.set_weight_link_or_dist))
        self._logger.info("init_weight_inf_or_zero: {}".format(self.init_weight_inf_or_zero))
        if self.weight_col != "":  # 根据weight_col确认权重列
            if isinstance(self.weight_col, list):
                if len(self.weight_col) != 1:
                    raise ValueError("`weight_col` parameter must be only one column!")
                self.weight_col = self.weight_col[0]
            if self.weight_col in relfile.columns:
                self.distance_df = relfile[~relfile[self.weight_col].isna()][
                    ["origin_id", "destination_id", self.weight_col]
                ]
            else:
                self._logger.warning(
                    "Configured weight_col `%s` not in %s.rel columns. Fallback to auto-detect.",
                    self.weight_col,
                    self.rel_file,
                )
                self.weight_col = ""
        if self.weight_col == "":
            if len(relfile.columns) > 5 or len(relfile.columns) < 4:  # properties不只一列，且未指定weight_col，报错
                raise ValueError("Don't know which column to be loaded! Please set `weight_col` parameter!")
            elif len(relfile.columns) == 4:  # 4列说明没有properties列，那就是rel文件中有的代表相邻，否则不相邻
                self.calculate_weight_adj = False
                self.set_weight_link_or_dist = "link"
                self.init_weight_inf_or_zero = "zero"
                self.distance_df = relfile[["origin_id", "destination_id"]]
            else:  # len(relfile.columns) == 5, properties只有一列，那就默认这一列是权重列
                self.weight_col = relfile.columns[-1]
                self.distance_df = relfile[~relfile[self.weight_col].isna()][
                    ["origin_id", "destination_id", self.weight_col]
                ]
        # 把数据转换成矩阵的形式
        self.adj_mx = np.zeros((len(self.geo_ids), len(self.geo_ids)), dtype=np.float32)
        if self.init_weight_inf_or_zero.lower() == "inf" and self.set_weight_link_or_dist.lower() != "link":
            self.adj_mx[:] = np.inf
        for row in self.distance_df.values:
            if row[0] not in self.geo_to_ind or row[1] not in self.geo_to_ind:
                continue
            if self.set_weight_link_or_dist.lower() == "dist":  # 保留原始的距离数值
                self.adj_mx[self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]] = row[2]
                if self.bidir_adj_mx:
                    self.adj_mx[self.geo_to_ind[row[1]], self.geo_to_ind[row[0]]] = row[2]
            else:  # self.set_weight_link_or_dist.lower()=='link' 只保留01的邻接性
                self.adj_mx[self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]] = 1
                if self.bidir_adj_mx:
                    self.adj_mx[self.geo_to_ind[row[1]], self.geo_to_ind[row[0]]] = 1
        self._logger.info("Loaded file " + self.rel_file + ".rel, shape=" + str(self.adj_mx.shape))
        # 计算权重
        if self.distance_inverse and self.set_weight_link_or_dist.lower() != "link":
            self._distance_inverse()
        elif self.calculate_weight_adj and self.set_weight_link_or_dist.lower() != "link":
            self._calculate_adjacency_matrix()

    def _load_grid_rel(self):
        """
        根据网格结构构建邻接矩阵，一个格子跟他周围的8个格子邻接

        Returns:
            np.ndarray: self.adj_mx, N*N的邻接矩阵
        """
        self.adj_mx = np.zeros((len(self.geo_ids), len(self.geo_ids)), dtype=np.float32)
        dirs = [[0, 1], [1, 0], [-1, 0], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
        for i in range(self.len_row):
            for j in range(self.len_column):
                index = i * self.len_column + j  # grid_id
                for d in dirs:
                    nei_i = i + d[0]
                    nei_j = j + d[1]
                    if nei_i >= 0 and nei_i < self.len_row and nei_j >= 0 and nei_j < self.len_column:
                        nei_index = nei_i * self.len_column + nei_j  # neighbor_grid_id
                        self.adj_mx[index][nei_index] = 1
                        self.adj_mx[nei_index][index] = 1
        self._logger.info("Generate grid rel file, shape=" + str(self.adj_mx.shape))

    def _calculate_adjacency_matrix(self):
        """
        使用带有阈值的高斯核计算邻接矩阵的权重，如果有其他的计算方法，可以覆盖这个函数,
        公式为：$ w_{ij} = \\exp \\left(- \\frac{d_{ij}^{2}}{\\sigma^{2}} \\right) $, $\\sigma$ 是方差,
        小于阈值`weight_adj_epsilon`的值设为0：$  w_{ij}[w_{ij}<\\epsilon]=0 $

        Returns:
            np.ndarray: self.adj_mx, N*N的邻接矩阵
        """
        self._logger.info("Start Calculate the weight by Gauss kernel!")
        distances = self.adj_mx[~np.isinf(self.adj_mx)].flatten()
        std = distances.std()
        self.adj_mx = np.exp(-np.square(self.adj_mx / std))
        self.adj_mx[self.adj_mx < self.weight_adj_epsilon] = 0

    def _distance_inverse(self):
        self._logger.info("Start Calculate the weight by _distance_inverse!")
        self.adj_mx = 1 / self.adj_mx
        self.adj_mx[np.isinf(self.adj_mx)] = 1
