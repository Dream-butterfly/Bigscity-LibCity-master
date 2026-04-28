# new_diffusion_fuzzy/INFO.md

## 目录职责

实现 **FuzDiff**（Fuzzy Graph Diffusion）实验模型——以多头自注意力为核心计算原语、以条件扩散为生成框架、以模糊图学习和物理守恒为增强机制的交通预测模型。对应论文投稿目标：Information Sciences。

## 关键文件

| 文件 | 作用 |
| --- | --- |
| `model.py` | 模型结构与前向逻辑（~870 行） |
| `utils/adjacency.py` | 邻接矩阵归一化与批量扩展工具 |
| `utils/attention_ops.py` | 时空注意力张量变换辅助逻辑 |
| `utils/time_embedding.py` | 扩散时间步正弦嵌入 |
| `utils/__init__.py` | 工具模块导出聚合 |
| `config.json` | 默认实验参数（58 项配置） |
| `manifest.json` | 模型注册元信息 |

## 核心架构

### 条件编码器（STEncoder）
- $L_e=2$ 层 STEncoderBlock，每层含：逐节点时间自注意力 → $K=2$ 跳图卷积 → FFN（$D \to D_{\rm ffn} \to D$，GELU）
- 自注意力：$h=4$ 头，$d_k=24$，PyTorch scaled_dot_product_attention
- 图卷积：对称归一化 $\hat{\mathbf{A}} = \mathbf{D}^{-1/2}(\mathbf{A}+\mathbf{I})\mathbf{D}^{-1/2}$，$K$ 阶扩散聚合
- Pre-LayerNorm 残差 + Dropout($p=0.1$) + 梯度检查点

### 模糊自适应图学习器
- 节点表征：特征投影 + 节点嵌入 + 扩散时间步嵌入 → tanh
- 高斯隶属函数：$\mu_m(s) = \exp(-\frac{1}{2}(\frac{\tanh(s)-c_m}{\sigma_m})^2)$
- $M=3$ 个模糊子集，$c_m$ 和 $\sigma_m$ 端到端可学习
- 可学习规则权重 $w_m$（softmax 归一化）聚合隶属度
- Top-$k=12$ 稀疏化 + 动态-静态融合权重 $\alpha$

### 注意力去噪器（AttentionDenoiser）
- $L_d=3$ 层 DenoiserBlock，每层含 5 个子层
- 三种注意力形态：逐节点时间自注意力、逐节点交叉注意力、全展平时空注意力
- 每层前动态计算模糊自适应图 $\mathbf{A}^{\rm adapt}$
- 交叉注意力桥接去噪特征（Query）与编码器输出（Key/Value）
- 全展平时空注意力：$T_{\rm out}N \times T_{\rm out}N$ 规模，通过 FlashAttention 实现

### 扩散过程
- 正向：$T=200$ 步，线性/余弦噪声调度（$\beta \in [10^{-4}, 2\times10^{-2}]$）
- 逆向：$S=50$ 步 DDPM/DDIM 采样，多样本预测 $M_{\rm sample}=2$

### 模糊交通守恒损失
- 离散化守恒残差：$\mathbf{R}_{\rm cons} = \Delta\mathbf{S} - \gamma \cdot \mathbf{F}_{\rm net}$
- Sigmoid 拥堵感知加权：$\omega_{it} \in [0.5, 1.5]$，$\tau=8.0$，$\theta=0.6$
- 物理预热：$S_{\rm warm}=3000$ 步线性/余弦递增，$\rho_{\rm start}=0.2$

## 开发历史

1. 基于 `new_diffusion_2` 引入模糊数学结构（模糊图学习 + 模糊守恒损失）
2. 第一阶段稳训优化：物理预热机制、梯度检查点
3. 第二阶段推理降耗：DDIM 确定性采样、多样本并行、采样时间步缓存
4. 第三阶段容量缩放：hidden_dim=96, denoiser_layers=3, ffn_hidden_dim=192
5. 论文写作：方法论 200 行、实验 175 行，强调 Attention/Transformer 核心地位

## 输入/输出

- **输入**：扩散模型兼容的时空批数据与训练参数
- **输出**：预测结果、损失（扩散 MSE + 物理守恒）与训练中间状态
- **推理延迟**：DDPM 50 步 ~80-120ms（METR-LA），DDIM 50 步 ~75-110ms，多样本 ~150-220ms

## 调用关系

1. 可通过注册机制接入主流程
2. 可复用独立实验入口并指定当前目录配置
3. paper/src/sections_cn/methodology.tex 中完整描述架构

## 修改注意事项

1. 修改 `model.py` 需同步更新论文 method 描述
2. 配置参数变更需同步 `config.json` 与论文 Table 1 超参数表
3. 保持与 `new_diffusion_2` 的差异可追踪
4. 新增注意力形态需考虑显存和计算开销
