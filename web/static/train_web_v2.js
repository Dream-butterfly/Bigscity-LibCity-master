/* ═══════════════════════════════════════════════════════
   GNNTP Web Console — v2 Main Script
   基于 main.js 迁移，保留核心功能，适配 sidebar 新布局
   v=20260508b — eval tab for checkpoint evaluation
   ═══════════════════════════════════════════════════════ */
console.log('[train_web_v2] loaded v=20260604b');

"use strict";

/* ═══════════════════════ GLOBALS ═══════════════════════ */
const DEFAULT_LANG = new URLSearchParams(window.location.search).get('lang')
  || window.localStorage.getItem('train_web_lang') || 'zh-CN';
const DEFAULT_THEME = new URLSearchParams(window.location.search).get('theme')
  || window.localStorage.getItem('train_web_theme') || 'dark';

const HIDDEN_TRAIN_PARAM_KEYS = new Set([
  'config_file', 'train_rate', 'eval_rate', 'dataset_class', 'task', 'model', 'dataset', 'seed',
]);
/* ── 参数说明（含用途、可选值、推荐值）── */
const PARAM_DESCRIPTIONS = {
  // ====== 训练控制 ======
  max_epoch:           '最大训练轮数。推荐: 100-200(小模型), 50-100(大模型)。可选: 任意正整数',
  batch_size:          '每批样本数。受GPU显存限制。推荐: 64(METR-LA), 32(PEMS大图)。可选: 8/16/32/64/128',
  eval_batch_size:     '评估时批大小，可大于训练batch。推荐: 与batch_size相同或更大。可选: 同batch_size',
  learning_rate:       '初始学习率。推荐: 0.001(Adam), 0.0005(AdamW)。可选: 1e-5 ~ 1e-2',
  base_lr:             '基础学习率(NEW_MODEL用)。推荐: 0.001。可选: 1e-5 ~ 1e-2',

  // ====== 优化器 ======
  learner:             '优化器类型。推荐: adamw(最佳泛化), adam(快速收敛)。可选: adam/adamw/sgd/adagrad/rmsprop/sparseadam',
  weight_decay:        '权重衰减(L2正则化)。推荐: 1e-4。可选: 1e-5 ~ 1e-2',
  clip_grad_norm:      '是否裁剪梯度范数，防止梯度爆炸。推荐: true。可选: true/false',
  max_grad_norm:       '梯度裁剪最大范数。推荐: 3。可选: 1/3/5/10',

  // ====== 学习率调度 ======
  lr_scheduler:        '学习率调度器类型。推荐: reducelronplateau(自适应), cosineannealinglr(SGD风格)。可选: multisteplr/steplr/exponentiallr/cosineannealinglr/lambdalr/reduceonplateau',
  lr_decay:            '是否启用学习率衰减。推荐: true。可选: true/false',
  lr_decay_ratio:      '学习率衰减系数(每次衰减乘以此值)。推荐: 0.5。可选: 0.1~0.9',
  lr_patience:         'ReduceLROnPlateau 耐心轮数。推荐: 5。可选: 3/5/10',
  lr_threshold:        'ReduceLROnPlateau 改善阈值。推荐: 0.001。可选: 1e-4 ~ 1e-2',
  lr_T_max:            'CosineAnnealing 周期长度(epochs)。推荐: =max_epoch。可选: 50/100/200',
  lr_eta_min:          'CosineAnnealing 最小学习率。推荐: 1e-5。可选: 1e-6 ~ 1e-4',
  lr_warmup_epoch:     '学习率预热轮数。推荐: 5。可选: 0~20',
  lr_warmup_init:      '预热初始学习率比例。推荐: 0.1。可选: 0.01~0.5',
  lr_epsilon:          'Adam epsilon，防除零。推荐: 1e-8。可选: 1e-9~1e-6',
  lr_beta1:            'Adam β1(一阶动量)。推荐: 0.9。可选: 0.5~0.99',
  lr_beta2:            'Adam β2(二阶动量)。推荐: 0.999。可选: 0.9~0.9999',
  lr_alpha:            'RMSProp alpha。推荐: 0.99。可选: 0.9~0.999',
  lr_momentum:         'SGD动量。推荐: 0.9。可选: 0.5~0.99',
  lr_lambda:           'LambdaLR衰减函数系数。推荐: 0.1。可选: 0.01~1.0',
  scale_lr:            '是否按batch_size缩放学习率。推荐: false。可选: true/false',
  steps:               'MultiStepLR 衰减节点(epoch列表)。推荐: [50,80]。可选: 逗号分隔epoch',
  step_size:           'StepLR 衰减步长(epoch)。推荐: 30。可选: 10/20/30/50',

  // ====== 早停 ======
  use_early_stop:      '是否启用早停。推荐: true(防止过拟合)。可选: true/false',
  patience:            '早停耐心轮数(连续N轮不改善则停)。推荐: 10(final_new), 30(diffusion)。可选: 5~50',

  // ====== 数据归一化 ======
  scaler:              '主数据归一化方法。推荐: standard(零均值单位方差)。可选: standard/normal/minmax01/minmax11/log/none',
  ext_scaler:          '外部特征归一化方法。推荐: none(不处理)或standard。可选: none/standard/normal/minmax01/minmax11/log',
  load_external:       '是否加载外部特征(天气/POI等)。推荐: false(纯交通数据)。可选: true/false',
  normal_external:     '是否对外部特征归一化。推荐: false(已在ext_scaler处理)。可选: true/false',

  // ====== 时间特征 ======
  add_time_in_day:     '是否添加日内时间编码(0-23时)。推荐: true(提升周期性建模)。可选: true/false',
  add_day_in_week:     '是否添加周内日编码(1-7)。推荐: true。可选: true/false',
  time_of_day:         '是否使用日内时间特征(STGformer风格)。推荐: true。可选: true/false',
  day_of_week:         '是否使用周内日特征(STGformer风格)。推荐: true。可选: true/false',
  steps_per_day:       '每日时间步数(如5分钟间隔=288)。推荐: 数据集自动推断。可选: 288/144/96',

  // ====== 模型维度 ======
  input_window:        '输入历史时间窗口(步数)。推荐: 12(1小时@5min)。可选: 6/12/24',
  output_window:       '预测未来时间窗口(步数)。推荐: 12(=input_window)。可选: 1/3/6/12',
  input_dim:           '输入特征维度(每节点特征数)。推荐: 1(速度)或3(速度+时间)。可选: 1~N',
  output_dim:          '输出维度。推荐: 1(单步速度)。可选: 1~N',
  hidden_dim:          '主隐藏维度。推荐: 96(final_new/final_2)。可选: 64/96/128/256',
  d_model:             'Transformer模型维度(NEW_MODEL/PDFormer用)。推荐: 64。可选: 32/64/128/256',
  ffn_hidden_dim:      '前馈网络隐藏维度(通常=hidden_dim×4)。推荐: 128(当hidden_dim=96)。可选: 128/256/512',
  embed_dim:           '嵌入维度(NEW_MODEL)。推荐: 64。可选: 32/64/128',
  external_dim:        '外部特征维度。推荐: 0(不使用外部特征)。可选: 0~N',
  skip_dim:            '跳连接维度(PDFormer)。推荐: 256。可选: 128/256/512',

  // ====== 注意力/Transformer ======
  num_heads:           '多头注意力头数。推荐: 2(hidden_dim=96时)。可选: 2/4/8(需整除hidden_dim)',
  num_layers:          '网络层数。推荐: 2-4。可选: 1/2/3/4/6',
  encoder_layers:      '编码器层数。推荐: 2。可选: 1/2/3/4',
  decoder_layers:      '解码器层数。推荐: 2。可选: 1/2/3/4',
  region_transformer_layers: '区域Transformer层数(final_new专用)。推荐: 1。可选: 1/2',
  dropout:             '通用Dropout率。推荐: 0.1。可选: 0.0/0.1/0.2/0.3/0.5',
  attn_drop:           '注意力Dropout率(PDFormer)。推荐: 0.1。可选: 0.0~0.3',
  drop_path:           'DropPath率(Stochastic Depth, PDFormer用)。推荐: 0.1。可选: 0.0~0.3',
  dropout_rate:        'Dropout率(STTN用)。推荐: 0.1。可选: 0.0~0.5',
  output_attention:    '是否输出注意力权重(调试用)。推荐: false。可选: true/false',
  use_mixed_proj:      '是否使用混合投影(STGformer)。推荐: true。可选: true/false',
  qkv_bias:            'QKV投影是否加偏置(PDFormer)。推荐: true。可选: true/false',

  // ====== 图结构 ======
  graph_k_hop:         '图卷积K跳邻域。推荐: 2。可选: 1/2/3',
  filter_type:         '图滤波类型(DCRNN)。推荐: dual_random_walk。可选: laplacian/random_walk/dual_random_walk',
  max_diffusion_step:  '最大扩散步数(DCRNN)。推荐: 2。可选: 1/2/3',
  cheb_order:          '切比雪夫多项式阶数(NEW_MODEL)。推荐: 3。可选: 1/2/3',
  bidir_adj_mx:        '是否双向邻接矩阵。推荐: true。可选: true/false',
  support_len:         '支持矩阵数量(NEW_MODEL)。推荐: 1。可选: 1/2/3',
  graph_conv_type:     '图卷积类型(STGCN)。推荐: chebconv。可选: gcnconv/chebconv/gatconv',

  // ====== 模糊集/区域(Fuzzy/Region) ======
  fuzzy_num_sets:      '每个节点的模糊集合数(隶属函数个数)。推荐: 4。可选: 2/3/4/6/8',
  num_cells:           '模糊区域单元数(空间聚类数)。推荐: 8。可选: 4/8/12/16',
  cell_blend_init:     '区域混合初始化值(0=硬划分,>0=软划分)。推荐: 0.0。可选: 0.0~0.5',
  band_center_init:    '模糊带中心初始化值(GMM均值缩放)。推荐: 1.1。可选: 0.5~2.0',
  band_width_init:     '模糊带宽度初始化值(GMM标准差缩放)。推荐: 0.7。可选: 0.3~1.5',
  fir_mode:            '模糊推理模式。推荐: lukasiewicz(Łukasiewicz逻辑)。可选: lukasiewicz/godel/product',
  use_cell_attention:  '是否使用区域注意力(new_fuzzy_cellattention)。推荐: true。可选: true/false',
  use_hollow_kernel:   '是否使用空心核(new_fuzzy_cellattention)。推荐: false。可选: true/false',

  // ====== 语义闭包(final_2/final_3_type2) ======
  use_semantic_closure:      '是否对模糊关系做语义闭包(传递闭包增强推理)。推荐: true。可选: true/false',
  semantic_closure_hops:     '语义闭包最大跳数。推荐: 3。可选: 1/2/3/5',

  // ====== 熵驱动动态图(final_2/final_3_type2) ======
  use_entropy_dynamic_graph: '是否用熵驱动动态图(区域不确定性调制)。推荐: true。可选: true/false',
  entropy_scale:             '熵驱动缩放系数(越大动态越强)。推荐: 0.1。可选: 0.01/0.05/0.1/0.2/0.5',
  use_entropy_fir_weight:    '是否用熵加权FIR(高熵区域降低推理权重)。推荐: true。可选: true/false',

  // ====== 模糊路由与稀疏化(final_2/final_3_type2) ======
  use_fuzzy_routing:           '是否启用模糊层级路由(区域→区域关系传递)。推荐: true。可选: true/false',
  use_fuzzy_sparsification:    '是否对模糊关系做稀疏化(去噪声边)。推荐: true。可选: true/false',
  sparsification_epsilon:      '稀疏化阈值(低于此值的关系边被裁剪)。推荐: 0.05。可选: 0.01/0.05/0.1/0.2',

  // ====== Type-2模糊(final_3_type2) ======
  use_type2_fuzzy:             '是否启用Type-2模糊(区间隶属度)。推荐: true(final_3_type2)。可选: true/false',
  type2_graph_mode:            'Type-2图模式。推荐: product。可选: product/min/max',
  type2_fou_gate_scale:        'Type-2 FOU门控缩放。推荐: 1.0。可选: 0.5~2.0',
  use_fou_entropy_modulation:  '是否用FOU熵调制。推荐: true。可选: true/false',
  fou_entropy_align_weight:    'FOU熵对齐权重。推荐: 0.1。可选: 0.01~1.0',

  // ====== 自适应图 ======
  use_adaptive_graph:          '是否使用自适应邻接矩阵(端到端学习)。推荐: true(图结构未知时)。可选: true/false',
  adaptive_graph_embed_dim:    '自适应图节点嵌入维度。推荐: 32。可选: 16/32/64',
  adaptive_graph_topk:         '自适应图稀疏化TopK。推荐: 12。可选: 5/10/12/20',
  adaptive_graph_blend_init:   '自适应图与静态图混合比例初始值。推荐: 0.5。可选: 0.0~1.0',

  // ====== 模糊图(new_fuzzy系列) ======
  use_fuzzy_graph:             '是否使用模糊图(高斯隶属函数构造边权重)。推荐: true(fuzzy系列)。可选: true/false',
  fuzzy_graph_num_sets:        '模糊图集合数。推荐: 3。可选: 2/3/4',
  fuzzy_graph_sigma_init:      '模糊图高斯sigma初始值。推荐: 0.7。可选: 0.3~1.5',

  // ====== 扩散模型 ======
  use_diffusion:               '是否使用扩散模型。推荐: true(diffusion系列)。可选: true/false',
  diffusion_steps:             '扩散过程总步数(训练时)。推荐: 200。可选: 50/100/200/500',
  diffusion_schedule:          '扩散噪声调度。推荐: linear。可选: linear/cosine',
  beta_start:                  '噪声调度起始beta。推荐: 1e-4。可选: 1e-5 ~ 1e-3',
  beta_end:                    '噪声调度终止beta。推荐: 0.01。可选: 0.005~0.05',
  num_sampling_steps:          '推理时采样步数(DDIM加速)。推荐: 50。可选: 10/25/50/200',
  num_prediction_samples:      '预测时采样次数(取均值)。推荐: 1。可选: 1/5/10',
  sampling_method:             '采样方法。推荐: ddim(加速推理)。可选: ddim/ddpm',
  ddim_eta:                    'DDIM噪声系数(0=确定性,1=随机)。推荐: 0.0。可选: 0.0/0.5/1.0',
  prediction_clamp_min:        '预测值下界裁剪(标准化后)。推荐: -3.0。可选: -5.0~0.0',
  prediction_clamp_max:        '预测值上界裁剪(标准化后)。推荐: 3.0。可选: 1.0~10.0',

  // ====== 时空注意力 ======
  use_spatiotemporal_attention: '是否使用时空分离注意力。推荐: true(diffusion_fuzzy系列)。可选: true/false',
  use_temporal_position_embedding: '是否使用时序位置编码。推荐: true。可选: true/false',

  // ====== 物理/守恒约束 ======
  conservation_loss_weight:    '守恒损失权重(控制物理约束强度)。推荐: 0.1。可选: 0.0~1.0',
  conservation_warmup_epochs:  '守恒损失预热轮数(前N轮逐渐增加权重)。推荐: 5。可选: 0~20',
  conservation_steps_per_epoch:'每轮守恒计算步数(随机采样次数)。推荐: 80。可选: 40/80/160',
  physics_channel_idx:         '物理约束作用的特征通道索引。推荐: 0(第0通道=速度)。可选: 0~N-1',
  physics_loss_weight:         '物理损失权重(diffusion系列)。推荐: 0.0(关闭)或0.01。可选: 0.0~1.0',
  physics_warmup_steps:        '物理损失预热步数(diffusion系列)。推荐: 3000。可选: 1000~10000',
  physics_warmup_start_ratio:  '物理损失预热起始比例。推荐: 0.2。可选: 0.0~0.5',
  physics_warmup_mode:         '物理预热调度模式。推荐: linear。可选: linear/cosine',
  flow_conservation_coeff:     '流守恒系数(diffusion系列)。推荐: 1.0。可选: 0.1~2.0',
  use_fuzzy_conservation:      '是否用模糊守恒(diffusion_fuzzy系列)。推荐: true。可选: true/false',
  fuzzy_conservation_threshold:'模糊守恒隶属度阈值。推荐: 0.6。可选: 0.3~0.9',
  fuzzy_conservation_temperature:'模糊守恒温度(soft程度)。推荐: 8.0。可选: 1.0~20.0',
  diffusion_conservation_enabled:'是否启用扩散守恒(new_fuzzy_3)。推荐: true。可选: true/false',
  fcm_loss_weight:             'FCM(模糊认知图)损失权重。推荐: 0.1。可选: 0.0~1.0',
  fcm_warmup_epochs:           'FCM预热轮次。推荐: 5。可选: 0~20',
  fcm_steps_per_epoch:         'FCM每轮计算步数。推荐: 80。可选: 40/80/160',

  // ====== 加速/内存 ======
  use_gradient_checkpointing: '是否用梯度检查点(省显存,稍慢)。推荐: false(显存够时), true(显存紧张时)。可选: true/false',
  use_amp:                    '是否用AMP混合精度训练(fp16加速)。推荐: true。可选: true/false',
  amp_dtype:                  'AMP数据类型。推荐: float16。可选: float16/bfloat16',
  num_workers:                'DataLoader工作线程数。推荐: 4。可选: 0/2/4/8',

  // ====== 损失函数 ======
  loss_fn:              '损失函数类型。推荐: masked_mae(交通预测标配)。可选: masked_mae/masked_mse/masked_rmse/masked_mape/masked_huber/log_cosh/r2/evar',
  train_loss:           '训练损失(可不同于评估损失)。推荐: none(=loss_fn)或masked_mse。可选: none/masked_mae/masked_mse/huber/log_cosh',
  huber_delta:          'Huber损失阈值。推荐: 1.0。可选: 0.5/1.0/2.0/5.0',
  set_loss:             '损失函数(PDFormer用)。推荐: masked_mae。可选: 同loss_fn',
  grad_accmu_steps:     '梯度累积步数(模拟大batch)。推荐: 1(不累积)。可选: 1/2/4/8',

  // ====== 课程学习 ======
  use_curriculum_learning: '是否用课程学习(逐步增加预测难度,DCRNN用)。推荐: true。可选: true/false',
  cl_decay_steps:          '课程学习衰减步数(DCRNN)。推荐: 2000。可选: 500~5000',
  task_level:              '课程学习起始级别(PDFormer)。推荐: 0。可选: 0/1/2',

  // ====== 评估 ======
  evaluator:            '评估器类名。推荐: TrafficStateEvaluator。可选: TrafficStateEvaluator',
  evaluator_mode:       '评估模式。推荐: single。可选: single/multi',
  metrics:              '评估指标列表。推荐: ["MAE","RMSE","MAPE"]。可选: MAE/MSE/RMSE/MAPE/WMAPE/R2/EVAR',

  // ====== 其他 ======
  seed:                 '随机种子(数据划分+模型初始化)。推荐: 42(主实验)或0/1/2(多跑)。可选: 任意整数',
  max_diffusion_step:   '最大扩散步数(图上的,非扩散模型)。推荐: 2。可选: 1/2/3',
  dataset_class:        '数据集加载类。推荐: TrafficStatePointDataset(点预测)。可选: TrafficStatePointDataset/TrafficStateDataset',
  device:               '计算设备。推荐: cuda(自动选择)。可选: cpu/cuda',
  saved_model:          '是否保存模型checkpoint。推荐: true。可选: true/false',
  train:                '是否执行训练(可设为false仅评估)。推荐: true。可选: true/false',
  log_level:            '日志级别。推荐: INFO。可选: DEBUG/INFO/WARNING/ERROR',
  log_every:            '日志打印间隔(epoch)。推荐: 1。可选: 1/5/10',
  load_best_epoch:      '是否加载验证集最佳epoch。推荐: true。可选: true/false',
  hyper_tune:           '是否超参调优模式。推荐: false。可选: true/false',
  is_distributed:       '是否分布式训练(DDP)。推荐: false(单机)。可选: true/false',
  save_mode:            '模型保存格式。推荐: best。可选: best/all',
  cache_dataset:        '是否缓存预处理数据。推荐: true(加速后续训练)。可选: true/false',
  pad_with_last_sample: '是否用最后样本填充不足batch。推荐: true。可选: true/false',
  robustness_test:      '是否鲁棒性测试。推荐: false。可选: true/false',
};

function getParamDescription(key) {
  return PARAM_DESCRIPTIONS[key] || '';
}

/* ── 参数候选值（有候选项的参数渲染为下拉菜单，无需手动输入）── */
const PARAM_CANDIDATES = {
  // 归一化
  scaler:      ['standard','normal','minmax01','minmax11','log','none'],
  ext_scaler:  ['none','standard','normal','minmax01','minmax11','log'],
  // 优化器 & 调度器
  learner:     ['adam','adamw','sgd','adagrad','rmsprop','sparseadam'],
  lr_scheduler: ['multisteplr','steplr','exponentiallr','cosineannealinglr','lambdalr','reduceonplateau'],
  // 损失函数
  loss_fn:     ['masked_mae','masked_mse','masked_rmse','masked_mape','masked_huber','log_cosh','r2','evar'],
  train_loss:  ['none','masked_mae','masked_mse','huber','log_cosh'],
  huber_delta: ['0.5','1.0','2.0','5.0'],
  // 扩散调度
  diffusion_schedule:   ['linear','cosine'],
  sampling_method:      ['ddim','ddpm'],
  physics_warmup_mode:  ['linear','cosine'],
  // bool 类
  saved_model:       ['true','false'],
  train:             ['true','false'],
  use_early_stop:    ['true','false'],
  lr_decay:          ['true','false'],
  clip_grad_norm:    ['true','false'],
  load_external:     ['true','false'],
  normal_external:   ['true','false'],
  time_of_day:       ['true','false'],
  day_of_week:       ['true','false'],
  add_time_in_day:   ['true','false'],
  add_day_in_week:   ['true','false'],
  use_mixed_proj:    ['true','false'],
  // 模糊 & 扩散 bool
  use_fuzzy_graph:               ['true','false'],
  use_fuzzy_conservation:        ['true','false'],
  use_adaptive_graph:            ['true','false'],
  use_spatiotemporal_attention:  ['true','false'],
  use_temporal_position_embedding: ['true','false'],
  use_gradient_checkpointing:    ['true','false'],
  use_amp:                       ['true','false'],
  scale_lr:                      ['true','false'],
  // 数据集
  dataset_class: ['TrafficStatePointDataset','TrafficStateDataset'],
};

const CLI_FIELDS = [
  'config_file', 'exp_id', 'seed', 'gpu', 'gpu_id',
  'train_rate', 'eval_rate', 'batch_size', 'learning_rate',
  'max_epoch', 'dataset_class', 'executor', 'evaluator',
  'num_gpus', 'gpu_ids',
];

/* ── Module-level state ── */
let _state = {
  paramRowsConfig: [],
  paramRowsExecutor: [],
  dataVersions: [],
  modelPlotType: 'pie',
  modelPlot: {},
  modelPlotOptionPie: {},
  modelPlotOptionBar: {},
  lossPlot: {},
  predictionRanges: {horizon: 0, node: 0, feature: 0},
  predictionSelection: {horizon: 1, node: 1, feature: 1},
};
let _resumeRuns = [];
let _evalRuns = [];
let _compareRuns = [];
let _charts = {};
let _trainLogLines = [];
let _resumeLogLines = [];
let _evalLogLines = [];
let trainLogIndex = 0;
let resumeLogIndex = 0;
let evalLogIndex = 0;

const TAB_MAP = {
  data:    { title: '数据处理',    sub: '数据工件生成 · 版本管理 · 数据集预览',      id: 'tab-data' },
  train:   { title: '训练',        sub: '参数配置 · 实时日志 · Loss/指标可视化',     id: 'tab-train' },
  resume:  { title: '继续训练',    sub: 'Checkpoint续训 · 历史运行恢复',             id: 'tab-resume' },
  eval:    { title: '模型评估',    sub: 'Checkpoint加载评估 · 指标/预测可视化',      id: 'tab-eval' },
  compare: { title: '模型对比',    sub: '多模型指标对比 · 预测曲线叠加',             id: 'tab-compare' },
  history: { title: '运行历史',    sub: '训练记录查询 · 指标追踪',                   id: 'tab-history' },
};

const TAB_I18N_KEYS = {
  data:    { title: 'tab_data_title',    sub: 'tab_data_sub' },
  train:   { title: 'tab_train_title',   sub: 'tab_train_sub' },
  resume:  { title: 'tab_resume_title',  sub: 'tab_resume_sub' },
  eval:    { title: 'tab_eval_title',    sub: 'tab_eval_sub' },
  compare: { title: 'tab_compare_title', sub: 'tab_compare_sub' },
  history: { title: 'tab_history_title', sub: 'tab_history_sub' },
};

/* ── i18n State ── */
let _i18nUI = {};
let _i18nParams = {};

function t(key, fallback) {
  return _i18nUI[key] || fallback || key;
}

function getParamDisplayName(key) {
  return _i18nParams[key] || key;
}

function applyI18n() {
  // data-i18n 元素
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    if (!key) return;
    if (!el.dataset.i18nFallback) el.dataset.i18nFallback = el.innerText;
    el.innerText = t(key, el.dataset.i18nFallback);
  });
  // data-i18n-placeholder 元素
  document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
    const key = el.getAttribute('data-i18n-placeholder');
    if (!key) return;
    if (!el.dataset.i18nPlaceholderFallback) {
      el.dataset.i18nPlaceholderFallback = el.getAttribute('placeholder') || '';
    }
    el.setAttribute('placeholder', t(key, el.dataset.i18nPlaceholderFallback));
  });
  // data-param-label 元素
  document.querySelectorAll('[data-param-label]').forEach(el => {
    const key = el.getAttribute('data-param-label');
    if (!key) return;
    el.innerText = getParamDisplayName(key);
  });
}

async function loadI18n(lang) {
  const normalized = lang.replace(/_/g, '-');
  window.localStorage.setItem('train_web_lang', normalized);
  try {
    const r = await fetch(`/static/i18n/${encodeURIComponent(normalized)}.json`);
    if (r.ok) {
      const data = await r.json();
      _i18nUI = data.ui || {};
      _i18nParams = data.params || {};
      document.documentElement.lang = data.lang || normalized;
    } else {
      _i18nUI = {}; _i18nParams = {};
    }
  } catch (_) {
    _i18nUI = {}; _i18nParams = {};
  }
  applyI18n();
  // 更新 lang 选择器
  const sel = document.getElementById('lang_select');
  if (sel) sel.value = normalized;
  // 更新 page header
  updatePageHeaderI18n();
}

function updatePageHeaderI18n() {
  const activeTab = document.querySelector('.sidebar-nav-item.active');
  if (!activeTab) return;
  const name = activeTab.dataset.tab;
  const keys = TAB_I18N_KEYS[name];
  if (!keys) return;
  const titleEl = document.getElementById('tabTitle');
  const subEl = document.getElementById('tabSub');
  if (titleEl && keys.title) {
    titleEl.textContent = t(keys.title, TAB_MAP[name].title);
  }
  if (subEl && keys.sub) {
    subEl.textContent = t(keys.sub, TAB_MAP[name].sub);
  }
}

/* ── ECharts 实例池 ── */
const CHART_IDS = ['chartModelParams','chartLoss','chartPred','chartEval','chartCompare','chartCompareMetrics'];

/* ═══════════════════════ TAB SWITCHING ═══════════════════════ */
function switchTab(name) {
  // 隐藏所有 tab
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  // 取消所有 sidebar 高亮
  document.querySelectorAll('.sidebar-nav-item').forEach(el => el.classList.remove('active'));
  // 激活目标 tab
  const tabEl = document.getElementById(TAB_MAP[name].id);
  if (tabEl) tabEl.classList.add('active');
  const btn = document.querySelector(`[data-tab="${name}"]`);
  if (btn) btn.classList.add('active');
  // 更新 header (优先用 i18n)
  const titleEl = document.getElementById('tabTitle');
  const subEl = document.querySelector('.page-header-left .sub');
  const i18nTitle = t(TAB_I18N_KEYS[name].title, TAB_MAP[name].title);
  const i18nSub = t(TAB_I18N_KEYS[name].sub, TAB_MAP[name].sub);
  if (titleEl) titleEl.textContent = i18nTitle;
  if (subEl) subEl.textContent = i18nSub;
  // 延迟 resize 图表
  setTimeout(resizeAllCharts, 300);
}

/* ═══════════════════════ THEME ═══════════════════════ */
function setTheme(mode) {
  document.documentElement.setAttribute('data-theme', mode);
  window.localStorage.setItem('train_web_theme', mode);
  const tweakSelect = document.getElementById('tweaksTheme');
  if (tweakSelect) tweakSelect.value = mode;
  setTimeout(resizeAllCharts, 100);
}

function toggleTheme() {
  const cur = document.documentElement.getAttribute('data-theme');
  setTheme(cur === 'dark' ? 'light' : 'dark');
}

/* ═══════════════════════ TWEAKS ═══════════════════════ */
function setupTweaks() {
  const toggle = document.getElementById('tweaksToggle');
  const panel = document.getElementById('tweaksPanel');
  if (toggle && panel) {
    toggle.addEventListener('click', () => panel.classList.toggle('visible'));
  }

  const themeSelect = document.getElementById('tweaksTheme');
  if (themeSelect) {
    themeSelect.addEventListener('change', e => setTheme(e.target.value));
  }

  // 间距调节
  document.getElementById('tweaksSpacing')?.addEventListener('change', e => {
    const scales = { compact: .75, default: 1, relaxed: 1.35 };
    const s = scales[e.target.value] || 1;
    const root = document.documentElement;
    root.style.setProperty('--spacing-xs',  `${4*s}px`);
    root.style.setProperty('--spacing-sm', `${8*s}px`);
    root.style.setProperty('--spacing-md', `${12*s}px`);
    root.style.setProperty('--spacing-lg', `${16*s}px`);
    root.style.setProperty('--spacing-xl', `${20*s}px`);
    root.style.setProperty('--spacing-2xl',`${24*s}px`);
    root.style.setProperty('--spacing-3xl',`${32*s}px`);
    root.style.setProperty('--spacing-4xl',`${40*s}px`);
  });

  // 圆角调节
  document.getElementById('tweaksRadius')?.addEventListener('change', e => {
    const radii = {
      sharp:   { sm:'2px', md:'4px', lg:'6px',  xl:'10px' },
      default: { sm:'6px', md:'10px',lg:'14px', xl:'20px' },
      round:   { sm:'10px',md:'16px',lg:'22px', xl:'28px' },
    };
    const r = radii[e.target.value] || radii.default;
    const root = document.documentElement;
    root.style.setProperty('--radius-sm', r.sm);
    root.style.setProperty('--radius-md', r.md);
    root.style.setProperty('--radius-lg', r.lg);
    root.style.setProperty('--radius-xl', r.xl);
  });

  // 密度调节
  document.getElementById('tweaksDensity')?.addEventListener('change', e => {
    const dense = e.target.value === 'dense';
    document.querySelectorAll('.card').forEach(c => c.style.padding = dense ? '10px 14px' : '');
    document.querySelectorAll('.group-grid').forEach(g => g.style.gap = dense ? '6px' : '');
  });
}

/* ═══════════════════════ ECHARTS ═══════════════════════ */
function getEChartsTheme() {
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  return {
    textStyle: { color: isDark ? '#9ba3b0' : '#4a4f5a' },
    legend: { textStyle: { color: isDark ? '#9ba3b0' : '#4a4f5a' } },
    tooltip: {
      backgroundColor: isDark ? 'rgba(14,18,26,.94)' : 'rgba(255,255,255,.94)',
      borderColor: isDark ? '#252b36' : '#e2e5ea',
      textStyle: { color: isDark ? '#e2e6ed' : '#1a1d23' },
    },
  };
}

function resizeAllCharts() {
  Object.values(_charts).forEach(chart => {
    if (chart) chart.resize();
  });
}
window.addEventListener('resize', resizeAllCharts);

/* ═══════════════════════ API HELPERS ═══════════════════════ */
async function apiGet(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`GET ${url} failed: ${res.status}`);
  return res.json();
}

async function apiPost(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `POST ${url} failed: ${res.status}`);
  }
  return res.json();
}

/* ═══════════════════════ METADATA LOAD ═══════════════════════ */
function buildTaskModelMap(modelsMeta) {
  const map = {};
  if (Array.isArray(modelsMeta)) {
    modelsMeta.forEach(item => {
      const task = String(item?.task || '').trim();
      const model = String(item?.model || '').trim();
      if (!task || !model) return;
      if (!map[task]) map[task] = [];
      if (!map[task].includes(model)) map[task].push(model);
    });
    return map;
  }
  if (modelsMeta && typeof modelsMeta === 'object') {
    Object.entries(modelsMeta).forEach(([task, models]) => {
      const taskName = String(task || '').trim();
      if (!taskName) return;
      const list = Array.isArray(models) ? models : [];
      map[taskName] = list
        .map(x => String(x || '').trim())
        .filter(Boolean);
    });
  }
  return map;
}

async function loadMeta() {
  try {
    const data = await apiGet('/api/meta');
    const taskModelMap = buildTaskModelMap(data.models);
    // 填充 task
    const taskSel = document.getElementById('data_task');
    if (taskSel) {
      const tasks = Object.keys(taskModelMap).sort();
      taskSel.innerHTML = tasks.map(t =>
        `<option value="${t}">${t}</option>`
      ).join('');
      const stgcnTask = Array.isArray(data.models)
        ? (data.models.find(x => String(x?.model || '') === 'STGCN')?.task || '')
        : '';
      if (stgcnTask && tasks.includes(stgcnTask)) {
        taskSel.value = stgcnTask;
      } else if (tasks.includes('traffic_state_pred')) {
        taskSel.value = 'traffic_state_pred';
      }
    }
    // 填充 model (依赖 task 变化)
    setupModelDatasetSelects(taskModelMap, data.datasets || []);
    // 填充数据版本
    if (data.data_versions) {
      const versions = Array.isArray(data.data_versions) ? data.data_versions : [];
      _state.dataVersions = versions;
      populateDataVersionSelects(versions);
      renderDataVersionTable(versions);
      populateDatasetFilter(versions);
    }
  } catch (e) {
    console.error('loadMeta failed:', e);
  }
}

function setupModelDatasetSelects(models, datasets) {
  const taskSel = document.getElementById('data_task');
  const modelSel = document.getElementById('data_model');
  const datasetSel = document.getElementById('data_dataset');
  if (!taskSel || !modelSel || !datasetSel) return;

  function updateModelDataset() {
    const task = taskSel.value;
    const modelList = models[task] || [];
    modelSel.innerHTML = modelList.map(m => `<option value="${m}">${m}</option>`).join('');
    if (modelList.includes('STGCN')) modelSel.value = 'STGCN';
    datasetSel.innerHTML = datasets.map(d => `<option value="${d}">${d}</option>`).join('');
    // 触发 populateTrainParams
    if (modelSel.value) populateTrainParams(modelSel.value);
  }
  taskSel.addEventListener('change', updateModelDataset);
  modelSel.addEventListener('change', () => populateTrainParams(modelSel.value));
  updateModelDataset();
}

function populateDataVersionSelects(versions) {
  const selectedResumeRunId = document.getElementById('resume_run_id')?.value || '';
  const selectedEvalRunId = document.getElementById('eval_run_id')?.value || '';
  const selectedResumeRun = (_resumeRuns || []).find(x => x.run_id === selectedResumeRunId);
  const selectedEvalRun = (_evalRuns || []).find(x => x.run_id === selectedEvalRunId);
  const ready = versions.filter(v => String(v.status || '').toLowerCase() === 'ready');
  const resumeReady = selectedResumeRun
    ? ready.filter(v => v.task === selectedResumeRun.task && v.model === selectedResumeRun.model && v.dataset === selectedResumeRun.dataset)
    : ready;
  const evalReady = selectedEvalRun
    ? ready.filter(v => v.task === selectedEvalRun.task && v.model === selectedEvalRun.model && v.dataset === selectedEvalRun.dataset)
    : ready;
  const renderSelect = (id, items) => {
    const sel = document.getElementById(id);
    if (!sel) return;
    const current = sel.value;
    sel.innerHTML = items.map(v =>
      `<option value="${escHtml(v.version_id)}">${escHtml(v.version_id)} (${escHtml(v.status || '-')} | ${escHtml(v.model || '-')}/${escHtml(v.dataset || '-')})</option>`
    ).join('');
    if (current && items.some(v => v.version_id === current)) {
      sel.value = current;
    } else if (items.length) {
      sel.value = items[0].version_id;
    }
  };
  renderSelect('train_data_version', ready);
  renderSelect('resume_data_version', resumeReady);
  renderSelect('eval_data_version', evalReady);
}

/* ═══════════════════════ PARAM TABLE UTILITIES ═══════════════════════ */
function inferType(v) {
  if (v === null || v === undefined) return 'str';
  if (typeof v === 'boolean') return 'bool';
  if (typeof v === 'number') return Number.isInteger(v) ? 'int' : 'float';
  if (typeof v === 'object') return 'json';
  return 'str';
}

function toInputString(v) {
  if (v === null || v === undefined) return '';
  if (typeof v === 'object') return JSON.stringify(v, null, 2);
  return String(v);
}

function parseValue(type, raw) {
  const t = (type || 'str').toLowerCase();
  if (t === 'bool') {
    const x = String(raw).trim().toLowerCase();
    if (['true', '1', 'yes'].includes(x)) return true;
    if (['false', '0', 'no'].includes(x)) return false;
    throw new Error(`布尔值错误: ${raw}`);
  }
  if (t === 'int') {
    const n = parseInt(String(raw).trim(), 10);
    if (Number.isNaN(n)) throw new Error(`整数错误: ${raw}`);
    return n;
  }
  if (t === 'float') {
    const n = Number(String(raw).trim());
    if (Number.isNaN(n)) throw new Error(`浮点数错误: ${raw}`);
    return n;
  }
  if (t === 'json') {
    const txt = String(raw).trim();
    return txt ? JSON.parse(txt) : null;
  }
  return String(raw);
}

function renderParamCell(section, idx) {
  if (idx === null) return '';
  const rows = section === 'executor' ? _state.paramRowsExecutor : _state.paramRowsConfig;
  const row = rows[idx];
  const candidates = PARAM_CANDIDATES[row.key];
  let valueEditor;
  if (candidates) {
    valueEditor = `<select class="param-value mono value-input" data-section="${section}" data-idx="${idx}">`
      + candidates.map(c => `<option value="${escHtml(c)}" ${row.value === c ? 'selected' : ''}>${escHtml(c)}</option>`).join('')
      + `</select>`;
  } else {
    valueEditor = `<input class="param-value mono value-input" data-section="${section}" data-idx="${idx}" value="${escHtml(row.value)}"/>`;
  }
  const desc = getParamDescription(row.key);
  return `
    <td class="param-localized">${escHtml(getParamDisplayName(row.key))}</td>
    <td class="mono param-key">${escHtml(row.key)}</td>
    <td>
      <select class="type-select" data-section="${section}" data-idx="${idx}">
        <option value="str" ${row.type === 'str' ? 'selected' : ''}>str</option>
        <option value="bool" ${row.type === 'bool' ? 'selected' : ''}>bool</option>
        <option value="int" ${row.type === 'int' ? 'selected' : ''}>int</option>
        <option value="float" ${row.type === 'float' ? 'selected' : ''}>float</option>
        <option value="json" ${row.type === 'json' ? 'selected' : ''}>json</option>
      </select>
    </td>
    <td>${valueEditor}</td>
    <td class="param-desc">${escHtml(desc)}</td>
  `;
}

function matchParamFilter(row, filter) {
  if (!filter) return true;
  const haystack = [
    getParamDisplayName(row.key),
    row.key,
    getParamDescription(row.key),
  ].join('\0').toLowerCase();
  return haystack.includes(filter);
}

function renderParamTable() {
  const filter = document.getElementById('param_filter')?.value.trim().toLowerCase() || '';
  const configIds = _state.paramRowsConfig
    .map((r, i) => ({r, i}))
    .filter(x => matchParamFilter(x.r, filter))
    .map(x => x.i);
  const executorIds = _state.paramRowsExecutor
    .map((r, i) => ({r, i}))
    .filter(x => matchParamFilter(x.r, filter))
    .map(x => x.i);
  const countEl = document.getElementById('param_count');
  if (countEl) countEl.textContent = String(configIds.length + executorIds.length);
  const c1 = document.getElementById('param_count_config');
  if (c1) c1.textContent = String(configIds.length);
  const c2 = document.getElementById('param_count_executor');
  if (c2) c2.textContent = String(executorIds.length);
  const bodyConfig = document.getElementById('param_tbody_config');
  const bodyExecutor = document.getElementById('param_tbody_executor');
  if (bodyConfig) {
    bodyConfig.innerHTML = configIds.map(idx => `<tr>${renderParamCell('config', idx)}</tr>`).join('')
      || `<tr><td colspan="5" class="small">${escHtml(t('table_no_params', '无参数'))}</td></tr>`;
  }
  if (bodyExecutor) {
    bodyExecutor.innerHTML = executorIds.map(idx => `<tr>${renderParamCell('executor', idx)}</tr>`).join('')
      || `<tr><td colspan="5" class="small">${escHtml(t('table_no_params', '无参数'))}</td></tr>`;
  }
  // Bind type select change
  document.querySelectorAll('#param_tbody_config .type-select, #param_tbody_executor .type-select').forEach(el => {
    el.addEventListener('change', e => {
      const section = e.target.getAttribute('data-section');
      const idx = Number(e.target.getAttribute('data-idx'));
      const rows = section === 'executor' ? _state.paramRowsExecutor : _state.paramRowsConfig;
      rows[idx].type = e.target.value;
      renderParamTable();
    });
  });
  // Bind value input change (input for text, change for select)
  document.querySelectorAll('#param_tbody_config .value-input, #param_tbody_executor .value-input').forEach(el => {
    const handler = e => {
      const section = e.target.getAttribute('data-section');
      const idx = Number(e.target.getAttribute('data-idx'));
      const rows = section === 'executor' ? _state.paramRowsExecutor : _state.paramRowsConfig;
      rows[idx].value = e.target.value;
    };
    el.addEventListener('input', handler);
    el.addEventListener('change', handler);
  });
}

function collectConfigFromTable() {
  const cfg = {};
  for (const r of _state.paramRowsConfig) cfg[r.key] = parseValue(r.type, r.value);
  for (const r of _state.paramRowsExecutor) cfg[r.key] = parseValue(r.type, r.value);
  return cfg;
}

function applyDefaultToCliFields(config) {
  const setIf = (k, v) => {
    if (v !== undefined && v !== null && document.getElementById(k)) document.getElementById(k).value = String(v);
  };
  setIf('max_epoch', config.max_epoch ?? 10);
  setIf('seed', config.seed);
  setIf('gpu_id', config.gpu_id);
  setIf('train_rate', config.train_rate);
  setIf('eval_rate', config.eval_rate);
  setIf('batch_size', config.batch_size);
  setIf('learning_rate', config.learning_rate);
  setIf('dataset_class', config.dataset_class);
  setIf('executor', config.executor);
  setIf('evaluator', config.evaluator);
  if (config.gpu !== undefined && document.getElementById('gpu')) {
    document.getElementById('gpu').value = String(config.gpu).toLowerCase();
  }
}

function collectCliOptions() {
  const out = {};
  for (const k of CLI_FIELDS) {
    const el = document.getElementById(k);
    if (!el) continue;
    const v = (el.value ?? '').trim();
    if (v !== '') out[k] = v;
  }
  return out;
}

function getTrainContextMeta() {
  const vid = document.getElementById('train_data_version')?.value || '';
  if (vid) {
    const v = (_state.dataVersions || []).find(x => x.version_id === vid);
    if (v && v.task && v.model && v.dataset) {
      return {task: String(v.task), model: String(v.model), dataset: String(v.dataset), version: v};
    }
  }
  const task = document.getElementById('data_task')?.value || '';
  const model = document.getElementById('data_model')?.value || '';
  const dataset = document.getElementById('data_dataset')?.value || '';
  if (task && model && dataset) return {task, model, dataset, version: null};
  return null;
}

/* ── GPU multi-select ── */
function onNumGpusChange() {
  const numGpus = parseInt(document.getElementById('num_gpus')?.value || '1', 10);
  const checkboxesDiv = document.getElementById('gpu_checkboxes');
  const display = document.getElementById('gpu_ids_display');
  if (!checkboxesDiv || !display) return;

  if (numGpus > 1) {
    checkboxesDiv.style.display = 'flex';
    // Auto-check first N GPUs
    const cbs = checkboxesDiv.querySelectorAll('input[type="checkbox"]');
    const currentChecked = new Set(getCheckedGpuIds());
    let need = numGpus - currentChecked.size;
    if (need > 0) {
      for (const cb of cbs) {
        if (need <= 0) break;
        if (!cb.checked) { cb.checked = true; need--; }
      }
    } else if (need < 0) {
      // Uncheck from the end
      for (let i = cbs.length - 1; i >= 0 && need < 0; i--) {
        if (cbs[i].checked) { cbs[i].checked = false; need++; }
      }
    }
  } else {
    checkboxesDiv.style.display = 'none';
  }
  updateGpuDisplay();
}

function onGpuCheckboxChange() {
  updateGpuDisplay();
}

function getCheckedGpuIds() {
  const checkboxes = document.querySelectorAll('#gpu_checkboxes input[type="checkbox"]:checked');
  return Array.from(checkboxes).map(cb => parseInt(cb.value, 10)).sort((a,b)=>a-b);
}

function updateGpuDisplay() {
  const display = document.getElementById('gpu_ids_display');
  if (!display) return;
  const ids = getCheckedGpuIds();
  display.textContent = ids.length > 0 ? `gpu_ids: [${ids.join(', ')}]` : 'gpu_ids: [0]';
}

function collectGpuOptions() {
  const numGpusEl = document.getElementById('num_gpus');
  const numGpus = parseInt(numGpusEl?.value || '1', 10);
  if (numGpus <= 1) return {num_gpus: '1', gpu_ids: ''};
  const ids = getCheckedGpuIds();
  return {num_gpus: String(numGpus), gpu_ids: ids.join(',')};
}

function applyTrainDataVersionSelection() {
  const vid = document.getElementById('train_data_version')?.value || '';
  const v = (_state.dataVersions || []).find(x => x.version_id === vid);
  const lockIds = ['seed', 'train_rate', 'eval_rate', 'dataset_class', 'config_file', 'batch_size'];
  lockIds.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.disabled = !!v;
  });
}

async function loadDefaults() {
  applyTrainDataVersionSelection();
  const ctx = getTrainContextMeta();
  if (!ctx) return;
  try {
    const data = await apiPost('/api/default_config', {
      task: ctx.task, model: ctx.model, dataset: ctx.dataset,
    });
    const cfg = data.config || {};
    cfg.max_epoch = 10;
    const executorKeys = new Set(data.executor_keys || []);
    const allRows = Object.keys(cfg).filter(k => !HIDDEN_TRAIN_PARAM_KEYS.has(k)).sort().map(k => ({
      key: k,
      defaultValue: cfg[k],
      type: inferType(cfg[k]),
      value: toInputString(cfg[k]),
    }));
    _state.paramRowsExecutor = allRows.filter(row => executorKeys.has(row.key));
    _state.paramRowsConfig = allRows.filter(row => !executorKeys.has(row.key));
    applyDefaultToCliFields(cfg);
    renderParamTable();
  } catch (e) {
    console.error('loadDefaults failed:', e);
  }
}

async function populateTrainParams(model) {
  await loadDefaults();
}

/* ═══════════════════════ DATA PREVIEW ═══════════════════════ */
async function loadDataPreview() {
  const fileType = document.getElementById('data_preview_file_type')?.value || 'dyna';
  const entityId = document.getElementById('data_preview_entity_id')?.value || '';
  const timeStart = document.getElementById('data_preview_time_start')?.value || '';
  const timeEnd = document.getElementById('data_preview_time_end')?.value || '';
  const columns = document.getElementById('data_preview_columns')?.value || '';
  const rows = document.getElementById('data_preview_rows')?.value || '20';
  const dataset = document.getElementById('data_dataset')?.value || '';

  const params = new URLSearchParams({ file_type: fileType, rows, dataset });
  if (entityId) params.set('entity_id', entityId);
  if (timeStart) params.set('time_start', timeStart);
  if (timeEnd) params.set('time_end', timeEnd);
  if (columns) params.set('columns', columns);

  try {
    const data = await apiGet(`/api/data/preview?${params}`);
    renderPreviewTable(data);
  } catch (e) {
    console.error('loadDataPreview failed:', e);
  }
}

function renderPreviewTable(data) {
  const meta = document.getElementById('data_preview_meta');
  const table = document.getElementById('data_preview_table');
  if (!table) return;
  if (meta) meta.textContent = data.meta || '';
  if (!data.data || data.data.length === 0) {
    table.innerHTML = '';
    return;
  }
  const cols = Object.keys(data.data[0]);
  table.innerHTML =
    `<thead><tr>${cols.map(c => `<th>${c}</th>`).join('')}</tr></thead>` +
    `<tbody>${data.data.map(row =>
      `<tr>${cols.map(c => `<td>${row[c] ?? ''}</td>`).join('')}</tr>`
    ).join('')}</tbody>`;
}

/* ═══════════════════════ DATA VERSIONS ═══════════════════════ */
async function loadDataVersions() {
  try {
    const data = await apiGet('/api/data/versions');
    const versions = Array.isArray(data.versions) ? data.versions : [];
    _state.dataVersions = versions;
    populateDataVersionSelects(versions);
    renderDataVersionTable(versions);
    populateDatasetFilter(versions);
  } catch (e) {
    console.error('loadDataVersions failed:', e);
  }
}

function renderDataVersionTable(versions) {
  const tbody = document.getElementById('data_version_tbody');
  const countEl = document.getElementById('dataVersionCount');
  if (countEl) countEl.textContent = `${versions.length} versions`;
  if (!tbody) return;
  const filterVal = (document.getElementById('data_versions_dataset_filter')?.value || '').trim();
  const filtered = filterVal ? versions.filter(v => (v.dataset || '') === filterVal) : versions;
  if (!filtered.some(v => v.version_id === _state.selectedDataVersionId)) {
    _state.selectedDataVersionId = filtered.length ? filtered[0].version_id : '';
  }
  tbody.innerHTML = filtered.map(v => {
    const status = (v.status || '').toLowerCase();
    const statusCls = status === 'ready' ? 'badge-finished' : status === 'processing' ? 'badge-processing' : 'badge-failed';
    const p = getVersionProfile(v);
    const splitText = `${fmtRate(p.trainRate)}/${fmtRate(p.evalRate)}/${fmtRate(p.testRate)}`;
    const selectedCls = v.version_id === _state.selectedDataVersionId ? 'is-selected' : '';
    return `<tr class="${selectedCls}" data-version-id="${escHtml(v.version_id || '')}">
      <td class="mono">${escHtml(v.version_id || '-')}</td>
      <td><span class="badge ${statusCls}">${v.status || '-'}</span></td>
      <td>${escHtml(v.task || '-')}</td>
      <td>${escHtml(v.model || '-')}</td>
      <td>${escHtml(v.dataset || '-')}</td>
      <td class="mono">${splitText}</td>
      <td>${escHtml(v.updated_at || '-')}</td>
    </tr>`;
  }).join('');
  tbody.querySelectorAll('tr[data-version-id]').forEach(row => {
    row.addEventListener('click', () => {
      _state.selectedDataVersionId = row.getAttribute('data-version-id') || '';
      renderDataVersionTable(_state.dataVersions || []);
    });
  });
}

function getVersionProfile(v) {
  const cli = v.cli_options || {};
  const cfg = v.config_payload || {};
  const tn = (n) => {
    const x = Number(n);
    return Number.isFinite(x) ? x : null;
  };
  const trainRate = tn(v.train_rate ?? cli.train_rate ?? cfg.train_rate);
  const evalRate = tn(v.eval_rate ?? cli.eval_rate ?? cfg.eval_rate);
  let testRate = tn(v.test_rate);
  if (testRate === null && trainRate !== null && evalRate !== null) testRate = 1 - trainRate - evalRate;
  return { trainRate, evalRate, testRate };
}

function fmtRate(v) { return Number.isFinite(v) ? Number(v).toFixed(3) : '-'; }

function populateDatasetFilter(versions) {
  const sel = document.getElementById('data_versions_dataset_filter');
  if (!sel) return;
  const prev = sel.value;
  const datasets = [...new Set(versions.map(v => v.dataset).filter(Boolean))];
  sel.innerHTML = '<option value="">全部数据集</option>' +
    datasets.map(d => `<option value="${escHtml(d)}">${escHtml(d)}</option>`).join('');
  if (prev && datasets.includes(prev)) sel.value = prev;
  sel.onchange = () => renderDataVersionTable(_state.dataVersions || []);
}

async function stopDataPrep() {
  try {
    await apiPost('/api/data/stop', {});
  } catch (e) {
    console.error('stopDataPrep failed:', e);
  }
}

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

/* ═══════════════════════ DATA PREP ═══════════════════════ */
async function startDataPrep() {
  const task = document.getElementById('data_task')?.value;
  const model = document.getElementById('data_model')?.value;
  const dataset = document.getElementById('data_dataset')?.value;
  const extraArgs = document.getElementById('data_extra_args')?.value || '';

  // 收集 CLI 字段
  const cliOptions = {};
  ['seed', 'batch_size', 'dataset_class', 'exp_id', 'gpu', 'gpu_id'].forEach(k => {
    const el = document.getElementById('data_' + k);
    if (el && el.value.trim()) cliOptions[k] = el.value.trim();
  });

  // 验证并收集 split 比例（test 自动计算）
  const t = parseFloat(document.getElementById('data_train_rate')?.value);
  const e = parseFloat(document.getElementById('data_eval_rate')?.value);
  if (!Number.isFinite(t) || !Number.isFinite(e)) {
    alert('请填写有效的训练/验证划分比例（必须为数字）');
    return;
  }
  if (t + e > 1) {
    alert(`train+eval 必须 ≤ 1（当前 ${(t+e).toFixed(4)}）`);
    return;
  }
  const test = Math.round((1 - t - e) * 1e6) / 1e6;
  document.getElementById('data_test_rate').value = String(test);
  cliOptions.train_rate = String(t);
  cliOptions.eval_rate = String(e);

  // 收集数据配置参数
  const config = collectDataConfig();

  try {
    dataLogPrevLen = 0;
    await apiPost('/api/data/start', {
      task, model, dataset,
      extra_args: extraArgs,
      cli_options: cliOptions,
      config: config,
    });
    pollDataLogs();
  } catch (e) {
    alert(`数据处理启动失败: ${e.message}`);
  }
}

function collectDataConfig() {
  const cfg = {};
  ['add_time_in_day', 'add_day_in_week', 'cache_dataset', 'load_external'].forEach(k => {
    const el = document.getElementById('data_' + k);
    if (el) cfg[k] = el.value === 'true';
  });
  const scalerEl = document.getElementById('data_scaler');
  if (scalerEl && scalerEl.value) cfg.scaler = scalerEl.value;
  ['input_window', 'output_window'].forEach(k => {
    const el = document.getElementById('data_' + k);
    if (el) { const v = parseFloat(el.value); if (Number.isFinite(v)) cfg[k] = v; }
  });
  return cfg;
}

/* ═══════════════════════ DATA LOG POLLING ═══════════════════════ */
let dataLogPrevLen = 0;

async function pollDataLogs() {
  const el = document.getElementById('data_logs');
  if (!el) return;
  try {
    const data = await apiGet(`/api/data/status?since=${dataLogPrevLen}`);
    const tail = Array.isArray(data.logs_tail) ? data.logs_tail : [];
    const logCount = data.log_count || 0;
    // 只追加新行（增量更新，用后端绝对行数追踪）
    if (tail.length > 0) {
      for (let i = 0; i < tail.length; i++) {
        appendDataLogLine(el, tail[i], dataLogPrevLen + i + 1);
      }
      dataLogPrevLen = logCount;
    }
    if (data.running) {
      setTimeout(pollDataLogs, 1000);
    }
  } catch (e) {
    setTimeout(pollDataLogs, 2000);
  }
}

function appendDataLogLine(container, rawLine, idx) {
  const text = String(rawLine ?? '');
  const levelMatch = text.match(/\b(INFO|WARNING|ERROR|DEBUG)\b/);
  const level = levelMatch ? levelMatch[1].toLowerCase() : 'info';
  const div = document.createElement('div');
  div.className = 'log-line';
  div.innerHTML =
    `<span class="log-index">${idx}</span>` +
    `<span class="log-level log-level-${level}">${level.toUpperCase()}</span>` +
    `<span class="log-message">${escapeHtml(text)}</span>`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

/* ═══════════════════════ LOG FILTERING ═══════════════════════ */
function trainLogLevel(line) {
  const text = typeof line === 'string' ? line : (line.message || '');
  const m = text.match(/\b(INFO|WARNING|ERROR|DEBUG)\b/);
  return m ? m[1].toLowerCase() : 'info';
}

function renderFilteredLogs(containerId, linesArray) {
  const container = document.getElementById(containerId);
  if (!container) return;
  const levelFilter = (document.getElementById('log_level_filter')?.value || 'all').toLowerCase();
  const keyword = (document.getElementById('log_keyword_filter')?.value || '').trim().toLowerCase();
  const filtered = (linesArray || []).filter(line => {
    const text = typeof line === 'string' ? line : (line.message || '');
    if (levelFilter !== 'all' && trainLogLevel(line) !== levelFilter) return false;
    if (keyword && !text.toLowerCase().includes(keyword)) return false;
    return true;
  });
  const html = filtered.map((line, i) => {
    const text = typeof line === 'string' ? line : (line.message || '');
    const lvl = trainLogLevel(line);
    return `<div class="log-line">
      <span class="log-index">${i + 1}</span>
      <span class="log-level log-level-${lvl}">${lvl.toUpperCase()}</span>
      <span class="log-message">${escapeHtml(text)}</span>
    </div>`;
  }).join('') || '<div class="log-empty" style="color:var(--text-muted);padding:1em">无匹配日志</div>';
  container.innerHTML = html;
  container.scrollTop = container.scrollHeight;
}

function applyLogFilter() {
  renderFilteredLogs('logs', _trainLogLines);
  renderFilteredLogs('resume_logs', _resumeLogLines);
}

function clearLogBuffer() {
  _trainLogLines = [];
  _resumeLogLines = [];
  _evalLogLines = [];
}

/* ═══════════════════════ TRAINING ═══════════════════════ */
async function startTrain() {
  const dataVersionId = document.getElementById('train_data_version')?.value;
  if (!dataVersionId) {
    alert('请先在训练页选择 ready 的 data_version_id');
    return;
  }
  const v = (_state.dataVersions || []).find(x => x.version_id === dataVersionId);
  if (!v || !v.task || !v.model || !v.dataset) {
    alert('所选 data_version 缺少 task/model/dataset 信息');
    return;
  }
  applyTrainDataVersionSelection();
  let config;
  try {
    config = collectConfigFromTable();
  } catch (e) {
    alert(`参数解析失败: ${e.message}`);
    return;
  }
  const cliOptions = collectCliOptions();
  const savedModel = document.getElementById('saved_model')?.value || 'true';
  const train = document.getElementById('train')?.value || 'true';
  const extraArgs = document.getElementById('extra_args')?.value || '';
  const gpuOpts = collectGpuOptions();

  try {
    await apiPost('/api/start', {
      task: v.task, model: v.model, dataset: v.dataset,
      data_version_id: dataVersionId,
      saved_model: savedModel, train,
      extra_args: extraArgs,
      cli_options: cliOptions,
      config: config,
      num_gpus: gpuOpts.num_gpus,
      gpu_ids: gpuOpts.gpu_ids,
    });
    clearLogBuffer();
    trainLogIndex = 0;
    pollTrainLogs();
  } catch (e) {
    alert(`启动失败: ${e.message}`);
  }
}

async function stopTrain() {
  try { await apiPost('/api/stop', {}); } catch (e) { console.error(e); }
}

async function clearState() {
  try { await apiPost('/api/clear', {}); } catch (e) { console.error(e); }
  clearLogBuffer();
  trainLogIndex = 0;
  resumeLogIndex = 0;
  dataLogPrevLen = 0;
  // 清空终端显示
  ['logs', 'resume_logs', 'data_logs'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.innerHTML = '';
  });
}

async function pollTrainLogs() {
  try {
    const data = await apiGet(`/api/status?since=${trainLogIndex}`);
    const rawLines = data.logs_tail || [];
    const logCount = data.log_count || 0;
    if (rawLines.length > 0) {
      _trainLogLines.push(...rawLines);
      trainLogIndex = logCount;  // 用后端绝对行数追踪，避免尾部截断导致的停滞
      renderFilteredLogs('logs', _trainLogLines);
    }
    // 更新 status badge
    const badge = document.getElementById('status');
    if (badge) {
      const s = data.running ? 'running' : (data.error ? 'error' : (data.return_code === 0 ? 'done' : 'idle'));
      badge.className = `badge badge-${s}`;
      badge.textContent = s;
    }
    // 更新模型参数量图
    if (data.model_plot || data.model_plot_option_pie || data.model_plot_option_bar) {
      _state.modelPlot = data.model_plot || _state.modelPlot;
      _state.modelPlotOptionPie = data.model_plot_option_pie || _state.modelPlotOptionPie;
      _state.modelPlotOptionBar = data.model_plot_option_bar || _state.modelPlotOptionBar;
      drawModelParamChart(_state.modelPlot, _state.modelPlotOptionPie, _state.modelPlotOptionBar);
    }
    // 更新 Loss 曲线
    if (data.loss_plot || data.loss_plot_option) {
      _state.lossPlot = data.loss_plot || _state.lossPlot;
      drawLossChart(data.loss_plot_option || {}, data.loss_plot || {});
    }
    // 结果就绪时获取指标和预测
    if (data.result_ready) {
      try {
        const result = await apiGet('/api/result');
        renderMetrics(result);
        const shape = Array.isArray(result?.shapes?.prediction) ? result.shapes.prediction : [];
        const ranges = result?.prediction_selector?.ranges || {};
        if (shape.length >= 4 || ranges.horizon) {
          applyPredictionSelectorMeta(result);
          refreshPredictionSeries();
        }
      } catch (_) {}
    }
    if (data.running) {
      setTimeout(pollTrainLogs, 1500);
    }
  } catch (e) {
    console.error('[pollTrainLogs] error:', e);
    setTimeout(pollTrainLogs, 2000);
  }
}

/* ═══════════════════════ RESUME TRAINING ═══════════════════════ */
function formatResumeRunTag(run) {
  const latest = Number(run?.latest_epoch);
  const latestText = Number.isFinite(latest)
    ? `latest=${latest}`
    : 'latest=-';
  return `${run.run_id} | ${run.model || '-'} | ${run.dataset || '-'} | ${latestText}`;
}

function renderResumeHint(run) {
  const el = document.getElementById('resume_hint');
  if (!el) return;
  if (!run) {
    el.textContent = t('resume_hint_empty', '请选择一个可续训的运行目录');
    return;
  }
  const epochs = Array.isArray(run.epochs) ? run.epochs : [];
  const latest = Number(run.latest_epoch);
  const minEpoch = epochs.length ? epochs[0] : '-';
  const maxEpoch = Number.isFinite(latest) ? latest : '-';
  el.textContent = `run=${run.run_id}, model=${run.model || '-'}, dataset=${run.dataset || '-'}, checkpoints=${minEpoch}..${maxEpoch}`;
}

function renderResumeEpochOptions(run) {
  const epochEl = document.getElementById('resume_epoch');
  if (!epochEl) return;
  const epochsRaw = Array.isArray(run?.epochs) ? run.epochs : [];
  const epochs = epochsRaw
    .map(x => Number(x))
    .filter(x => Number.isFinite(x) && x >= 0)
    .map(x => Math.floor(x))
    .sort((a, b) => a - b);
  if (!epochs.length) {
    epochEl.innerHTML = '<option value=""></option>';
    epochEl.value = '';
    epochEl.disabled = true;
    return;
  }
  let selected = Math.floor(Number(epochEl.value));
  if (!epochs.includes(selected)) {
    selected = Math.floor(Number(run?.latest_epoch));
  }
  if (!epochs.includes(selected)) {
    selected = epochs[epochs.length - 1];
  }
  epochEl.innerHTML = epochs
    .map(epoch => `<option value="${epoch}" ${epoch === selected ? 'selected' : ''}>${epoch}</option>`)
    .join('');
  epochEl.value = String(selected);
  epochEl.disabled = false;
}

function onResumeRunChanged() {
  const runId = document.getElementById('resume_run_id')?.value || '';
  const run = (_resumeRuns || []).find(x => x.run_id === runId);
  populateDataVersionSelects(_state.dataVersions || []);
  renderResumeHint(run || null);
  renderResumeEpochOptions(run || null);
  if (!run) return;
  const resumeEpoch = Math.floor(Number(document.getElementById('resume_epoch').value));
  if (!Number.isFinite(resumeEpoch) || resumeEpoch < 0) return;
  const currentMax = Math.max(1, Math.floor(Number(document.getElementById('resume_max_epoch').value || (resumeEpoch + 10))));
  document.getElementById('resume_max_epoch').value = String(Math.max(currentMax, resumeEpoch + 1));
}

async function loadResumeRuns() {
  try {
    const data = await apiGet('/api/resume_runs');
    _resumeRuns = Array.isArray(data.runs) ? data.runs : [];
    renderResumeRunOptions();
    populateDataVersionSelects(_state.dataVersions || []);
  } catch (e) {
    console.error('loadResumeRuns failed:', e);
  }
}

function renderResumeRunOptions() {
  const el = document.getElementById('resume_run_id');
  if (!el) return;
  const selected = el.value;
  const list = _resumeRuns || [];
  el.innerHTML = '<option value=""></option>' +
    list.map(run => {
      const sel = selected && selected === run.run_id ? 'selected' : '';
      return `<option value="${escHtml(run.run_id)}" ${sel}>${escHtml(formatResumeRunTag(run))}</option>`;
    }).join('');
  if (!el.value && list.length) el.value = list[0].run_id;
  onResumeRunChanged();
}

async function startResumeTrain() {
  const runId = (document.getElementById('resume_run_id').value || '').trim();
  if (!runId) {
    alert(t('resume_select_first', '请先选择可续训模型'));
    return;
  }
  const run = (_resumeRuns || []).find(x => x.run_id === runId);
  if (!run) {
    alert(t('resume_run_not_found', '未找到对应运行目录'));
    return;
  }
  if (!run.task || !run.model || !run.dataset) {
    alert(t('resume_run_meta_missing', '所选运行目录缺少 task/model/dataset 信息，无法继续训练'));
    return;
  }
  const dataVersionId = String(document.getElementById('resume_data_version')?.value || '').trim();
  if (!dataVersionId) {
    alert('请先选择可用的 data_version_id');
    return;
  }
  const resumeEpoch = Math.floor(Number(document.getElementById('resume_epoch').value));
  if (!Number.isFinite(resumeEpoch) || resumeEpoch < 0) {
    alert(t('resume_invalid_epoch', '请选择有效的续训 epoch'));
    return;
  }
  const availableEpochs = (Array.isArray(run.epochs) ? run.epochs : []).map(x => Math.floor(Number(x)));
  if (!availableEpochs.includes(resumeEpoch)) {
    alert(t('resume_invalid_epoch', '请选择有效的续训 epoch'));
    return;
  }
  const targetMaxEpoch = Math.max(1, Math.floor(Number(document.getElementById('resume_max_epoch').value || (resumeEpoch + 1))));
  if (targetMaxEpoch <= resumeEpoch) {
    alert(t('resume_invalid_max_epoch', '目标 max_epoch 必须大于续训起始 epoch'));
    return;
  }
  document.getElementById('resume_epoch').value = String(resumeEpoch);
  document.getElementById('resume_max_epoch').value = String(targetMaxEpoch);
  try {
    await apiPost('/api/start_resume', {
      task: run.task,
      model: run.model,
      dataset: run.dataset,
      data_version_id: dataVersionId,
      saved_model: document.getElementById('resume_saved_model').value,
      train: true,
      extra_args: document.getElementById('resume_extra_args').value || '',
      cli_options: {exp_id: run.run_id},
      config: {epoch: resumeEpoch, max_epoch: targetMaxEpoch},
    });
    // Start polling resume logs
    pollResumeLogs();
  } catch (e) {
    alert(`继续训练启动失败: ${e.message}`);
  }
}

async function pollResumeLogs() {
  try {
    const data = await apiGet(`/api/status?since=${resumeLogIndex}`);
    const rawLines = data.logs_tail || [];
    const logCount = data.log_count || 0;
    if (rawLines.length > 0) {
      _resumeLogLines.push(...rawLines);
      resumeLogIndex = logCount;
      renderFilteredLogs('resume_logs', _resumeLogLines);
    }
    const badge = document.getElementById('resume_status');
    if (badge) {
      const s = data.running ? 'running' : (data.error ? 'error' : (data.return_code === 0 ? 'done' : 'idle'));
      badge.className = `badge badge-${s}`;
      badge.textContent = s;
    }
    if (data.running) {
      setTimeout(pollResumeLogs, 1000);
    }
  } catch (e) {
    console.error('[pollResumeLogs] error:', e);
    setTimeout(pollResumeLogs, 2000);
  }
}

/* ═══════════════════════ EVAL FROM CHECKPOINT ═══════════════════════ */

function formatEvalRunTag(run) {
  const latest = Number(run?.latest_epoch);
  const latestText = Number.isFinite(latest) ? `latest=${latest}` : 'latest=-';
  return `${run.run_id} | ${run.model || '-'} | ${run.dataset || '-'} | ${latestText}`;
}

function renderEvalHint(run) {
  const el = document.getElementById('eval_hint');
  if (!el) return;
  if (!run) {
    el.textContent = t('eval_hint_empty', '请选择一个有 checkpoint 的运行目录');
    return;
  }
  const epochs = Array.isArray(run.epochs) ? run.epochs : [];
  const minEpoch = epochs.length ? epochs[0] : '-';
  const maxEpoch = epochs.length ? epochs[epochs.length - 1] : '-';
  el.textContent = `run=${run.run_id}, model=${run.model || '-'}, dataset=${run.dataset || '-'}, checkpoints=${minEpoch}..${maxEpoch} (${epochs.length} 个)`;
}

function renderEvalEpochOptions(run) {
  const epochEl = document.getElementById('eval_epoch');
  if (!epochEl) return;
  const epochsRaw = Array.isArray(run?.epochs) ? run.epochs : [];
  const epochs = epochsRaw
    .map(x => Number(x))
    .filter(x => Number.isFinite(x) && x >= 0)
    .map(x => Math.floor(x))
    .sort((a, b) => a - b);
  if (!epochs.length) {
    epochEl.innerHTML = '<option value=""></option>';
    epochEl.value = '';
    epochEl.disabled = true;
    return;
  }
  let selected = Math.floor(Number(epochEl.value));
  if (!epochs.includes(selected)) {
    selected = Math.floor(Number(run?.latest_epoch));
  }
  if (!epochs.includes(selected)) {
    selected = epochs[epochs.length - 1];
  }
  epochEl.innerHTML = epochs
    .map(epoch => `<option value="${epoch}" ${epoch === selected ? 'selected' : ''}>${epoch}</option>`)
    .join('');
  epochEl.value = String(selected);
  epochEl.disabled = false;
}

function populateEvalDataVersionOptions() {
  const runId = document.getElementById('eval_run_id')?.value || '';
  const run = (_evalRuns || []).find(x => x.run_id === runId);
  const versions = _state.dataVersions || [];
  const ready = versions.filter(v => String(v.status || '').toLowerCase() === 'ready');
  const filtered = run
    ? ready.filter(v => v.task === run.task && v.model === run.model && v.dataset === run.dataset)
    : ready;
  const sel = document.getElementById('eval_data_version');
  if (!sel) return;
  const current = sel.value;
  sel.innerHTML = filtered.map(v =>
    `<option value="${escHtml(v.version_id)}">${escHtml(v.version_id)} (${escHtml(v.status || '-')} | ${escHtml(v.model || '-')}/${escHtml(v.dataset || '-')})</option>`
  ).join('');
  if (current && filtered.some(v => v.version_id === current)) {
    sel.value = current;
  } else if (filtered.length) {
    sel.value = filtered[0].version_id;
  }
}

function onEvalRunChanged() {
  const runId = document.getElementById('eval_run_id')?.value || '';
  const run = (_evalRuns || []).find(x => x.run_id === runId);
  populateEvalDataVersionOptions();
  renderEvalHint(run || null);
  renderEvalEpochOptions(run || null);
}

async function loadEvalRuns() {
  try {
    const data = await apiGet('/api/eval/checkpoint_runs');
    _evalRuns = Array.isArray(data.runs) ? data.runs : [];
    renderEvalRunOptions();
  } catch (e) {
    console.error('loadEvalRuns failed:', e);
  }
}

function renderEvalRunOptions() {
  const el = document.getElementById('eval_run_id');
  if (!el) return;
  const selected = el.value;
  const list = _evalRuns || [];
  el.innerHTML = '<option value=""></option>' +
    list.map(run => {
      const sel = selected && selected === run.run_id ? 'selected' : '';
      return `<option value="${escHtml(run.run_id)}" ${sel}>${escHtml(formatEvalRunTag(run))}</option>`;
    }).join('');
  if (!el.value && list.length) el.value = list[0].run_id;
  onEvalRunChanged();
}

async function startEval() {
  const runId = (document.getElementById('eval_run_id')?.value || '').trim();
  if (!runId) {
    alert(t('eval_select_run_first', '请先选择可评估的运行目录'));
    return;
  }
  const run = (_evalRuns || []).find(x => x.run_id === runId);
  if (!run) {
    alert(t('eval_run_not_found', '未找到对应运行目录'));
    return;
  }
  const dataVersionId = String(document.getElementById('eval_data_version')?.value || '').trim();
  if (!dataVersionId) {
    alert(t('eval_select_data_version', '请先选择 data_version_id'));
    return;
  }
  const epoch = Math.floor(Number(document.getElementById('eval_epoch')?.value));
  if (!Number.isFinite(epoch) || epoch < 0) {
    alert(t('eval_invalid_epoch', '请选择有效的评估 epoch'));
    return;
  }
  const extraArgs = document.getElementById('eval_extra_args')?.value || '';

  try {
    await apiPost('/api/eval/start', {
      run_id: runId,
      epoch: epoch,
      data_version_id: dataVersionId,
      extra_args: extraArgs,
      config: {},
      cli_options: {},
    });
    // Clear previous eval results
    _evalLogLines = [];
    evalLogIndex = 0;
    document.getElementById('eval_logs').innerHTML = '';
    document.getElementById('eval_metrics').innerHTML = '';
    const chartEval = _charts.chartEval;
    if (chartEval) chartEval.clear();
    pollEvalLogs();
  } catch (e) {
    alert(`评估启动失败: ${e.message}`);
  }
}

async function stopEval() {
  try { await apiPost('/api/eval/stop', {}); } catch (e) { console.error(e); }
}

async function clearEvalState() {
  try { await apiPost('/api/eval/clear', {}); } catch (e) { console.error(e); }
  _evalLogLines = [];
  evalLogIndex = 0;
  const el = document.getElementById('eval_logs');
  if (el) el.innerHTML = '';
  document.getElementById('eval_metrics').innerHTML = '';
  const chartEval = _charts.chartEval;
  if (chartEval) chartEval.clear();
  // Reset prediction selectors
  ['eval_pred_horizon', 'eval_pred_node', 'eval_pred_feature'].forEach(id => {
    const sel = document.getElementById(id);
    if (sel) { sel.innerHTML = '<option value="1">1</option>'; sel.value = '1'; sel.disabled = true; }
  });
  const badge = document.getElementById('eval_status');
  if (badge) { badge.className = 'badge badge-idle'; badge.textContent = 'idle'; }
}

async function pollEvalLogs() {
  try {
    const data = await apiGet(`/api/eval/status?since=${evalLogIndex}`);
    const rawLines = data.logs_tail || [];
    const logCount = data.log_count || 0;
    if (rawLines.length > 0) {
      _evalLogLines.push(...rawLines);
      evalLogIndex = logCount;
      renderEvalLogs();
    }
    const badge = document.getElementById('eval_status');
    if (badge) {
      const s = data.running ? 'running' : (data.error ? 'error' : (data.return_code === 0 ? 'done' : 'idle'));
      badge.className = `badge badge-${s}`;
      badge.textContent = s;
    }
    if (data.result_ready) {
      try {
        const result = await apiGet('/api/eval/result');
        renderEvalMetrics(result);
        const shape = Array.isArray(result?.shapes?.prediction) ? result.shapes.prediction : [];
        const ranges = result?.prediction_selector?.ranges || {};
        if (shape.length >= 4 || ranges.horizon) {
          applyEvalPredictionSelectorMeta(result);
          refreshEvalPredictionSeries();
        }
      } catch (_) {}
    }
    if (data.running) {
      setTimeout(pollEvalLogs, 1500);
    }
  } catch (e) {
    console.error('[pollEvalLogs] error:', e);
    setTimeout(pollEvalLogs, 2000);
  }
}

function renderEvalLogs() {
  const container = document.getElementById('eval_logs');
  if (!container) return;
  const html = _evalLogLines.map((line, i) => {
    const text = typeof line === 'string' ? line : (line.message || '');
    const levelMatch = text.match(/\b(INFO|WARNING|ERROR|DEBUG)\b/);
    const lvl = levelMatch ? levelMatch[1].toLowerCase() : 'info';
    return `<div class="log-line">
      <span class="log-index">${i + 1}</span>
      <span class="log-level log-level-${lvl}">${lvl.toUpperCase()}</span>
      <span class="log-message">${escapeHtml(text)}</span>
    </div>`;
  }).join('');
  container.innerHTML = html;
  container.scrollTop = container.scrollHeight;
}

function renderEvalMetrics(result) {
  const el = document.getElementById('eval_metrics');
  if (el) {
    el.innerHTML = result.metrics_table_html || '<div class="small">无数据</div>';
  }
}

function applyEvalPredictionSelectorMeta(result) {
  const shape = Array.isArray(result?.shapes?.prediction) ? result.shapes.prediction : [];
  const ranges = result?.prediction_selector?.ranges || {};
  const selection = result?.prediction_selector?.selection || {};
  const toInt = v => { const n = Number(v); return Number.isFinite(n) ? Math.max(0, Math.floor(n)) : 0; };
  const horizonMax = toInt(ranges.horizon || shape[1] || 0);
  const nodeMax = toInt(ranges.node || shape[2] || 0);
  const featureMax = toInt(ranges.feature || shape[3] || 0);
  const setOpts = (id, max, sel) => {
    const el = document.getElementById(id);
    if (!el) return;
    const upper = Math.max(0, Math.floor(max));
    if (upper <= 0) { el.innerHTML = '<option value="1">1</option>'; el.value = '1'; el.disabled = true; return; }
    const safe = Math.min(Math.max(1, Number(sel) || 1), upper);
    const opts = [];
    for (let i = 1; i <= upper; i += 1) opts.push(`<option value="${i}" ${i === safe ? 'selected' : ''}>${i}</option>`);
    el.innerHTML = opts.join(''); el.value = String(safe); el.disabled = false;
  };
  setOpts('eval_pred_horizon', horizonMax, selection.horizon || 1);
  setOpts('eval_pred_node', nodeMax, selection.node || 1);
  setOpts('eval_pred_feature', featureMax, selection.feature || 1);
}

async function refreshEvalPredictionSeries() {
  const horizonEl = document.getElementById('eval_pred_horizon');
  const nodeEl = document.getElementById('eval_pred_node');
  const featureEl = document.getElementById('eval_pred_feature');
  if (!horizonEl || !nodeEl || !featureEl) return;
  const horizon = Number(horizonEl.value || 1);
  const node = Number(nodeEl.value || 1);
  const feature = Number(featureEl.value || 1);
  try {
    const qs = new URLSearchParams({horizon: String(horizon), node: String(node), feature: String(feature)});
    const data = await apiGet(`/api/eval/result_series?${qs.toString()}`);
    renderEChart('chartEval', data.chart_option || {});
  } catch (_) {}
}

/* ═══════════════════════ COMPARE ═══════════════════════ */
function renderCompareRunOptions() {
  const list = _compareRuns || [];
  const render = (id, selected) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.innerHTML = '<option value=""></option>' +
      list.map(r => {
        const tag = `${r.run_id} | ${r.model || '-'} | ${r.dataset || '-'}`;
        const sel = selected && selected === r.run_id ? 'selected' : '';
        return `<option value="${escHtml(r.run_id)}" ${sel}>${escHtml(tag)}</option>`;
      }).join('');
  };
  const a = document.getElementById('compare_run_a').value;
  const b = document.getElementById('compare_run_b').value;
  const c = document.getElementById('compare_run_c').value;
  render('compare_run_a', a);
  render('compare_run_b', b);
  render('compare_run_c', c);
  if (!document.getElementById('compare_run_a').value && list.length) {
    document.getElementById('compare_run_a').value = list[0].run_id;
  }
  if (!document.getElementById('compare_run_b').value && list.length > 1) {
    document.getElementById('compare_run_b').value = list[1].run_id;
  }
}

async function loadCompareRuns() {
  try {
    const data = await apiGet('/api/runs');
    _compareRuns = Array.isArray(data.runs) ? data.runs : [];
    renderCompareRunOptions();
  } catch (e) {
    console.error('loadCompareRuns failed:', e);
  }
}

/* ── 核心指标（默认只显示这些，避免图表过密）── */
const CORE_METRICS = ['masked_MAE', 'masked_RMSE', 'R2'];
const LOWER_IS_BETTER = new Set(['MAE','MSE','RMSE','MAPE','masked_MAE','masked_MSE','masked_RMSE','masked_MAPE']);
let _lastCompareItems = [];
let _showAllMetrics = false;

function toggleCompareMetrics() {
  _showAllMetrics = document.getElementById('compare_show_all_metrics')?.checked || false;
  renderCompareMetrics(_lastCompareItems);
}

function buildCompareChartOption(items, metricList, normalize) {
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  const textColor = isDark ? '#9ba3b0' : '#4a4f5a';
  const gridColor = isDark ? '#1e2430' : '#e8eaed';

  const categories = [];
  metricList.forEach(m => {
    categories.push(m + ' (h1)', m + ' (avg)');
  });

  // 如果归一化，计算每个 metric/dim 的 best 值
  const normRef = {};
  if (normalize) {
    metricList.forEach(m => {
      const vals = items.map(it => it.metrics_summary?.[m]).filter(Boolean);
      if (!vals.length) return;
      if (LOWER_IS_BETTER.has(m)) {
        normRef[m + '_h1'] = Math.min(...vals.map(v => Number(v.h1)));
        normRef[m + '_avg'] = Math.min(...vals.map(v => Number(v.avg)));
      } else {
        normRef[m + '_h1'] = Math.max(...vals.map(v => Number(v.h1)));
        normRef[m + '_avg'] = Math.max(...vals.map(v => Number(v.avg)));
      }
    });
  }

  const xAxisName = normalize ? '相对值（最佳=1.00）' : '原始值';

  const colors = ['#5470c6', '#91cc75', '#fac858', '#ee6666'];
  const series = items.map((it, idx) => {
    const data = [];
    metricList.forEach(m => {
      const x = it.metrics_summary?.[m];
      if (!x) { data.push(null, null); return; }
      let vH1 = Number(x.h1), vAvg = Number(x.avg);
      if (normalize) {
        const refH1 = normRef[m + '_h1'] || 1;
        const refAvg = normRef[m + '_avg'] || 1;
        if (LOWER_IS_BETTER.has(m)) {
          vH1 = refH1 / vH1;
          vAvg = refAvg / vAvg;
        } else {
          vH1 = vH1 / refH1;
          vAvg = vAvg / refAvg;
        }
      }
      data.push(vH1, vAvg);
    });
    const label = it.run_id || ('模型' + (idx + 1));
    const shortLabel = label.length > 28 ? '…' + label.slice(-24) : label;
    return { name: shortLabel, type: 'bar', data, color: colors[idx % colors.length],
      label: { show: true, position: 'right', fontSize: 10,
        formatter: p => p.value != null ? Number(p.value).toFixed(3) : ''
      }
    };
  });

  return {
    tooltip: {
      trigger: 'axis', axisPointer: { type: 'shadow' },
      backgroundColor: isDark ? 'rgba(14,18,26,.94)' : 'rgba(255,255,255,.94)',
      borderColor: gridColor,
      textStyle: { color: isDark ? '#e2e6ed' : '#1a1d23', fontSize: 12 },
      formatter: function(params) {
        let s = '<b>' + params[0].axisValue + '</b><br/>';
        params.forEach(p => {
          if (p.value != null) s += p.marker + ' ' + p.seriesName + ': <b>' + Number(p.value).toFixed(4) + '</b><br/>';
        });
        return s;
      }
    },
    legend: { data: series.map(s => s.name), bottom: 0, textStyle: { color: textColor, fontSize: 11 } },
    grid: { left: '3%', right: '8%', top: '3%', bottom: '12%', containLabel: true },
    xAxis: { type: 'value', name: xAxisName, nameTextStyle: { color: textColor, fontSize: 11 },
      axisLabel: { color: textColor, fontSize: 10 }, splitLine: { lineStyle: { color: gridColor } }
    },
    yAxis: { type: 'category', data: categories, axisLabel: { color: textColor, fontSize: 10 }, inverse: true },
    series,
  };
}

function renderCompareMetrics(items) {
  _lastCompareItems = items || [];
  const wrap = document.getElementById('compare_metrics_wrap');
  if (!wrap) return;
  if (!items || !items.length) {
    wrap.innerHTML = `<div class="small" style="padding:1.5em;text-align:center;color:var(--text-muted)">${escHtml(t('compare_no_data', '无可对比数据'))}</div>`;
    return;
  }
  const metrics = new Set();
  items.forEach(it => Object.keys(it.metrics_summary || {}).forEach(k => metrics.add(k)));
  let metricList = Array.from(metrics).sort();
  if (!_showAllMetrics) {
    metricList = metricList.filter(m => CORE_METRICS.includes(m));
    if (!metricList.length) metricList = Array.from(metrics).sort().slice(0, 3);
  }
  const lowerMetrics = metricList.filter(m => LOWER_IS_BETTER.has(m));
  const higherMetrics = metricList.filter(m => !LOWER_IS_BETTER.has(m));
  metricList = [...lowerMetrics, ...higherMetrics];
  const normalize = document.getElementById('compare_normalize')?.checked || false;

  // 截短 run_id
  const labels = items.map((it, i) => {
    const s = it.run_id || ('模型' + (i + 1));
    return s.length > 24 ? '…' + s.slice(-20) : s;
  });

  // 表头
  let html = '<table><thead><tr><th>指标</th>';
  labels.forEach(l => { html += `<th class="mono">${escHtml(l)}</th>`; });
  if (normalize) html += '<th>说明</th>';
  html += '</tr></thead><tbody>';

  // 每个指标生成两行：h1 和 avg
  metricList.forEach(m => {
    const vals = items.map(it => {
      const x = it.metrics_summary?.[m];
      return x ? { h1: Number(x.h1), avg: Number(x.avg) } : null;
    });
    // 找最优
    let bestH1, bestAvg;
    if (LOWER_IS_BETTER.has(m)) {
      bestH1 = Math.min(...vals.filter(Boolean).map(v => v.h1));
      bestAvg = Math.min(...vals.filter(Boolean).map(v => v.avg));
    } else {
      bestH1 = Math.max(...vals.filter(Boolean).map(v => v.h1));
      bestAvg = Math.max(...vals.filter(Boolean).map(v => v.avg));
    }

    const renderRow = (dim, dimLabel) => {
      html += '<tr>';
      html += `<td class="param-localized" style="padding-left:${dim === 'avg' ? '20px' : '8px'}">${dim === 'h1' ? escHtml(m) : ''} ${dimLabel}</td>`;
      vals.forEach((v, i) => {
        if (!v) { html += '<td>-</td>'; return; }
        const raw = v[dim];
        const isBest = Math.abs(raw - (dim === 'h1' ? bestH1 : bestAvg)) < 1e-8;
        let display;
        if (normalize) {
          const ref = dim === 'h1' ? bestH1 : bestAvg;
          const norm = LOWER_IS_BETTER.has(m) ? ref / raw : raw / ref;
          display = Number(norm).toFixed(3);
        } else {
          display = Number(raw).toFixed(4);
        }
        html += `<td class="mono${isBest ? ' compare-best' : ''}">${display}${isBest ? ' ✓' : ''}</td>`;
      });
      if (normalize) html += '<td class="small">最佳=1.00</td>';
      html += '</tr>';
    };
    renderRow('h1', '(h1)');
    renderRow('avg', '(avg)');
  });

  html += '</tbody></table>';
  wrap.innerHTML = html;
}

async function loadComparison() {
  const runIds = [
    document.getElementById('compare_run_a')?.value,
    document.getElementById('compare_run_b')?.value,
    document.getElementById('compare_run_c')?.value,
  ].filter(Boolean);
  if (!runIds.length) {
    document.getElementById('compare_metrics').innerHTML = `<tr><td>${escHtml(t('compare_select_model_first', '请先选择模型'))}</td></tr>`;
    return;
  }
  try {
    const data = await apiPost('/api/compare', {run_ids: runIds});
    renderCompareMetrics(data.items || []);
    renderEChart('chartCompare', data.chart_option || {});
  } catch (e) {
    alert(`对比失败: ${e.message}`);
  }
}

/* ═══════════════════════ HISTORY ═══════════════════════ */
function formatDuration(sec) {
  if (!Number.isFinite(Number(sec)) || Number(sec) < 0) return '-';
  const s = Math.floor(Number(sec));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  if (h > 0) return `${h}h ${m}m ${r}s`;
  if (m > 0) return `${m}m ${r}s`;
  return `${r}s`;
}

function formatTime(ts) {
  if (!Number.isFinite(Number(ts))) return '-';
  const d = new Date(Number(ts) * 1000);
  if (Number.isNaN(d.getTime())) return '-';
  return d.toLocaleString();
}

function renderHistoryTable(items) {
  const table = document.getElementById('history_table');
  if (!table) return;
  if (!Array.isArray(items) || !items.length) {
    table.innerHTML = `<tr><td>${escHtml(t('history_empty', '暂无历史记录'))}</td></tr>`;
    return;
  }
  let html = `<tr>
    <th>${escHtml(t('history_col_status', '状态'))}</th>
    <th>${escHtml(t('history_col_run', 'Run'))}</th>
    <th>${escHtml(t('history_col_model', '模型'))}</th>
    <th>${escHtml(t('history_col_dataset', '数据集'))}</th>
    <th>${escHtml(t('history_col_duration', '耗时'))}</th>
    <th>${escHtml(t('history_col_metrics', '主要指标'))}</th>
    <th>${escHtml(t('history_col_output', '输出目录'))}</th>
    <th>${escHtml(t('history_col_end_time', '结束时间'))}</th>
  </tr>`;
  for (const it of items) {
    const mm = it.major_metrics || {};
    const metricText = Object.keys(mm).map(k => `${k}=${Number(mm[k]).toFixed(4)}`).join(', ') || '-';
    html += `<tr>
      <td>${escHtml(t('history_col_' + (it.status || 'finished'), it.status || 'finished'))}</td>
      <td class="mono">${escHtml(it.run_id || '-')}</td>
      <td>${escHtml(it.model || '-')}</td>
      <td>${escHtml(it.dataset || '-')}</td>
      <td>${escHtml(formatDuration(it.duration_sec))}</td>
      <td>${escHtml(metricText)}</td>
      <td class="mono">${escHtml(it.output_dir || '-')}</td>
      <td>${escHtml(formatTime(it.ended_at))}</td>
    </tr>`;
  }
  table.innerHTML = html;
}

async function loadHistory() {
  const limitInput = document.getElementById('history_limit');
  const n = clampTopK(limitInput?.value, 20);
  if (limitInput) limitInput.value = String(n);
  try {
    const data = await apiGet(`/api/history?limit=${encodeURIComponent(n)}`);
    renderHistoryTable(Array.isArray(data.items) ? data.items : []);
  } catch (e) {
    console.error('loadHistory failed:', e);
  }
}

function clampTopK(v, d) {
  const n = Number(v);
  if (!Number.isFinite(n)) return d;
  return Math.max(1, Math.min(100, Math.floor(n)));
}

/* ═══════════════════════ SPLIT PRESET ═══════════════════════ */
function applySplitPreset() {
  const sel = document.getElementById('data_split_preset');
  if (!sel || !sel.value) return;
  const parts = sel.value.split(',').map(Number);
  const trainInput = document.getElementById('data_train_rate');
  const evalInput = document.getElementById('data_eval_rate');
  const testInput = document.getElementById('data_test_rate');
  if (trainInput) trainInput.value = parts[0];
  if (evalInput) evalInput.value = parts[1];
  if (testInput) testInput.value = String(Math.round((1 - parts[0] - parts[1]) * 1e6) / 1e6);
  updateSplitRatioHint();
}

function updateSplitRatioHint() {
  const hint = document.getElementById('data_split_ratio_hint');
  if (!hint) return;
  const t = Number(document.getElementById('data_train_rate')?.value || 0);
  const e = Number(document.getElementById('data_eval_rate')?.value || 0);
  const s = Math.round((1 - t - e) * 1e6) / 1e6;
  hint.textContent = `train/eval/test = ${t.toFixed(3)} / ${e.toFixed(3)} / ${s.toFixed(3)}`;
}

/* ═══════════════════════ DATA VERSION ACTIONS ═══════════════════════ */
function applySelectedDataVersionToTrain() {
  const sourceId = String(_state.selectedDataVersionId || document.getElementById('train_data_version')?.value || '').trim();
  const trainSel = document.getElementById('train_data_version');
  if (!sourceId || !trainSel) {
    alert('请先选择有效的 data_version_id');
    return;
  }
  const version = (_state.dataVersions || []).find(v => v.version_id === sourceId);
  if (!version) {
    alert('请先选择有效的 data_version_id');
    return;
  }
  if (String(version.status || '').toLowerCase() !== 'ready') {
    alert('仅 ready 数据版本可同步到训练参数');
    return;
  }
  trainSel.value = sourceId;
  applyTrainDataVersionSelection();
  loadDefaults();
}

/* ═══════════════════════ ECHARTS ═══════════════════════ */
function renderEChart(containerId, option) {
  const dom = document.getElementById(containerId);
  if (!dom || typeof echarts === 'undefined') return;
  let chart = _charts[containerId];
  if (!chart) {
    chart = echarts.init(dom);
    _charts[containerId] = chart;
  }
  // Revive function strings from JSON
  const reviveFn = (node) => {
    if (!node || typeof node !== 'object') return;
    if (Array.isArray(node)) { node.forEach(reviveFn); return; }
    for (const [k, v] of Object.entries(node)) {
      if ((k === 'formatter' || k === 'valueFormatter') && typeof v === 'string') {
        const s = v.trim();
        if (s.startsWith('function(') || s.startsWith('(function(')) {
          try { node[k] = new Function(`return (${s})`)(); } catch (_) {}
        }
      } else if (v && typeof v === 'object') {
        reviveFn(v);
      }
    }
  };
  const opt = option || {};
  reviveFn(opt);
  try {
    chart.setOption(opt, true);
  } catch (err) {
    console.warn(`renderEChart failed: ${containerId}`, err);
  }
}

function updateLossSummary(plot) {
  const xs = plot?.epochs || [];
  const maxEpoch = plot?.max_epoch;
  const maxText = maxEpoch ? ` / ${maxEpoch}` : '';
  const el = document.getElementById('loss_summary');
  if (el) el.textContent = xs.length ? `已记录 ${xs.length} 个 epoch${maxText}` : '暂无 loss 数据';
}

function drawLossChart(option, plot) {
  updateLossSummary(plot);
  renderEChart('chartLoss', option || {});
}

function drawModelParamChart(plot, pieOption, barOption) {
  const total = plot?.total_params;
  const summaryEl = document.getElementById('model_param_summary');
  if (summaryEl) {
    summaryEl.textContent = total
      ? `总参数量: ${total.toLocaleString()} | 展示: Top 参数分布`
      : '暂无参数分布数据';
  }
  const useBar = _state.modelPlotType === 'bar';
  let option = useBar ? (barOption || {}) : (pieOption || {});
  if (!useBar) {
    const hasSeries = Array.isArray(option?.series) && option.series.length > 0;
    if (!hasSeries) {
      const labels = Array.isArray(plot?.labels) ? plot.labels : [];
      const counts = Array.isArray(plot?.counts) ? plot.counts : [];
      const pairs = labels.map((name, i) => ({name, value: Number(counts[i] || 0)}))
        .filter(x => Number.isFinite(x.value) && x.value > 0)
        .sort((a, b) => b.value - a.value);
      const k = 8;
      const top = pairs.slice(0, k);
      const rest = pairs.slice(k).reduce((s, x) => s + x.value, 0);
      if (rest > 0) top.push({name: 'Others', value: rest});
      const fallbackData = top.length ? top : [{name: '暂无数据', value: 1}];
      option = {
        tooltip: {trigger: 'item', formatter: '{b}: {c} ({d}%)'},
        legend: {type: 'scroll', orient: 'vertical', left: '68%', top: '12%'},
        series: [{
          type: 'pie',
          radius: ['35%', '65%'],
          center: ['40%', '55%'],
          data: fallbackData,
          label: {formatter: '{b}: {d}%'},
        }],
      };
    }
  }
  renderEChart('chartModelParams', option);
}

function clearResultPanels() {
  const metricsEl = document.getElementById('metrics');
  if (metricsEl) metricsEl.innerHTML = `<div class="small">${escHtml('等待训练全部完成后展示')}</div>`;
  const chart = _charts.chartPred;
  if (chart) chart.clear();
  resetPredictionSelectors();
}

function resetPredictionSelectors() {
  const setOpts = (id, count, selected) => {
    const el = document.getElementById(id);
    if (!el) return;
    const upper = Number.isFinite(Number(count)) ? Math.max(0, Math.floor(Number(count))) : 0;
    if (upper <= 0) {
      el.innerHTML = '<option value="1">1</option>';
      el.value = '1';
      el.disabled = true;
      return;
    }
    const safe = Math.min(Math.max(1, Number(selected) || 1), upper);
    const opts = [];
    for (let i = 1; i <= upper; i += 1) opts.push(`<option value="${i}" ${i === safe ? 'selected' : ''}>${i}</option>`);
    el.innerHTML = opts.join('');
    el.value = String(safe);
    el.disabled = false;
  };
  setOpts('pred_horizon', 0, 1);
  setOpts('pred_node', 0, 1);
  setOpts('pred_feature', 0, 1);
  _state.predictionRanges = {horizon: 0, node: 0, feature: 0};
  _state.predictionSelection = {horizon: 1, node: 1, feature: 1};
}

async function refreshPredictionSeries() {
  if ((_state.predictionRanges.horizon || 0) <= 0 ||
      (_state.predictionRanges.node || 0) <= 0 ||
      (_state.predictionRanges.feature || 0) <= 0) return;
  const horizon = Number(document.getElementById('pred_horizon')?.value || 1);
  const node = Number(document.getElementById('pred_node')?.value || 1);
  const feature = Number(document.getElementById('pred_feature')?.value || 1);
  _state.predictionSelection = {horizon, node, feature};
  try {
    const qs = new URLSearchParams({horizon: String(horizon), node: String(node), feature: String(feature)});
    const data = await apiGet(`/api/result_series?${qs.toString()}`);
    renderEChart('chartPred', data.chart_option || {});
  } catch (_) {}
}

function applyPredictionSelectorMeta(result) {
  const shape = Array.isArray(result?.shapes?.prediction) ? result.shapes.prediction : [];
  const ranges = result?.prediction_selector?.ranges || {};
  const selection = result?.prediction_selector?.selection || {};
  const toInt = v => { const n = Number(v); return Number.isFinite(n) ? Math.max(0, Math.floor(n)) : 0; };
  const horizonMax = toInt(ranges.horizon || shape[1] || 0);
  const nodeMax = toInt(ranges.node || shape[2] || 0);
  const featureMax = toInt(ranges.feature || shape[3] || 0);
  const setOpts = (id, max, sel) => {
    const el = document.getElementById(id);
    if (!el) return;
    const upper = Math.max(0, Math.floor(max));
    if (upper <= 0) { el.innerHTML = '<option value="1">1</option>'; el.value = '1'; el.disabled = true; return; }
    const safe = Math.min(Math.max(1, Number(sel) || 1), upper);
    const opts = [];
    for (let i = 1; i <= upper; i += 1) opts.push(`<option value="${i}" ${i === safe ? 'selected' : ''}>${i}</option>`);
    el.innerHTML = opts.join(''); el.value = String(safe); el.disabled = false;
  };
  setOpts('pred_horizon', horizonMax, selection.horizon || _state.predictionSelection.horizon);
  setOpts('pred_node', nodeMax, selection.node || _state.predictionSelection.node);
  setOpts('pred_feature', featureMax, selection.feature || _state.predictionSelection.feature);
  _state.predictionRanges = {horizon: horizonMax, node: nodeMax, feature: featureMax};
  _state.predictionSelection = {
    horizon: Number(document.getElementById('pred_horizon')?.value || 1),
    node: Number(document.getElementById('pred_node')?.value || 1),
    feature: Number(document.getElementById('pred_feature')?.value || 1),
  };
}

function renderMetrics(result) {
  const metricsEl = document.getElementById('metrics');
  if (metricsEl) {
    metricsEl.innerHTML = result.metrics_table_html || '<div class="small">无数据</div>';
  }
}

function initCharts() {
  // Predictor controls binding
  document.getElementById('pred_horizon')?.addEventListener('change', refreshPredictionSeries);
  document.getElementById('pred_node')?.addEventListener('change', refreshPredictionSeries);
  document.getElementById('pred_feature')?.addEventListener('change', refreshPredictionSeries);

  // Eval predictor controls binding
  document.getElementById('eval_pred_horizon')?.addEventListener('change', refreshEvalPredictionSeries);
  document.getElementById('eval_pred_node')?.addEventListener('change', refreshEvalPredictionSeries);
  document.getElementById('eval_pred_feature')?.addEventListener('change', refreshEvalPredictionSeries);

  // Model plot type switch
  document.getElementById('model_plot_type')?.addEventListener('change', e => {
    _state.modelPlotType = e.target.value === 'bar' ? 'bar' : 'pie';
    drawModelParamChart(_state.modelPlot, _state.modelPlotOptionPie, _state.modelPlotOptionBar);
  });
}

/* ═══════════════════════ 页面刷新状态恢复 ═══════════════════════ */
async function restoreRunningState() {
  // 恢复训练状态
  try {
    const trainData = await apiGet('/api/status');
    if (trainData.running) {
      console.log('[restore] 检测到正在运行的训练，恢复日志和图表');
      trainLogIndex = 0;
      _trainLogLines = [];
      pollTrainLogs();
    } else if (trainData.return_code === 0 && trainData.log_count > 0) {
      // 训练已完成，直接渲染全部日志和结果
      console.log('[restore] 检测到已完成的训练，恢复日志');
      _trainLogLines = trainData.logs_tail || [];
      trainLogIndex = trainData.log_count || 0;
      renderFilteredLogs('logs', _trainLogLines);
      _state.modelPlot = trainData.model_plot || {};
      _state.modelPlotOptionPie = trainData.model_plot_option_pie || {};
      _state.modelPlotOptionBar = trainData.model_plot_option_bar || {};
      drawModelParamChart(_state.modelPlot, _state.modelPlotOptionPie, _state.modelPlotOptionBar);
      _state.lossPlot = trainData.loss_plot || {};
      drawLossChart(trainData.loss_plot_option || {}, trainData.loss_plot || {});
      const badge = document.getElementById('status');
      if (badge) { badge.className = 'badge badge-done'; badge.textContent = 'done'; }
      // 获取指标和预测
      if (trainData.result_ready) {
        try {
          const result = await apiGet('/api/result');
          renderMetrics(result);
          applyPredictionSelectorMeta(result);
          refreshPredictionSeries();
        } catch (_) {}
      }
    }
  } catch (e) {
    console.warn('[restore] 训练状态恢复失败:', e);
  }

  // 恢复评估状态
  try {
    const evalData = await apiGet('/api/eval/status');
    if (evalData.running) {
      console.log('[restore] 检测到正在运行的评估，恢复日志和图表');
      evalLogIndex = 0;
      _evalLogLines = [];
      pollEvalLogs();
    } else if (evalData.return_code === 0 && evalData.log_count > 0) {
      console.log('[restore] 检测到已完成的评估，恢复日志');
      _evalLogLines = evalData.logs_tail || [];
      evalLogIndex = evalData.log_count || 0;
      renderEvalLogs();
      if (evalData.result_ready) {
        try {
          const result = await apiGet('/api/eval/result');
          renderEvalMetrics(result);
          applyEvalPredictionSelectorMeta(result);
          refreshEvalPredictionSeries();
        } catch (_) {}
      }
    }
  } catch (e) {
    console.warn('[restore] 评估状态恢复失败:', e);
  }
}

/* ═══════════════════════ INIT ═══════════════════════ */
function init() {
  // Tab 切换绑定（含按需加载）
  document.querySelectorAll('.sidebar-nav-item').forEach(btn => {
    btn.addEventListener('click', async () => {
      const tab = btn.dataset.tab;
      switchTab(tab);
      if (tab === 'resume') {
        if (!_resumeRuns.length) await loadResumeRuns();
      } else if (tab === 'eval') {
        if (!_evalRuns.length) await loadEvalRuns();
      } else if (tab === 'compare') {
        if (!_compareRuns.length) await loadCompareRuns();
      } else if (tab === 'history') {
        await loadHistory();
      }
    });
  });

  // 主题切换
  document.getElementById('themeToggleSidebar')?.addEventListener('click', toggleTheme);
  setTheme(DEFAULT_THEME);

  // Tweaks
  setupTweaks();

  // Split 预设
  document.getElementById('data_split_preset')?.addEventListener('change', applySplitPreset);

  // 数据版本同步到训练
  document.getElementById('btnSyncToTrain')?.addEventListener('click', applySelectedDataVersionToTrain);

  // i18n 加载（自动调用 applyI18n）
  loadI18n(DEFAULT_LANG);

  // 语言切换
  const langSelect = document.getElementById('lang_select');
  if (langSelect) {
    langSelect.value = DEFAULT_LANG;
    langSelect.addEventListener('change', e => loadI18n(e.target.value));
  }

  // 参数表过滤
  document.getElementById('param_filter')?.addEventListener('input', renderParamTable);

  // 数据版本切换 → 加载默认参数
  document.getElementById('train_data_version')?.addEventListener('change', () => {
    applyTrainDataVersionSelection();
    loadDefaults();
  });

  // 继续训练 — run 切换
  document.getElementById('resume_run_id')?.addEventListener('change', onResumeRunChanged);

  // 模型评估 — run 切换
  document.getElementById('eval_run_id')?.addEventListener('change', onEvalRunChanged);

  // 日志过滤
  document.getElementById('log_level_filter')?.addEventListener('change', applyLogFilter);
  document.getElementById('log_keyword_filter')?.addEventListener('input', applyLogFilter);

  // 加载元数据
  loadMeta();
  loadDataVersions();

  // Charts（含绑定）
  initCharts();

  // 页面刷新后恢复正在运行的训练/评估状态
  restoreRunningState();
}

document.addEventListener('DOMContentLoaded', init);
