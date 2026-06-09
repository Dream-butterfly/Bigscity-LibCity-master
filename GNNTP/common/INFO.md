# GNNTP/common/INFO.md

训练执行与评估层：负责训练循环、验证评估、模型保存恢复、调参流程和执行器/评估器注册。

## 关键文件

| 文件 | 字节 | 作用 |
|------|------|------|
| `abstract_executor.py` | 1.2K | 执行器抽象接口 |
| `abstract_evaluator.py` | 1.1K | 评估器抽象接口 |
| `traffic_state_executor.py` | 28.8K | 交通状态任务默认训练执行器（train、evaluate、save/load_model、checkpoint 管理） |
| `traffic_state_evaluator.py` | 11.2K | 交通状态任务默认评估器（MAE/RMSE/MAPE 等指标计算 + predictions npz 输出） |
| `registry_executor.py` / `registry_evaluator.py` | ~200B | 名称到实现类的注册与定位 |
| `hyper_tuning.py` | 8.8K | Optuna 调参流程封装（搜索、记录、结果保存） |
| `evaluator_utils.py` | 7.6K | 评估辅助函数 |
| `TrafficStateExecutor.json` | 711B | 执行器默认配置模板 |
| `TrafficStateEvaluator.json` | 204B | 评估器默认配置模板 |

## 执行器职责（TrafficStateExecutor）

1. `train(train_loader, valid_loader)` → 训练循环（forward → loss.backward → optimizer.step）
   - 周期性 evaluate validation set
   - 保存 checkpoint：`model_cache/<model>_<dataset>_epoch<N>.tar`
   - 输出 Loss 日志到 stdout（Web 后端正则解析）
2. `evaluate(test_loader)` → 在测试集上评估
   - 输出：`evaluate_cache/<metrics>.csv` + `<predictions>.npz`
3. `save_model(path)` / `load_model(path)` → 序列化/反序列化
4. `load_model_with_epoch(epoch)` → 从 checkpoint 恢复指定 epoch

## 评估器职责（TrafficStateEvaluator）

- 多 horizon 指标计算：MAE、RMSE、MAPE、masked_MAE 等
- 按 horizon 逐行输出 metrics CSV
- 输出 prediction vs truth 的 npz 文件（供前端预测曲线渲染）

## 输入/输出

- **输入**：配置对象、模型实例、DataLoader
- **输出**：训练日志 stdout、model_cache/*.tar checkpoint、evaluate_cache/*.csv + *.npz

## 修改注意事项

1. 新增执行器或评估器必须同步注册器与配置声明
2. 训练循环变更要同时考虑：续训（checkpoint 恢复）、早停、日志格式
3. 指标计算口径改动应同步更新 Web 展示与历史结果解读
4. 避免在此层硬编码单模型逻辑，模型特有行为优先下沉到模型目录
5. checkpoint 格式变更要考虑续训兼容性

