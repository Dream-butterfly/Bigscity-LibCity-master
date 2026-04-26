### 分析 10

时间：2026-04-22 13:05:00 +08:00
来源类型：提问
来源说明：用户确认“继续完成数据集处理模块解耦”的实施边界，并要求先落盘待办清单。

分析类型：方案设计/实施规划/待办拆解
分析范围：跨模块
状态：待执行

## 已确认决策（本轮约束）

1. 直接使用新链路，不考虑旧兼容。
2. 可全量切换到新链路。
3. `force_reuse` 对普通用户暴露。
4. 续训统一使用完整 `run_id`。
5. 本阶段不做旧方案兼容处理。
6. 超参搜索仅为想法，本阶段不实现。
7. 工件生命周期策略单独后续实现，本轮仅登记待办。

## 待办清单（按优先级）

### P0（本阶段必须完成）

1. Web 数据处理改为仅走 `scripts/run/run_data_artifact.py`。
2. Web 训练改为仅走 `scripts/run/run_train_artifact.py`。
3. Web 续训改为仅走 `scripts/run/run_resume_artifact.py`。
4. Web API 请求与状态字段统一为 `artifact_id`、`artifact_path`、`run_id`。
5. 续训前后端统一完整 `run_id`，移除任何截断逻辑。
6. 前端暴露 `force_reuse` 开关，并配套高风险提示文案。
7. 后端透传 `force_reuse` 到 artifact 脚本，并在日志保留 `[FORCE_REUSE]` 提示。

### P1（P0 完成后紧接实施）

1. 更新文档口径：`README.md`、`INFO.md`、`scripts/run/INFO.md`、`web/INFO.md` 改为“本阶段仅新链路”。
2. 更新 Web 页面文案：统一“数据工件（artifact）”术语，移除“数据版本（data_version）”歧义表述。
3. 增加最小冒烟验证记录模板，覆盖“构建工件 -> 训练 -> 续训”链路。

### P2（登记，不在本阶段实现）

1. 超参数搜索 Web 化（基于 `scripts/run/run_hyper_artifact.py`）。
2. 工件生命周期策略（保留期、清理规则、磁盘配额、告警阈值）。

## 验收标准（用于关闭 P0/P1）

1. Web 全链路运行时不再调用 `run_data_prep.py`、`run_model.py`、`run_resume.py`。
2. 训练/续训在签名不一致时默认阻断，开启 `force_reuse` 后可继续并给出风险日志。
3. 续训必须以完整 `run_id` 成功定位 `outputs/<run_id>/run_meta.json`。
4. 文档与页面术语一致，不再混用 `data_version_id` 与 `artifact_id`。

## 风险备注

1. 全量切换后，历史旧流程脚本不再作为 Web 路径，需避免误触发。
2. `force_reuse` 暴露给普通用户后，需确保默认关闭且风险提示足够明显。
3. `run_id` 口径切换期间需重点验证续训入口参数校验与错误提示可读性。

## 下一步建议

1. 先完成 `web/train_web_fastapi.py` 的入口脚本切换和参数字段收敛。
2. 再处理 `web/static/main.js` 与模板页面字段映射。
3. 最后统一更新文档并补一条变更日志。

