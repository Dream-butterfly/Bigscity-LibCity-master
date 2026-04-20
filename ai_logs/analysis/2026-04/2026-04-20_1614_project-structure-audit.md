### 分析 7

时间：2026-04-20 16:14:15 +08:00
来源类型：提问
来源说明：用户要求分析当前项目结构中冗余、缺失、结构异常和文件位置异常，并记录本次对话。

分析类型-目的：问题定位
分析类型-产出：事实结论
结论状态：可用

输入内容：
对根目录与递归子目录进行结构审计，重点检查运行入口、模型注册链路、Web 调用链路、文档引用一致性与实验目录归位情况。

分析结果：
- 缺失问题（高优先级）：`scripts/run/run_model.py`、`run_data_prep.py`、`run_hyper.py`、`run_resume.py` 的默认模型均为 `GRU`，但当前 `GNNTP/models/**/manifest.json` 中不存在 `GRU`（也不存在 `RNN`）模型声明，默认参数存在失配风险。
- 结构契约问题（高优先级）：Web 训练后台统一拼装 `--train` 参数；`/api/start_resume` 走 `run_resume.py` 时同样会带上 `--train`，但 `run_resume.py` CLI 未定义该参数，存在入口参数契约不一致。
- 文档缺失/错位（中优先级）：
  - `README.md` 引用了 `AI_EDIT_LOG.md`，仓库中不存在该文件；
  - 根目录 `INFO.md` 引用了 `readme_zh.md` 与 `AI_EDIT_LOG.md`，仓库中不存在；
  - `README.md` 声明了 `resource_data/INFO.md`、`cache/INFO.md`、`outputs/INFO.md`，当前均不存在。
- 文件位置异常（中优先级）：`GNNTP/models/traffic_speed_prediction/STGformer-独立实现/` 位于主模型树中，但无 `manifest.json` 接入主注册链路，且训练脚本使用独立相对路径（如 `../data`），更接近“独立实验项目”而非主模型模块。
- 冗余问题（中优先级）：`scripts/experiments/train_new_diffusion.py` 与 `train_new_diffusion_2.py` 结构高度重复，仅模型导入与默认输出名不同，存在可抽象合并空间。
- 冗余代码（低优先级）：`GNNTP/utils/argument_list.py` 中 `hyper_arguments` 与 `add_hyper_args` 当前未被调用。

关键结论：
当前工程主链路（manifest 驱动的模型加载、run 脚本入口、Web 启动）总体清晰，但存在“默认参数与已注册模型脱节”“Web 与 CLI 参数契约不一致”“文档引用与实际文件不一致”三类关键结构性问题，属于会影响可用性和维护稳定性的优先修复项。

启发，或是下一步可以完成的工作：
- 统一 run 入口默认模型为已注册模型（如 STGCN）。
- 修正 Web resume 调用与 run_resume CLI 的参数契约（去除或显式支持 `--train`）。
- 同步修正文档引用，补齐缺失 `INFO.md` 或删除失效链接。
- 将 `STGformer-独立实现` 明确归档到实验区，或补齐 manifest 并定义接入边界。
- 合并 diffusion 两个独立训练脚本的公共流程，减少重复维护成本。

补充说明：
- 风险点：默认模型失配和 resume 参数失配会直接影响“开箱即用”体验。
- 适用范围：适用于当前仓库主分支下的 CLI/Web 训练与模型注册体系。
- 限制条件：本次为静态结构审计，未执行完整训练流程回归。
