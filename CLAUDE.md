# GNNTP Project — Claude Auto-Read Entry

这是一个交通时空预测实验工程（GNNTP）。核心设计：**数据处理与模型训练解耦**（数据工件机制）。

## 快速入口

- **根目录** → `INFO.md`：全景架构、两条流水线、六个 Web 页面、模型黑盒约定
- **Web 控制台** → `web/INFO.md`：v2 前端、六页面功能、三状态机、stdout 实时解析
- **运行入口** → `scripts/run/INFO.md`：CLI 脚本、工件流水线与旧流水线、调用链路
- **数据处理** → `GNNTP/data/INFO.md`：数据工件机制、DataRuntime、签名校验
- **训练执行** → `GNNTP/common/INFO.md`：Executor/Evaluator、checkpoint 管理
- **模型目录** → `GNNTP/models/INFO.md`：四层模型结构、接入规范、黑盒约定
- **工具函数** → `GNNTP/utils/INFO.md`：路径/日志/seed/归一化/参数解析

## 关键路径速查

```
cache/data_artifacts/    ← 数据工件（npy + meta.json）
outputs/<exp_id>/        ← 实验产物（checkpoint + metrics + logs）
outputs/data_versions/   ← 数据版本元信息
outputs/web_train_history.json  ← 运行历史
```

## 常用命令

```bash
uv sync                          # 安装依赖
uv run run_web.py                # 启动 Web 控制台 (port 7817)
uv run scripts/run/run_data_artifact.py --model STGCN --dataset METR_LA
uv run scripts/run/run_train_artifact.py --model STGCN --dataset METR_LA --artifact_id <id>
uv run scripts/run/run_eval_checkpoint.py --run_id <id> --epoch 10 --artifact_id <id>
```

所有子目录的详细说明见各目录的 `INFO.md` 文件。
