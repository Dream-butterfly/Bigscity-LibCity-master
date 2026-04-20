# test/INFO.md

## 目录职责

`test/` 存放测试与基线验证代码，当前以传统时间序列基线测试和主框架接口冒烟测试为主。

## 关键内容

| 文件 | 说明 |
| --- | --- |
| `test_ARIMA.py` | ARIMA 基线测试 |
| `test_HA.py` | HA（Historical Average）基线测试 |
| `test_SVR.py` | SVR 基线测试 |
| `test_VAR.py` | VAR 基线测试 |
| `test_model_api.py` | 主框架模型 API 冒烟测试 |
| `baseline_utils.py` | 基线测试公共工具 |
| `readme.md` | 测试相关说明 |

## 输入/输出

- **输入**：数据集与测试配置。  
- **输出**：基线模型预测结果与指标（通常输出到终端或项目输出目录）。

## 调用关系

- 该目录用于验证和对比，不直接驱动主训练流水线。  
- 与 `libcity.data` / `libcity.utils` 存在工具级复用关系。

## 修改注意事项

1. 新增基线测试时，优先复用 `baseline_utils.py`。  
2. 保持测试脚本命名风格一致（`test_*.py`）。  
3. 需要固定随机性时显式设置 seed，避免结果不可复现。  
4. 若测试依赖数据缓存，请在文档中注明前置步骤。

