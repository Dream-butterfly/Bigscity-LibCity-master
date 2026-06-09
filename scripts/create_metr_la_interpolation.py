"""
METR_LA → METR_LA_INTERPOLATION 数据集生成脚本

插值策略（滑动平均）：
1. 0 值 → NaN（METR_LA 用 0 标记缺失）
2. 保留所有 34,272 时间步，不做丢弃
3. 每个 NaN 位置：取该传感器前后 ±288 步（24h）有效值的指数加权平均
   - 权重: exp(-α × |distance|)，α = 0.01
   - 近处值权重高，远处值作为日周期参考背景
4. 首尾 NaN：前向/后向填充
5. 窗口内无有效值 → 扩大窗口 → 全局传感器均值兜底

输出：resource_data/METR_LA_INTERPOLATION/
  - METR_LA_INTERPOLATION.dyna  （插值后数据）
  - METR_LA_INTERPOLATION.geo    （复制自 METR_LA）
  - METR_LA_INTERPOLATION.rel    （复制自 METR_LA）
  - config.json                  （更新 dataset 名称）
"""

import os
import shutil
import json
import time
import pandas as pd
import numpy as np

# ── 参数 ──────────────────────────────────────────────────────────────
WINDOW = 288       # ±288 步 = ±24 小时（5分钟间隔）
ALPHA = 0.01       # 指数衰减系数
EXPAND_WINDOW = 576  # 扩大窗口（±48 小时）

# ── 路径 ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "resource_data", "METR_LA")
DST_DIR = os.path.join(ROOT, "resource_data", "METR_LA_INTERPOLATION")
SRC_NAME = "METR_LA"
DST_NAME = "METR_LA_INTERPOLATION"


def fill_sensor_ewma(values, window=WINDOW, alpha=ALPHA, expand=EXPAND_WINDOW):
    """
    对单个传感器的时间序列做指数加权滑动平均填充。

    Args:
        values: 1D np.array，NaN 为缺失位置
        window: 滑动窗口半径（步）
        alpha: 指数衰减系数
        expand: 无有效值时的扩大窗口半径

    Returns:
        填充后的 1D np.array，无 NaN
    """
    v = values.copy().astype(np.float64)
    T = len(v)
    valid_mask = ~np.isnan(v)
    nan_idx = np.where(np.isnan(v))[0]

    if len(nan_idx) == 0:
        return v

    global_mean = np.mean(v[valid_mask]) if valid_mask.any() else 0.0

    for idx in nan_idx:
        # 尝试主窗口
        left = max(0, idx - window)
        right = min(T, idx + window + 1)
        wm = valid_mask[left:right]

        if not wm.any():
            # 扩大窗口
            left = max(0, idx - expand)
            right = min(T, idx + expand + 1)
            wm = valid_mask[left:right]

        if wm.any():
            wi = np.where(wm)[0] + left
            distances = np.abs(wi - idx).astype(np.float64)
            weights = np.exp(-alpha * distances)
            v[idx] = float(np.average(v[wi], weights=weights))
        else:
            v[idx] = global_mean

    # 防御：仍有 NaN 则前向后向填充
    still_nan = np.isnan(v)
    if still_nan.any():
        s = pd.Series(v)
        s = s.ffill().bfill()
        v = s.values

    return v


# ══════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════

# ── 1. 加载 ──────────────────────────────────────────────────────────
print(f"[1/6] 加载 {SRC_NAME}.dyna ...")
t0 = time.time()
df = pd.read_csv(os.path.join(SRC_DIR, f"{SRC_NAME}.dyna"))
n_sensors = df["entity_id"].nunique()
n_timesteps = df["time"].nunique()
print(f"  原始: {len(df):,} 行, {n_sensors} 传感器, {n_timesteps} 时间步")

sensor_order = list(df["entity_id"].unique())

# ── 2. 转矩阵 & 0→NaN ────────────────────────────────────────────────
print(f"[2/6] 转矩阵 & 标记缺失 (0 → NaN) ...")
pivot = df.pivot_table(
    values="traffic_speed", index="time", columns="entity_id",
    aggfunc="first"
)
pivot = pivot[sensor_order]
total_zeros = (pivot == 0).sum().sum()
print(f"  矩阵: {pivot.shape[0]} 时间步 × {pivot.shape[1]} 传感器")
print(f"  零值总数: {total_zeros:,} ({total_zeros / pivot.size * 100:.2f}%)")

pivot = pivot.replace(0.0, np.nan)

# ── 3. 传感器级 EWMA 滑动平均 ────────────────────────────────────────
print(f"[3/6] 传感器级指数加权滑动平均 (W={WINDOW}, α={ALPHA}) ...")
total_nan_before = pivot.isna().sum().sum()
print(f"  待填充 NaN: {total_nan_before:,}")

filled_cols = []
for i, col in enumerate(pivot.columns):
    vals = pivot[col].values
    filled = fill_sensor_ewma(vals, window=WINDOW, alpha=ALPHA)
    filled_cols.append(filled)
    if (i + 1) % 50 == 0:
        remaining = total_nan_before - sum(np.isnan(f).sum() for f in filled_cols)
        print(f"  进度: {i+1}/{len(pivot.columns)} 传感器, 已处理 NaN ~{remaining:,}")

# 组装回 DataFrame
filled_array = np.column_stack(filled_cols)
pivot_filled = pd.DataFrame(filled_array, index=pivot.index, columns=pivot.columns)

total_nan_after = pivot_filled.isna().sum().sum()
new_zeros = (pivot_filled == 0).sum().sum()
print(f"  填充后 NaN: {total_nan_after}")
print(f"  填充后零值: {new_zeros}")
print(f"  耗时: {time.time() - t0:.1f}s")

# ── 4. 统计验证 ──────────────────────────────────────────────────────
print(f"[4/6] 统计验证 ...")
orig_valid = df[df["traffic_speed"] > 0]["traffic_speed"]
new_all = pivot_filled.values.flatten()
new_all = new_all[~np.isnan(new_all)]

print(f"  均值: {orig_valid.mean():.2f} → {new_all.mean():.2f}")
print(f"  标准差: {orig_valid.std():.2f} → {new_all.std():.2f}")
print(f"  min: {orig_valid.min():.4f} → {new_all.min():.4f}")
print(f"  max: {orig_valid.max():.2f} → {new_all.max():.2f}")

# ── 5. 写回 .dyna ────────────────────────────────────────────────────
print(f"[5/6] 写回 {DST_NAME}.dyna ...")

os.makedirs(DST_DIR, exist_ok=True)

pivot_melted = pivot_filled.reset_index().melt(
    id_vars="time", var_name="entity_id", value_name="traffic_speed"
)

sensor_rank = {sid: i for i, sid in enumerate(sensor_order)}
time_rank = {t: i for i, t in enumerate(pivot_filled.index)}
pivot_melted["_sr"] = pivot_melted["entity_id"].map(sensor_rank)
pivot_melted["_tr"] = pivot_melted["time"].map(time_rank)
pivot_melted = pivot_melted.sort_values(["_sr", "_tr"]).reset_index(drop=True)
pivot_melted = pivot_melted.drop(columns=["_sr", "_tr"])

pivot_melted.insert(0, "dyna_id", range(len(pivot_melted)))
pivot_melted.insert(1, "type", "state")
pivot_melted = pivot_melted[["dyna_id", "type", "time", "entity_id", "traffic_speed"]]

dyna_path = os.path.join(DST_DIR, f"{DST_NAME}.dyna")
pivot_melted.to_csv(dyna_path, index=False)
print(f"  写入: {dyna_path}")
print(f"  新数据: {len(pivot_melted):,} 行, {pivot_melted['entity_id'].nunique()} 传感器, {pivot_melted['time'].nunique()} 时间步")

# ── 6. 复制 .geo / .rel 和 config ────────────────────────────────────
print(f"\n[6/6] 复制辅助文件 & 生成 config ...")
for ext in ["geo", "rel"]:
    src = os.path.join(SRC_DIR, f"{SRC_NAME}.{ext}")
    dst = os.path.join(DST_DIR, f"{DST_NAME}.{ext}")
    shutil.copy2(src, dst)
    print(f"  {src} → {dst}")

config = {
    "geo": {"including_types": ["Point"], "Point": {}},
    "rel": {"including_types": ["geo"], "geo": {"cost": "num"}},
    "dyna": {"including_types": ["state"], "state": {"entity_id": "geo_id", "traffic_speed": "num"}},
    "info": {
        "data_col": ["traffic_speed"],
        "weight_col": "cost",
        "data_files": [DST_NAME],
        "geo_file": DST_NAME,
        "rel_file": DST_NAME,
        "output_dim": 1,
        "time_intervals": 300,
        "init_weight_inf_or_zero": "inf",
        "set_weight_link_or_dist": "dist",
        "calculate_weight_adj": True,
        "weight_adj_epsilon": 0.1,
    },
}
config_path = os.path.join(DST_DIR, "config.json")
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
print(f"  写入: {config_path}")

# ── 最终统计 ──────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"METR_LA_INTERPOLATION 生成完成")
print(f"{'='*60}")
print(f"  策略:       指数加权滑动平均 (W={WINDOW}, α={ALPHA})")
print(f"  时间步:     全保留 {n_timesteps}（无丢弃）")
print(f"  传感器数:   {n_sensors}")
print(f"  填充前 NaN: {total_nan_before:,}")
print(f"  填充后 NaN: {total_nan_after}")
print(f"  总数据点:   {len(pivot_melted):,}")
print(f"  总耗时:     {time.time() - t0:.1f}s")
print(f"  输出路径:   {DST_DIR}")
