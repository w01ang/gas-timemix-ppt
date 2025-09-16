#!/usr/bin/env python3
"""
分析无平滑过渡的跳跃情况
"""

import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv('test_results/long_term_forecast_wellmix_dynamic_input_v2_dynamic_input_v2_TimeMixer_WELLS_sl3000_pl256_dm256_nh16_el6_dl3_df1024_fc1_ebtimeF_dtTrue_enhanced_0/one_well_true_pred.csv')

print("📊 无平滑过渡预测跳跃分析")
print("=" * 50)

# 计算跳跃
true_values = df['true'].values
pred_values = df['preds'].values

# 第一个预测值与最后一个真实值的跳跃
# 这里我们需要从输入段获取最后一个值
# 让我们查看输入段数据
input_df = pd.read_csv('test_results/long_term_forecast_wellmix_dynamic_input_v2_dynamic_input_v2_TimeMixer_WELLS_sl3000_pl256_dm256_nh16_el6_dl3_df1024_fc1_ebtimeF_dtTrue_enhanced_0/one_well_enhanced_3color.csv')

# 获取输入段的最后一个值
input_values = input_df['input_segment'].dropna().values
last_input_value = input_values[-1]
first_pred_value = pred_values[0]

jump = first_pred_value - last_input_value

print(f"输入段最后一个值: {last_input_value:.2f}")
print(f"预测段第一个值: {first_pred_value:.2f}")
print(f"跳跃大小: {jump:.2f}")
print(f"跳跃百分比: {(jump/last_input_value)*100:.2f}%")

# 计算预测精度
mae = np.mean(np.abs(pred_values - true_values))
rmse = np.sqrt(np.mean((pred_values - true_values) ** 2))
mape = np.mean(np.abs((pred_values - true_values) / (true_values + 1e-8))) * 100

print(f"\n预测精度指标:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.1f}%")

# 分析跳跃的分布
jumps = np.diff(pred_values)
print(f"\n预测值内部跳跃分析:")
print(f"平均内部跳跃: {np.mean(np.abs(jumps)):.2f}")
print(f"最大内部跳跃: {np.max(np.abs(jumps)):.2f}")
print(f"最小内部跳跃: {np.min(np.abs(jumps)):.2f}")

print(f"\n✅ 分析完成！当前模型已去除平滑过渡设计。")
