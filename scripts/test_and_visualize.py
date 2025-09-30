#!/usr/bin/env python3
"""
TimeMixer Well Lifecycle Testing and Visualization Script (No Smooth Transition)
去除平滑过渡设计的测试和可视化脚本
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
import json
from argparse import Namespace

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from data_provider.data_factory import data_provider
import torch

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_experiment_config(model_id):
    """加载实验配置"""
    config_path = f"experiments/{model_id}/config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 填充必要默认参数，兼容训练时保存的精简配置
    defaults = {
        'task_name': 'long_term_forecast',
        'is_training': 0,
        'model': 'TimeMixer',
        'data': 'WELLS',
        'root_path': '/Users/wangjr/Documents/yk/timemixer/timemixer-ppt/data',
        'data_path': 'preprocessed_daily_gas_by_well.csv',
        'features': 'S',
        'target': 'OT',
        'freq': 'd',
        'checkpoints': './checkpoints/',
        'embed': 'timeF',
        'dropout': 0.1,
        'seasonal_patterns': 'Monthly',
        'inverse': True,
        'top_k': 5,
        'num_kernels': 6,
        'enc_in': 1,
        'dec_in': 1,
        'c_out': 1,
        'channel_independence': 1,
        'moving_avg': 49,
        'decomp_method': 'moving_avg',
        'factor': 1,
        'use_norm': 1,
        'down_sampling_layers': 1,
        'down_sampling_window': 2,
        'down_sampling_method': 'avg',
        'use_future_temporal_feature': 0,
        'mask_rate': 0.125,
        'anomaly_ratio': 0.25,
        'num_workers': 0,
        'itr': 1,
        'batch_size': 8,
        'patience': 20,
        'learning_rate': 1e-4,
        'des': 'enhanced',
        'loss': 'MSE',
        'drop_last': True,
        'lradj': 'TST',
        'pct_start': 0.2,
        'use_amp': False,
        'use_gpu': False,
        'gpu': 0,
        'use_multi_gpu': False,
        'devices': '0,1',
        'p_hidden_dims': [128, 128],
        'p_hidden_layers': 2,
    }
    # 如果未提供label_len，则令其等于pred_len
    if 'pred_len' in config and 'label_len' not in config:
        config['label_len'] = config['pred_len']

    merged = {**defaults, **config}
    args = Namespace(**merged)
    return args

def improved_predict_no_smooth(full_series, split_idx, model, args, mean, std):
    """改进的预测方法，无平滑过渡"""
    device = next(model.parameters()).device  # 获取模型所在设备
    ctx = full_series[max(0, split_idx - args.seq_len):split_idx]
    window = np.zeros((args.seq_len, 1), dtype=np.float32)
    ctx_scaled = ((ctx - mean) / std).astype(np.float32).reshape(-1,1)
    window[-len(ctx_scaled):] = ctx_scaled
    
    step_len = min(args.pred_len * 3, 384)
    remain = len(full_series) - split_idx
    extended = []
    
    while remain > 0:
        current = min(step_len, remain)
        x_enc = torch.from_numpy(window[-args.seq_len:]).unsqueeze(0).float().to(device)
        x_mark_enc = torch.zeros((1, args.seq_len, 3), dtype=torch.float32).to(device)
        y_mark = torch.zeros((1, args.label_len + current, 3), dtype=torch.float32).to(device)
        dec_inp = None if args.down_sampling_layers != 0 else torch.zeros((1, args.label_len + current, 1), dtype=torch.float32).to(device)
        
        with torch.no_grad():
            out = model(x_enc, x_mark_enc, dec_inp, y_mark)
            if isinstance(out, tuple):
                out = out[0]
            out = out[:, -current:, :]
            out_np = out.detach().cpu().numpy()[0, :, 0]
        
        out_inv = out_np * std + mean
        extended.append(out_inv)
        
        # 更新窗口
        pred_std = (out_inv - mean) / std
        pred_std = pred_std.reshape(-1,1).astype(np.float32)
        window = np.concatenate([window[current:], pred_std], axis=0)
        if window.shape[0] < args.seq_len:
            pad = np.zeros((args.seq_len - window.shape[0], 1), dtype=np.float32)
            window = np.concatenate([pad, window], axis=0)
        
        remain -= current
    
    return np.concatenate(extended, axis=0)

def enhanced_well_prediction_visualization(exp, well_idx, ratio, output_dir, plot_color='orange'):
    """增强的井预测可视化（无平滑过渡）"""
    print(f"  Well {well_idx}, Ratio {ratio*100:.0f}%")
    
    # 获取测试数据（直接通过数据工厂创建）
    test_dataset, _ = data_provider(exp.args, 'test')
    test_data = test_dataset
    full_series = test_data.well_series[well_idx]
    total_len = len(full_series)
    split_idx = int(total_len * ratio)
    
    # 确保有足够的数据（放宽条件以适应更长的seq_len）
    min_input_len = 100  # 最小输入长度
    if split_idx < min_input_len or (total_len - split_idx) < exp.args.pred_len:
        print(f"    Skipped: insufficient data (total: {total_len}, split: {split_idx})")
        return None
    
    # 计算统计量用于反归一化
    mean = test_data.scaler.mean_[0]
    std = test_data.scaler.scale_[0]
    
    # 进行预测
    pred = improved_predict_no_smooth(full_series, split_idx, exp.model, exp.args, mean, std)
    
    # 确保预测长度不超过剩余数据
    max_pred_len = min(len(pred), total_len - split_idx)
    pred = pred[:max_pred_len]
    
    # 获取真实值
    true_output = full_series[split_idx:split_idx + len(pred)]
    
    # 计算评估指标
    mae = np.mean(np.abs(pred - true_output))
    rmse = np.sqrt(np.mean((pred - true_output) ** 2))
    mape = np.mean(np.abs((pred - true_output) / (true_output + 1e-8))) * 100
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # 计算输入段的起始位置（动态长度）
    input_start_idx = max(0, split_idx - exp.args.seq_len)
    
    # 绘制不同部分
    x_full = np.arange(total_len)
    
    # 早期历史（紫色）
    if input_start_idx > 0:
        ax.plot(x_full[:input_start_idx], full_series[:input_start_idx], 
                color='purple', linewidth=1.5, alpha=0.7, label='早期历史')
    
    # 输入段（蓝色）
    ax.plot(x_full[input_start_idx:split_idx], full_series[input_start_idx:split_idx], 
            color='blue', linewidth=2, label='输入段')
    
    # 真实输出段（绿色）
    true_x = x_full[split_idx:split_idx + len(pred)]
    ax.plot(true_x, true_output, color='green', linewidth=2, label='真实输出')
    
    # 预测输出段（橙色/黄色）
    ax.plot(true_x, pred, color=plot_color, linewidth=2, label='预测输出')
    
    # 添加分割线（红色虚线=预测起点，蓝色点线=输入起点）
    ax.axvline(x=split_idx, color='red', linestyle='--', alpha=0.8, linewidth=2, label='预测起点')
    ax.axvline(x=input_start_idx, color='blue', linestyle=':', alpha=0.8, linewidth=1.5, label='输入起点')
    
    # 设置图形属性
    ax.set_xlabel('时间步', fontsize=12)
    ax.set_ylabel('产量', fontsize=12)
    ax.set_title(f'井 {well_idx} - 分割比例 {ratio*100:.0f}% (无平滑过渡)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, total_len - 1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 添加统计信息
    stats_text = f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nMAPE: {mape:.1f}%'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 保存图形
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f'well_{well_idx}_ratio_{ratio*100:.0f}_no_smooth.pdf')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存CSV
    csv_data = pd.DataFrame({
        'time_step': true_x,
        'true_value': true_output,
        'predicted_value': pred,
        'input_end': full_series[split_idx-1],
        'prediction_start': pred[0] if len(pred) > 0 else np.nan
    })
    csv_path = os.path.join(output_dir, f'well_{well_idx}_ratio_{ratio*100:.0f}_no_smooth.csv')
    csv_data.to_csv(csv_path, index=False)
    
    return {
        'well_idx': well_idx,
        'ratio': ratio,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'input_end': full_series[split_idx-1],
        'prediction_start': pred[0] if len(pred) > 0 else np.nan,
        'jump': pred[0] - full_series[split_idx-1] if len(pred) > 0 else np.nan
    }

def run_test_and_visualize(args, test_wells, ratios, output_dir):
    """运行测试和可视化"""
    print(f"Starting test and visualization...")
    print(f"Model ID: {args.model_id}")
    print(f"Test wells: {test_wells}")
    print(f"Split ratios: {[f'{r*100:.0f}%' for r in ratios]}")
    print(f"Output directory: {output_dir}")
    
    # 创建实验对象
    exp = Exp_Long_Term_Forecast(args)
    
    # 加载模型
    exp.model.load_state_dict(torch.load(f"checkpoints/{args.model_id}/checkpoint.pth", map_location='cpu'))
    exp.model.eval()
    
    # 运行测试
    results = []
    for well_idx in test_wells:
        for ratio in ratios:
            try:
                result = enhanced_well_prediction_visualization(exp, well_idx, ratio, output_dir)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"    Error in well {well_idx}, ratio {ratio*100:.0f}%: {e}")
                continue
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(results)
    if results_df.empty:
        print("No results generated. Please check model checkpoints, test_wells or ratios.")
        return results_df, pd.DataFrame(), pd.DataFrame()
    
    # 计算按井汇总
    by_well = results_df.groupby('well_idx').agg({
        'mae': 'mean',
        'rmse': 'mean', 
        'mape': 'mean',
        'jump': 'mean'
    }).round(4)
    
    # 计算整体汇总
    overall = results_df.agg({
        'mae': 'mean',
        'rmse': 'mean',
        'mape': 'mean',
        'jump': 'mean'
    }).round(4)
    overall = pd.DataFrame([overall])
    
    # 保存结果
    results_df.to_csv(os.path.join(output_dir, 'detailed_results_no_smooth.csv'), index=False)
    by_well.to_csv(os.path.join(output_dir, 'by_well_summary_no_smooth.csv'))
    overall.to_csv(os.path.join(output_dir, 'overall_summary_no_smooth.csv'), index=False)
    
    print(f"\nResults saved to {output_dir}")
    print(f"Total samples: {len(results_df)}")
    
    return results_df, by_well, overall

def main():
    parser = argparse.ArgumentParser(description='TimeMixer Well Lifecycle Testing and Visualization (No Smooth)')
    
    # 实验标识
    parser.add_argument('--model_id', type=str, required=True, help='Experiment ID')
    
    # 测试参数
    parser.add_argument('--test_wells', type=str, default='0,1,2,3,4,5,6,7,8,9', help='Comma-separated well indices to test')
    parser.add_argument('--ratios', type=str, default='10,20,30,40,50,60,70,80,90', help='Comma-separated split ratios (%)')
    
    # 输出目录
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (default: results_archive/{model_id})')
    
    args = parser.parse_args()
    
    # 解析参数
    test_wells = [int(x.strip()) for x in args.test_wells.split(',')]
    ratios = [int(x.strip())/100 for x in args.ratios.split(',')]
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = f"results_archive/{args.model_id}_no_smooth"
    
    # 加载实验配置
    config_args = load_experiment_config(args.model_id)
    
    # 运行测试和可视化
    results_df, by_well, overall = run_test_and_visualize(config_args, test_wells, ratios, args.output_dir)
    
    # 打印汇总结果
    print(f"\nOverall Results Summary (No Smooth Transition):")
    print(f"Average MAE: {overall['mae'].iloc[0]:.2f}")
    print(f"Average RMSE: {overall['rmse'].iloc[0]:.2f}")
    print(f"Average MAPE: {overall['mape'].iloc[0]:.2f}%")
    print(f"Average Jump: {overall['jump'].iloc[0]:.2f}")

if __name__ == "__main__":
    main()
