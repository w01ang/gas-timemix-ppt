#!/usr/bin/env python3
"""
测试无平滑过渡的预测效果（修复版）
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from argparse import Namespace

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def test_no_smooth_prediction():
    """测试无平滑过渡的预测"""
    print("🚀 开始测试无平滑过渡的预测效果...")
    
    # 设置参数
    args = Namespace(
        task_name='long_term_forecast',
        is_training=0,
        model_id='wellmix_dynamic_input_v2',
        model='TimeMixer',
        data='WELLS',
        root_path='/Users/wangjr/Documents/yk/timemixer/data',
        data_path='preprocessed_daily_gas_by_well.csv',
        features='S',
        target='OT',
        freq='d',
        checkpoints='./checkpoints/',
        seq_len=3000,
        label_len=256,
        pred_len=256,
        seasonal_patterns='Monthly',
        inverse=True,
        top_k=5,
        num_kernels=6,
        enc_in=1,
        dec_in=1,
        c_out=1,
        d_model=256,
        n_heads=16,
        e_layers=6,
        d_layers=3,
        d_ff=1024,
        moving_avg=49,
        factor=1,
        distil=True,
        dropout=0.1,
        embed='timeF',
        activation='gelu',
        output_attention=False,
        channel_independence=1,
        decomp_method='moving_avg',
        use_norm=1,
        down_sampling_layers=1,
        down_sampling_window=2,
        down_sampling_method='avg',
        use_future_temporal_feature=0,
        mask_rate=0.125,
        anomaly_ratio=0.25,
        num_workers=0,
        itr=1,
        train_epochs=100,
        batch_size=8,
        patience=20,
        learning_rate=0.0001,
        des='enhanced',
        loss='MSE',
        drop_last=True,
        lradj='TST',
        pct_start=0.2,
        use_amp=False,
        comment='dynamic_input_v2',
        use_gpu=False,
        gpu=0,
        use_multi_gpu=False,
        devices='0,1',
        p_hidden_dims=[128, 128],
        p_hidden_layers=2
    )
    
    # 创建实验对象
    exp = Exp_Long_Term_Forecast(args)
    
    # 加载模型
    setting = f"long_term_forecast_{args.model_id}_{args.comment}_{args.model}_{args.data}_sl{args.seq_len}_pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_fc{args.factor}_eb{args.embed}_dt{args.distil}_{args.des}_0"
    exp.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
    exp.model.eval()
    
    # 获取测试数据
    test_data = exp.data_loader.test_dataset
    well_idx = 0  # 测试第一口井
    ratio = 0.5   # 50%分割点
    
    full_series = test_data.well_series[well_idx]
    total_len = len(full_series)
    split_idx = int(total_len * ratio)
    
    print(f"📊 测试井 {well_idx}:")
    print(f"   总长度: {total_len}")
    print(f"   分割点: {split_idx} ({ratio*100:.0f}%)")
    print(f"   输入末尾值: {full_series[split_idx-1]:.2f}")
    
    # 计算统计量用于反归一化
    mean = test_data.scaler.mean_[0]
    std = test_data.scaler.scale_[0]
    
    # 进行预测（无平滑过渡）
    ctx = full_series[max(0, split_idx - args.seq_len):split_idx]
    window = np.zeros((args.seq_len, 1), dtype=np.float32)
    ctx_scaled = ((ctx - mean) / std).astype(np.float32).reshape(-1,1)
    window[-len(ctx_scaled):] = ctx_scaled
    
    step_len = min(args.pred_len * 3, 384)
    remain = len(full_series) - split_idx
    extended = []
    
    while remain > 0:
        current = min(step_len, remain)
        x_enc = torch.from_numpy(window[-args.seq_len:]).unsqueeze(0).float()
        x_mark_enc = torch.zeros((1, args.seq_len, 3), dtype=torch.float32)
        y_mark = torch.zeros((1, args.label_len + current, 3), dtype=torch.float32)
        dec_inp = None if args.down_sampling_layers != 0 else torch.zeros((1, args.label_len + current, 1), dtype=torch.float32)
        
        with torch.no_grad():
            out = exp.model(x_enc, x_mark_enc, dec_inp, y_mark)
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
    
    pred = np.concatenate(extended, axis=0)
    
    # 确保预测长度不超过剩余数据
    max_pred_len = min(len(pred), total_len - split_idx)
    pred = pred[:max_pred_len]
    
    # 获取真实值
    true_output = full_series[split_idx:split_idx + len(pred)]
    
    print(f"   预测起始值: {pred[0]:.2f}")
    print(f"   跳跃大小: {pred[0] - full_series[split_idx-1]:.2f}")
    
    # 计算评估指标
    mae = np.mean(np.abs(pred - true_output))
    rmse = np.sqrt(np.mean((pred - true_output) ** 2))
    mape = np.mean(np.abs((pred - true_output) / (true_output + 1e-8))) * 100
    
    print(f"   评估指标:")
    print(f"     MAE: {mae:.2f}")
    print(f"     RMSE: {rmse:.2f}")
    print(f"     MAPE: {mape:.1f}%")
    
    # 创建对比图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # 上图：完整生命周期
    x_full = np.arange(total_len)
    input_start_idx = max(0, split_idx - args.seq_len)
    
    # 早期历史（紫色）
    if input_start_idx > 0:
        ax1.plot(x_full[:input_start_idx], full_series[:input_start_idx], 
                color='purple', linewidth=1.5, alpha=0.7, label='早期历史')
    
    # 输入段（蓝色）
    ax1.plot(x_full[input_start_idx:split_idx], full_series[input_start_idx:split_idx], 
            color='blue', linewidth=2, label='输入段')
    
    # 真实输出段（绿色）
    true_x = x_full[split_idx:split_idx + len(pred)]
    ax1.plot(true_x, true_output, color='green', linewidth=2, label='真实输出')
    
    # 预测输出段（橙色）
    ax1.plot(true_x, pred, color='orange', linewidth=2, label='预测输出（无平滑）')
    
    # 添加分割线
    ax1.axvline(x=split_idx, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax1.axvline(x=input_start_idx, color='purple', linestyle=':', alpha=0.7, linewidth=1)
    
    ax1.set_xlabel('时间步', fontsize=12)
    ax1.set_ylabel('产量', fontsize=12)
    ax1.set_title(f'井 {well_idx} - 完整生命周期预测（无平滑过渡）', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 下图：预测窗口放大
    ax2.plot(true_x, true_output, color='green', linewidth=2, label='真实输出')
    ax2.plot(true_x, pred, color='orange', linewidth=2, label='预测输出（无平滑）')
    ax2.axvline(x=split_idx, color='red', linestyle='--', alpha=0.7, linewidth=2, label='分割点')
    
    ax2.set_xlabel('时间步', fontsize=12)
    ax2.set_ylabel('产量', fontsize=12)
    ax2.set_title(f'预测窗口放大图 - 跳跃大小: {pred[0] - full_series[split_idx-1]:.2f}', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 添加统计信息
    stats_text = f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nMAPE: {mape:.1f}%\n跳跃: {pred[0] - full_series[split_idx-1]:.2f}'
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图形
    os.makedirs('test_results/no_smooth_test', exist_ok=True)
    plot_path = 'test_results/no_smooth_test/no_smooth_prediction_test.pdf'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 测试结果已保存到: {plot_path}")
    print("✅ 无平滑过渡测试完成！")

if __name__ == "__main__":
    test_no_smooth_prediction()
