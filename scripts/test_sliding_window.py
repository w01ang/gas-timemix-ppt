#!/usr/bin/env python3
"""
滑动窗口模型测试与可视化脚本
与train_sliding_window.py配套使用
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from argparse import Namespace

def load_experiment_config(model_id):
    """加载实验配置"""
    config_path = f"experiments/{model_id}/config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"✅ 已加载实验配置: {config_path}")
    print(f"   输入长度: {config['input_len']}")
    print(f"   输出长度: {config['output_len']}")
    print(f"   滑动步长: {config['step_len']}")
    
    return config

def create_test_args(config):
    """根据配置创建测试参数"""
    args = Namespace(
        task_name='long_term_forecast',
        is_training=0,
        model_id=config['model_id'],
        model='TimeMixer',
        data='WELLS',
        root_path=config.get('root_path', '/Users/wangjr/Documents/yk/timemixer/data'),
        data_path=config.get('data_path', 'preprocessed_daily_gas_by_well.csv'),
        features='S',
        target='OT',
        freq='d',
        checkpoints='./checkpoints/',
        seq_len=config['input_len'],
        label_len=config['output_len'],
        pred_len=config['output_len'],
        step_len=config['step_len'],
        seasonal_patterns='Monthly',
        inverse=True,
        top_k=5,
        num_kernels=6,
        enc_in=1,
        dec_in=1,
        c_out=1,
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        e_layers=config['e_layers'],
        d_layers=config['d_layers'],
        d_ff=config['d_ff'],
        moving_avg=999,  # 改为999（奇数），适合3000步输入的季节性分解
        factor=1,
        distil=True,
        dropout=0.1,
        embed='timeF',
        activation='gelu',
        output_attention=False,
        channel_independence=1,
        decomp_method='moving_avg',
        use_norm=1,
        down_sampling_layers=2,  # 改为2层，产生3个尺度(3000/1500/750)
        down_sampling_window=2,
        down_sampling_method='avg',
        use_future_temporal_feature=0,
        mask_rate=0.125,
        anomaly_ratio=0.25,
        num_workers=0,
        itr=1,
        train_epochs=config['train_epochs'],
        batch_size=1,  # 测试时batch_size=1
        patience=config['patience'],
        learning_rate=config['learning_rate'],
        des=config.get('description', 'Test'),
        loss='MSE',
        drop_last=False,
        lradj='TST',
        pct_start=0.2,
        use_amp=False,
        comment=config.get('comment', 'test'),
        use_gpu=config.get('use_gpu', False),
        gpu=0,
        use_multi_gpu=False,
        devices='0,1',
        p_hidden_dims=[128, 128],
        p_hidden_layers=2
    )
    return args

def predict_well_sliding_windows(exp, well_data, input_len, output_len, step_len):
    """
    对单口井使用滑动窗口进行预测
    
    返回: 所有窗口的预测结果列表
    [(start_idx, true_output, predicted_output), ...]
    """
    model = exp.model
    device = next(model.parameters()).device
    model.eval()
    
    well_length = len(well_data)
    min_window_size = input_len + output_len
    results = []
    
    # 生成滑动窗口
    num_windows = (well_length - min_window_size) // step_len + 1
    
    with torch.no_grad():
        for window_idx in range(num_windows):
            start_idx = window_idx * step_len
            end_idx = start_idx + input_len + output_len
            
            if end_idx > well_length:
                break
            
            # 提取输入和真实输出
            input_seq = well_data[start_idx:start_idx + input_len]
            true_output = well_data[start_idx + input_len:end_idx]
            
            # 标准化
            input_normalized = exp.test_data.dataset._transform(input_seq).reshape(-1, 1)
            
            # 构造模型输入
            x_enc = torch.from_numpy(input_normalized).float().unsqueeze(0).to(device)
            x_mark_enc = torch.zeros((1, input_len, 3)).to(device)
            y_mark = torch.zeros((1, output_len, 3)).to(device)
            
            # 解码器输入（使用部分真实值+零填充）
            dec_inp = torch.zeros((1, output_len, 1)).to(device)
            
            # 预测
            outputs = model(x_enc, x_mark_enc, dec_inp, y_mark)
            pred = outputs.detach().cpu().numpy()[0, :, 0]
            
            # 反标准化
            pred = exp.test_data.dataset.inverse_transform(pred)
            
            results.append({
                'start_idx': start_idx,
                'input_end_idx': start_idx + input_len,
                'end_idx': end_idx,
                'true_output': true_output,
                'predicted_output': pred
            })
    
    return results

def visualize_well_predictions(well_idx, well_data, predictions, input_len, output_len, 
                                output_dir, model_id):
    """
    可视化单口井的所有窗口预测结果
    """
    fig, ax = plt.subplots(figsize=(16, 6))
    
    well_length = len(well_data)
    x_full = np.arange(well_length)
    
    # 绘制完整井序列（浅灰色背景）
    ax.plot(x_full, well_data, color='lightgray', linewidth=1, alpha=0.5, 
            label='完整序列', zorder=1)
    
    # 绘制每个窗口的预测
    colors = plt.cm.rainbow(np.linspace(0, 1, len(predictions)))
    
    for i, (pred_info, color) in enumerate(zip(predictions, colors)):
        start_idx = pred_info['start_idx']
        input_end = pred_info['input_end_idx']
        end_idx = pred_info['end_idx']
        
        # 输入段（蓝色）
        ax.plot(x_full[start_idx:input_end], well_data[start_idx:input_end], 
                color='blue', linewidth=1.5, alpha=0.3, zorder=2)
        
        # 真实输出（绿色）
        ax.plot(x_full[input_end:end_idx], pred_info['true_output'], 
                color='green', linewidth=2, alpha=0.6, zorder=3)
        
        # 预测输出（橙色）
        ax.plot(x_full[input_end:end_idx], pred_info['predicted_output'], 
                color='orange', linewidth=2, alpha=0.6, linestyle='--', zorder=4)
        
        # 标记窗口边界
        if i == 0:  # 只在第一个窗口添加图例
            ax.axvline(x=input_end, color='red', linestyle='--', alpha=0.5, 
                      linewidth=1, label='预测起点', zorder=5)
        else:
            ax.axvline(x=input_end, color='red', linestyle='--', alpha=0.3, 
                      linewidth=0.5, zorder=5)
    
    ax.set_xlabel('时间步', fontsize=12)
    ax.set_ylabel('日产量', fontsize=12)
    ax.set_title(f'井 {well_idx} - 滑动窗口预测 ({len(predictions)}个窗口)', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, well_length)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = f"{output_dir}/well_{well_idx}_sliding_windows.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   已保存: {output_path}")

def calculate_metrics(true_values, predicted_values):
    """计算评估指标"""
    mae = np.mean(np.abs(true_values - predicted_values))
    rmse = np.sqrt(np.mean((true_values - predicted_values) ** 2))
    mape = np.mean(np.abs((true_values - predicted_values) / (true_values + 1e-8))) * 100
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def main():
    parser = argparse.ArgumentParser(description='滑动窗口模型测试与可视化')
    parser.add_argument('--model_id', type=str, required=True, help='实验ID')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录（默认: test_results/<model_id>）')
    parser.add_argument('--max_wells', type=int, default=None, help='最多测试多少口井（默认: 全部）')
    
    args = parser.parse_args()
    
    # 加载配置
    print(f"🔍 加载实验配置...")
    config = load_experiment_config(args.model_id)
    
    # 创建测试参数
    test_args = create_test_args(config)
    
    # 创建实验对象
    print(f"\n📊 初始化实验...")
    exp = Exp_Long_Term_Forecast(test_args)
    
    # 加载测试数据
    _, exp.test_data = exp._get_data(flag='test')
    
    # 加载模型
    checkpoint_path = f"./checkpoints/{args.model_id}/checkpoint.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")
    
    exp.model.load_state_dict(torch.load(checkpoint_path, map_location=exp.device))
    print(f"✅ 已加载模型: {checkpoint_path}")
    
    # 准备输出目录
    if args.output_dir is None:
        args.output_dir = f"test_results/{args.model_id}"
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取测试数据
    print(f"\n🧪 开始测试...")
    test_wells = exp.test_data.dataset.well_series
    
    input_len = config['input_len']
    output_len = config['output_len']
    step_len = config['step_len']
    
    print(f"   测试集井数: {len(test_wells)}")
    print(f"   输入长度: {input_len}, 输出长度: {output_len}, 步长: {step_len}")
    
    # 测试每口井
    all_results = []
    max_wells = args.max_wells if args.max_wells else len(test_wells)
    
    for well_idx, well_data in enumerate(test_wells[:max_wells]):
        print(f"\n🔹 测试井 {well_idx} (长度: {len(well_data)})...")
        
        # 预测所有窗口
        predictions = predict_well_sliding_windows(
            exp, well_data, input_len, output_len, step_len
        )
        
        if len(predictions) == 0:
            print(f"   ⚠️  井 {well_idx} 长度不足，跳过")
            continue
        
        print(f"   生成了 {len(predictions)} 个窗口预测")
        
        # 计算每个窗口的指标
        for pred_info in predictions:
            metrics = calculate_metrics(
                pred_info['true_output'], 
                pred_info['predicted_output']
            )
            all_results.append({
                'well_idx': well_idx,
                'start_idx': pred_info['start_idx'],
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE'],
                'MAPE': metrics['MAPE']
            })
        
        # 可视化
        visualize_well_predictions(
            well_idx, well_data, predictions, 
            input_len, output_len, args.output_dir, args.model_id
        )
    
    # 保存结果
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_path = f"{args.output_dir}/results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\n📄 结果已保存: {results_path}")
        
        # 统计汇总
        print(f"\n📊 总体评估指标:")
        print(f"   平均 MAE: {results_df['MAE'].mean():.4f}")
        print(f"   平均 RMSE: {results_df['RMSE'].mean():.4f}")
        print(f"   平均 MAPE: {results_df['MAPE'].mean():.2f}%")
        
        # 按井统计
        well_stats = results_df.groupby('well_idx').agg({
            'MAE': 'mean',
            'RMSE': 'mean',
            'MAPE': 'mean'
        }).reset_index()
        
        well_stats_path = f"{args.output_dir}/well_statistics.csv"
        well_stats.to_csv(well_stats_path, index=False)
        print(f"   井统计已保存: {well_stats_path}")
    
    print(f"\n✅ 测试完成！")
    print(f"📁 所有结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main()

