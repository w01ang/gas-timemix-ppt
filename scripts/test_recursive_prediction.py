#!/usr/bin/env python3
"""
递归预测测试与可视化脚本
使用滑动窗口递归预测：每次预测后，将预测结果拼接到输入序列，然后继续预测
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
        step_len=config.get('step_len', config['output_len']),
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
        batch_size=1,
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

def predict_one_step(model, input_seq, input_len, output_len, scaler, device):
    """
    单次预测
    
    Args:
        model: 训练好的模型
        input_seq: 输入序列（原始值）
        input_len: 输入长度
        output_len: 输出长度
        scaler: 数据标准化器
        device: 计算设备
    
    Returns:
        预测结果（原始值）
    """
    # 确保输入长度正确
    if len(input_seq) > input_len:
        input_seq = input_seq[-input_len:]
    elif len(input_seq) < input_len:
        # 如果输入不足，用零填充
        pad = np.zeros((input_len - len(input_seq),), dtype=np.float32)
        input_seq = np.concatenate([pad, input_seq])
    
    # 标准化
    input_normalized = scaler._transform(input_seq).reshape(-1, 1)
    
    # 构造模型输入
    x_enc = torch.from_numpy(input_normalized).float().unsqueeze(0).to(device)
    x_mark_enc = torch.zeros((1, input_len, 3)).to(device)
    y_mark = torch.zeros((1, output_len, 3)).to(device)
    dec_inp = torch.zeros((1, output_len, 1)).to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(x_enc, x_mark_enc, dec_inp, y_mark)
        pred = outputs.detach().cpu().numpy()[0, :, 0]
    
    # 反标准化
    pred = scaler.inverse_transform(pred)
    
    return pred

def recursive_predict_well(exp, well_data, input_len, output_len, max_steps=None):
    """
    对单口井进行递归预测
    
    Args:
        exp: 实验对象
        well_data: 井的完整序列
        input_len: 输入长度
        output_len: 每次预测长度
        max_steps: 最大预测步数（None则预测到原始序列长度）
    
    Returns:
        dict: 包含预测历史的字典
    """
    model = exp.model
    device = next(model.parameters()).device
    scaler = exp.test_data.dataset
    model.eval()
    
    well_length = len(well_data)
    
    # 初始输入：井的前input_len步
    current_sequence = well_data[:input_len].copy()
    
    # 存储预测历史
    prediction_history = []
    
    # 计算需要预测的次数
    if max_steps is None:
        remaining_length = well_length - input_len
        num_predictions = (remaining_length + output_len - 1) // output_len
    else:
        num_predictions = max_steps
    
    print(f"      初始输入长度: {len(current_sequence)}")
    print(f"      目标长度: {well_length}")
    print(f"      需要预测: {num_predictions} 次")
    
    for step in range(num_predictions):
        # 获取当前输入（最后input_len步）
        if len(current_sequence) >= input_len:
            current_input = current_sequence[-input_len:]
        else:
            current_input = current_sequence
        
        # 预测下一个output_len步
        pred = predict_one_step(
            model, current_input, input_len, output_len, scaler, device
        )
        
        # 记录预测
        pred_start_idx = len(current_sequence)
        pred_end_idx = pred_start_idx + output_len
        
        # 获取真实值（如果存在）
        if pred_end_idx <= well_length:
            true_values = well_data[pred_start_idx:pred_end_idx]
        else:
            # 超出原始序列长度，只取部分真实值
            true_values = well_data[pred_start_idx:well_length]
            pred = pred[:len(true_values)]  # 截断预测值
        
        prediction_history.append({
            'step': step + 1,
            'pred_start_idx': pred_start_idx,
            'pred_end_idx': pred_start_idx + len(pred),
            'predicted': pred.copy(),
            'true': true_values.copy(),
            'input_start_idx': max(0, pred_start_idx - input_len),
            'input_end_idx': pred_start_idx
        })
        
        # 将预测结果拼接到当前序列
        current_sequence = np.concatenate([current_sequence, pred])
        
        # 如果已经达到或超过目标长度，停止
        if len(current_sequence) >= well_length:
            break
    
    return {
        'well_length': well_length,
        'input_len': input_len,
        'output_len': output_len,
        'predictions': prediction_history,
        'final_sequence': current_sequence,
        'original_sequence': well_data
    }

def visualize_recursive_prediction(well_idx, result, output_dir, model_id):
    """
    可视化递归预测结果
    """
    fig, ax = plt.subplots(figsize=(20, 6))
    
    well_data = result['original_sequence']
    well_length = result['well_length']
    input_len = result['input_len']
    predictions = result['predictions']
    
    x_full = np.arange(well_length)
    
    # 1. 绘制完整真实序列（浅灰色背景）
    ax.plot(x_full, well_data, color='lightgray', linewidth=1.5, alpha=0.6, 
            label='真实序列（完整）', zorder=1)
    
    # 2. 绘制初始输入段（深蓝色）
    ax.plot(x_full[:input_len], well_data[:input_len], 
            color='darkblue', linewidth=2.5, label='初始输入段', zorder=3)
    
    # 3. 标记初始输入结束点
    ax.axvline(x=input_len, color='blue', linestyle=':', alpha=0.7, 
              linewidth=2, label='初始输入终点', zorder=2)
    
    # 4. 绘制每次预测的结果
    colors = plt.cm.rainbow(np.linspace(0, 1, len(predictions)))
    
    for i, (pred_info, color) in enumerate(zip(predictions, colors)):
        pred_start = pred_info['pred_start_idx']
        pred_end = pred_info['pred_end_idx']
        
        # 预测值（虚线）
        x_pred = np.arange(pred_start, pred_end)
        if i == 0:
            ax.plot(x_pred, pred_info['predicted'], 
                   color='orange', linewidth=2, linestyle='--', alpha=0.8,
                   label='递归预测值', zorder=4)
        else:
            ax.plot(x_pred, pred_info['predicted'], 
                   color='orange', linewidth=2, linestyle='--', alpha=0.8, zorder=4)
        
        # 标记每次预测的起点
        if i > 0:  # 跳过第一次（已经有初始输入终点标记）
            ax.axvline(x=pred_start, color='red', linestyle='--', alpha=0.3, 
                      linewidth=0.8, zorder=2)
    
    # 5. 添加图例和标签
    ax.set_xlabel('时间步', fontsize=13, fontweight='bold')
    ax.set_ylabel('日产量', fontsize=13, fontweight='bold')
    ax.set_title(f'井 {well_idx} - 递归预测 (共{len(predictions)}次预测，每次{result["output_len"]}步)', 
                fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, well_length)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = f"{output_dir}/well_{well_idx}_recursive_prediction.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"      已保存可视化: {output_path}")

def calculate_metrics(true_values, predicted_values):
    """计算评估指标"""
    mae = np.mean(np.abs(true_values - predicted_values))
    rmse = np.sqrt(np.mean((true_values - predicted_values) ** 2))
    
    # 避免除零
    non_zero_mask = np.abs(true_values) > 1e-8
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((true_values[non_zero_mask] - predicted_values[non_zero_mask]) 
                              / true_values[non_zero_mask])) * 100
    else:
        mape = 0.0
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def main():
    parser = argparse.ArgumentParser(description='递归预测测试与可视化')
    parser.add_argument('--model_id', type=str, required=True, help='实验ID')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='输出目录（默认: test_results/<model_id>_recursive）')
    parser.add_argument('--max_wells', type=int, default=None, 
                       help='最多测试多少口井（默认: 全部）')
    parser.add_argument('--max_steps', type=int, default=None, 
                       help='每口井最多预测多少次（默认: 预测到原始序列长度）')
    
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
        args.output_dir = f"test_results/{args.model_id}_recursive"
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取测试数据
    print(f"\n🧪 开始递归预测...")
    test_wells = exp.test_data.dataset.well_series
    
    input_len = config['input_len']
    output_len = config['output_len']
    
    print(f"   测试集井数: {len(test_wells)}")
    print(f"   输入长度: {input_len}")
    print(f"   每次预测长度: {output_len}")
    print(f"   预测策略: 递归预测（每次预测后拼接到输入序列）")
    
    # 测试每口井
    all_results = []
    well_summaries = []
    max_wells = args.max_wells if args.max_wells else len(test_wells)
    
    for well_idx, well_data in enumerate(test_wells[:max_wells]):
        print(f"\n🔹 测试井 {well_idx} (长度: {len(well_data)})...")
        
        if len(well_data) <= input_len:
            print(f"      ⚠️  井 {well_idx} 长度不足（需要>{input_len}），跳过")
            continue
        
        # 递归预测
        result = recursive_predict_well(
            exp, well_data, input_len, output_len, max_steps=args.max_steps
        )
        
        # 计算每次预测的指标
        step_metrics = []
        for pred_info in result['predictions']:
            metrics = calculate_metrics(pred_info['true'], pred_info['predicted'])
            step_metrics.append({
                'well_idx': well_idx,
                'step': pred_info['step'],
                'pred_start_idx': pred_info['pred_start_idx'],
                'pred_end_idx': pred_info['pred_end_idx'],
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE'],
                'MAPE': metrics['MAPE']
            })
            all_results.append(step_metrics[-1])
        
        # 计算整体指标（所有预测步合并）
        all_pred = np.concatenate([p['predicted'] for p in result['predictions']])
        all_true = np.concatenate([p['true'] for p in result['predictions']])
        overall_metrics = calculate_metrics(all_true, all_pred)
        
        well_summaries.append({
            'well_idx': well_idx,
            'well_length': len(well_data),
            'num_predictions': len(result['predictions']),
            'total_pred_length': len(all_pred),
            'overall_MAE': overall_metrics['MAE'],
            'overall_RMSE': overall_metrics['RMSE'],
            'overall_MAPE': overall_metrics['MAPE']
        })
        
        print(f"      完成 {len(result['predictions'])} 次预测")
        print(f"      整体 MAE: {overall_metrics['MAE']:.2f}")
        print(f"      整体 RMSE: {overall_metrics['RMSE']:.2f}")
        print(f"      整体 MAPE: {overall_metrics['MAPE']:.2f}%")
        
        # 可视化
        visualize_recursive_prediction(
            well_idx, result, args.output_dir, args.model_id
        )
    
    # 保存详细结果
    if all_results:
        # 保存每步结果
        results_df = pd.DataFrame(all_results)
        results_path = f"{args.output_dir}/step_by_step_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\n📄 详细结果已保存: {results_path}")
        
        # 保存井汇总
        summary_df = pd.DataFrame(well_summaries)
        summary_path = f"{args.output_dir}/well_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"📄 井汇总已保存: {summary_path}")
        
        # 整体统计
        print(f"\n📊 递归预测整体评估:")
        print(f"   测试井数: {len(well_summaries)}")
        print(f"   总预测次数: {len(all_results)}")
        print(f"   平均每井预测次数: {len(all_results)/len(well_summaries):.1f}")
        print(f"\n   整体平均指标:")
        print(f"   - MAE: {summary_df['overall_MAE'].mean():.2f}")
        print(f"   - RMSE: {summary_df['overall_RMSE'].mean():.2f}")
        print(f"   - MAPE: {summary_df['overall_MAPE'].mean():.2f}%")
        
        print(f"\n   按预测步数的指标变化:")
        step_stats = results_df.groupby('step').agg({
            'MAE': 'mean',
            'RMSE': 'mean',
            'MAPE': 'mean'
        }).reset_index()
        for _, row in step_stats.head(10).iterrows():
            print(f"   步骤 {int(row['step'])}: MAE={row['MAE']:.2f}, "
                  f"RMSE={row['RMSE']:.2f}, MAPE={row['MAPE']:.2f}%")
        
        # 保存步骤统计
        step_stats_path = f"{args.output_dir}/step_statistics.csv"
        step_stats.to_csv(step_stats_path, index=False)
        print(f"\n📄 步骤统计已保存: {step_stats_path}")
    
    print(f"\n✅ 递归预测测试完成！")
    print(f"📁 所有结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main()

