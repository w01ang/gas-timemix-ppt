#!/usr/bin/env python3
"""
TimeMixer 8:2比例专用训练脚本
输入不定长时序数据序列，输出定长序列片段（输入:输出=8:2）
"""

import os
import sys
import argparse
import json
import datetime
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from argparse import Namespace

def train_8_2_ratio_model(args):
    """训练8:2比例模型"""
    print(f"🚀 开始训练8:2比例TimeMixer模型...")
    print(f"📊 配置参数:")
    print(f"   模型ID: {args.model_id}")
    print(f"   输入长度: {args.seq_len} (80%)")
    print(f"   输出长度: {args.pred_len} (20%)")
    print(f"   比例: {args.seq_len}:{args.pred_len} = {args.seq_len/args.pred_len:.1f}:1")
    print(f"   模型维度: {args.d_model}")
    print(f"   训练轮数: {args.train_epochs}")
    
    # 创建实验对象
    exp = Exp_Long_Term_Forecast(args)
    
    # 开始训练
    print(f"\n🔄 开始训练...")
    exp.train(args.model_id)
    
    print(f"✅ 8:2比例模型训练完成！")
    print(f"📁 模型保存在: checkpoints/{args.model_id}/")

def main():
    parser = argparse.ArgumentParser(description='TimeMixer 8:2比例专用训练脚本')
    
    # 实验标识
    parser.add_argument('--model_id', type=str, required=True, help='Experiment ID')
    parser.add_argument('--comment', type=str, default='8_2_ratio', help='Experiment comment')
    parser.add_argument('--description', type=str, default='8:2 input-output ratio model', help='Experiment description')
    
    # 8:2比例参数
    parser.add_argument('--total_length', type=int, default=1000, help='Total sequence length for 8:2 ratio calculation')
    parser.add_argument('--input_ratio', type=float, default=0.8, help='Input ratio (default: 0.8)')
    parser.add_argument('--output_ratio', type=float, default=0.2, help='Output ratio (default: 0.2)')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--e_layers', type=int, default=6, help='Number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=3, help='Number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=1024, help='Feed-forward dimension')
    
    # 训练参数
    parser.add_argument('--train_epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    # 计算8:2比例参数
    seq_len = int(args.total_length * args.input_ratio)
    pred_len = int(args.total_length * args.output_ratio)
    
    # 验证比例
    actual_ratio = seq_len / pred_len
    expected_ratio = args.input_ratio / args.output_ratio
    
    print(f"📊 8:2比例计算:")
    print(f"   总长度: {args.total_length}")
    print(f"   输入比例: {args.input_ratio*100:.0f}%")
    print(f"   输出比例: {args.output_ratio*100:.0f}%")
    print(f"   输入长度: {seq_len}")
    print(f"   输出长度: {pred_len}")
    print(f"   实际比例: {actual_ratio:.1f}:1")
    print(f"   期望比例: {expected_ratio:.1f}:1")
    
    if abs(actual_ratio - expected_ratio) > 0.1:
        print(f"⚠️  警告: 实际比例与期望比例差异较大")
    
    # 设置模型参数
    model_args = Namespace(
        task_name='long_term_forecast',
        is_training=1,
        model_id=args.model_id,
        model='TimeMixer',
        data='WELLS',
        root_path='/Users/wangjr/Documents/yk/timemixer/data',
        data_path='preprocessed_daily_gas_by_well.csv',
        features='S',
        target='OT',
        freq='d',
        checkpoints='./checkpoints/',
        seq_len=seq_len,
        label_len=pred_len,  # label_len通常等于pred_len
        pred_len=pred_len,
        seasonal_patterns='Monthly',
        inverse=True,
        top_k=5,
        num_kernels=6,
        enc_in=1,
        dec_in=1,
        c_out=1,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_layers=args.d_layers,
        d_ff=args.d_ff,
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
        train_epochs=args.train_epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        learning_rate=args.learning_rate,
        des=args.description,
        loss='MSE',
        drop_last=True,
        lradj='TST',
        pct_start=0.2,
        use_amp=False,
        comment=args.comment,
        use_gpu=False,
        gpu=0,
        use_multi_gpu=False,
        devices='0,1',
        p_hidden_dims=[128, 128],
        p_hidden_layers=2
    )
    
    # 保存配置
    config_dir = f"experiments/{args.model_id}"
    os.makedirs(config_dir, exist_ok=True)
    
    config = {
        "model_id": args.model_id,
        "comment": args.comment,
        "description": args.description,
        "total_length": args.total_length,
        "input_ratio": args.input_ratio,
        "output_ratio": args.output_ratio,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "actual_ratio": actual_ratio,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "e_layers": args.e_layers,
        "d_layers": args.d_layers,
        "d_ff": args.d_ff,
        "train_epochs": args.train_epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "created_at": datetime.datetime.now().isoformat()
    }
    
    with open(f"{config_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"📁 配置已保存到: {config_dir}/config.json")
    
    # 开始训练
    train_8_2_ratio_model(model_args)

if __name__ == "__main__":
    main()
