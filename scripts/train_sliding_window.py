#!/usr/bin/env python3
"""
TimeMixer 滑动窗口训练脚本
使用固定input_len和output_len的滑动窗口进行训练
"""

import os
import sys
import argparse
import json
import datetime
from pathlib import Path
import platform

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from argparse import Namespace
import torch

def train_sliding_window_model(args):
    """训练滑动窗口模型"""
    print(f"🚀 开始训练滑动窗口TimeMixer模型...")
    print(f"📊 配置参数:")
    print(f"   模型ID: {args.model_id}")
    print(f"   输入长度 (input_len): {args.seq_len}")
    print(f"   输出长度 (output_len): {args.pred_len}")
    print(f"   滑动步长 (step_len): {args.step_len}")
    print(f"   比例: {args.seq_len}:{args.pred_len} = {args.seq_len/args.pred_len:.2f}:1")
    print(f"   模型维度: {args.d_model}")
    print(f"   训练轮数: {args.train_epochs}")
    
    # 创建实验对象
    exp = Exp_Long_Term_Forecast(args)
    
    # 开始训练
    print(f"\n🔄 开始训练...")
    exp.train(args.model_id)
    
    print(f"✅ 滑动窗口模型训练完成！")
    print(f"📁 模型保存在: checkpoints/{args.model_id}/")

def main():
    parser = argparse.ArgumentParser(description='TimeMixer 滑动窗口训练脚本')
    
    # 实验标识
    parser.add_argument('--model_id', type=str, required=True, help='实验ID（唯一标识）')
    parser.add_argument('--comment', type=str, default='sliding_window', help='实验注释')
    parser.add_argument('--description', type=str, default='Sliding window model', help='实验描述')
    
    # 数据路径
    parser.add_argument('--root_path', type=str, default='/Users/wangjr/Documents/yk/timemixer/data', 
                        help='数据集根目录')
    parser.add_argument('--data_path', type=str, default='preprocessed_daily_gas_by_well.csv', 
                        help='数据集文件名')
    
    # 核心窗口参数
    parser.add_argument('--input_len', type=int, required=True, 
                        help='输入序列长度（固定）')
    parser.add_argument('--output_len', type=int, required=True, 
                        help='输出序列长度（固定）')
    parser.add_argument('--step_len', type=int, default=None, 
                        help='滑动窗口步长（默认=output_len，即无重叠）')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=256, help='模型维度')
    parser.add_argument('--n_heads', type=int, default=16, help='注意力头数')
    parser.add_argument('--e_layers', type=int, default=6, help='编码器层数')
    parser.add_argument('--d_layers', type=int, default=3, help='解码器层数')
    parser.add_argument('--d_ff', type=int, default=1024, help='前馈网络维度')
    parser.add_argument('--use_gpu', action='store_true', help='启用GPU/MPS加速')
    
    # 训练参数
    parser.add_argument('--train_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批大小')
    parser.add_argument('--patience', type=int, default=20, help='早停耐心值')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    
    args = parser.parse_args()
    
    # 设置默认步长
    if args.step_len is None:
        args.step_len = args.output_len
        print(f"ℹ️  步长未指定，使用默认值: step_len = output_len = {args.step_len}")
    
    # 显示参数信息
    print(f"\n📊 滑动窗口配置:")
    print(f"   输入长度 (input_len): {args.input_len}")
    print(f"   输出长度 (output_len): {args.output_len}")
    print(f"   滑动步长 (step_len): {args.step_len}")
    print(f"   窗口总长度: {args.input_len + args.output_len}")
    
    if args.step_len < args.output_len:
        overlap = args.output_len - args.step_len
        overlap_pct = (overlap / args.output_len) * 100
        print(f"   窗口重叠: {overlap} 步 ({overlap_pct:.1f}%)")
    elif args.step_len == args.output_len:
        print(f"   窗口重叠: 无重叠（相邻窗口）")
    else:
        gap = args.step_len - args.output_len
        print(f"   窗口间隙: {gap} 步")
    
    print(f"   输入:输出比例 = {args.input_len}:{args.output_len} = {args.input_len/args.output_len:.2f}:1")
    
    # 构建模型配置
    model_args = Namespace(
        task_name='long_term_forecast',
        is_training=1,
        model_id=args.model_id,
        model='TimeMixer',
        data='WELLS',
        root_path=args.root_path,
        data_path=args.data_path,
        features='S',
        target='OT',
        freq='d',
        checkpoints='./checkpoints/',
        seq_len=args.input_len,      # 输入长度
        label_len=args.output_len,   # 标签长度（通常等于pred_len）
        pred_len=args.output_len,    # 预测长度
        step_len=args.step_len,      # 滑动步长（新增参数）
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
        use_gpu=args.use_gpu,
        gpu=0,
        use_multi_gpu=False,
        devices='0,1',
        p_hidden_dims=[128, 128],
        p_hidden_layers=2
    )

    # 自动检测GPU可用性
    if model_args.use_gpu and platform.system() == 'Darwin':
        if not torch.backends.mps.is_available():
            print('⚠️  MPS不可用，回退到CPU')
            model_args.use_gpu = False
        else:
            print('✅ 使用MPS加速')
    elif model_args.use_gpu and torch.cuda.is_available():
        print('✅ 使用CUDA加速')
    elif model_args.use_gpu:
        print('⚠️  GPU不可用，回退到CPU')
        model_args.use_gpu = False
    
    # 保存实验配置
    config_dir = f"experiments/{args.model_id}"
    os.makedirs(config_dir, exist_ok=True)
    
    config = {
        "model_id": args.model_id,
        "comment": args.comment,
        "description": args.description,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "step_len": args.step_len,
        "window_total_len": args.input_len + args.output_len,
        "input_output_ratio": args.input_len / args.output_len,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "e_layers": args.e_layers,
        "d_layers": args.d_layers,
        "d_ff": args.d_ff,
        "train_epochs": args.train_epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "use_gpu": model_args.use_gpu,
        "root_path": args.root_path,
        "data_path": args.data_path,
        "created_at": datetime.datetime.now().isoformat()
    }
    
    with open(f"{config_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"📁 配置已保存到: {config_dir}/config.json")
    
    # 开始训练
    train_sliding_window_model(model_args)

if __name__ == "__main__":
    main()

