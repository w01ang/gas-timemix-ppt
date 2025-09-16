#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TimeMixer Well Lifecycle Prediction - Training Script
井生命周期预测训练脚本

Usage:
    python scripts/train_experiment.py --config experiments/config_template.json
    python scripts/train_experiment.py --model_id my_experiment --seq_len 3000 --d_model 256 --train_epochs 100
"""

import os
import sys
import json
import argparse
import datetime
from pathlib import Path

def create_experiment_config(args):
    """创建实验配置"""
    config = {
        # 基础设置
        "task_name": "long_term_forecast",
        "is_training": 1,
        "model_id": args.model_id,
        "model": "TimeMixer",
        "data": "WELLS",
        "root_path": "/Users/wangjr/Documents/yk/timemixer/data",
        "data_path": "preprocessed_daily_gas_by_well.csv",
        
        # 数据参数
        "features": "S",
        "freq": "d",
        "seq_len": args.seq_len,
        "label_len": args.label_len,
        "pred_len": args.pred_len,
        "inverse": True,
        "enc_in": 1,
        "dec_in": 1,
        "c_out": 1,
        
        # 模型参数
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "e_layers": args.e_layers,
        "d_layers": args.d_layers,
        "d_ff": args.d_ff,
        "moving_avg": 49,
        "factor": 1,
        "use_norm": 1,
        "down_sampling_layers": 1,
        "down_sampling_window": 2,
        
        # 训练参数
        "num_workers": 0,
        "itr": 1,
        "train_epochs": args.train_epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "drop_last": True,
        "lradj": "TST",
        "pct_start": 0.2,
        "use_amp": False,
        
        # 其他
        "use_gpu": True,
        "comment": args.comment,
        "des": "enhanced",
        "loss": "MSE",
        "checkpoints": "./checkpoints/",
        
        # 实验元数据
        "experiment_timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "experiment_description": args.description,
        "git_commit": "unknown"  # 可后续添加git信息
    }
    return config

def save_config(config, output_path):
    """保存配置文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def build_command(config):
    """构建训练命令"""
    cmd_parts = ["python", "run.py"]
    
    # 添加所有参数
    for key, value in config.items():
        if key.startswith("experiment_"):
            continue
        if isinstance(value, bool):
            if value:
                cmd_parts.append(f"--{key}")
        else:
            cmd_parts.append(f"--{key}")
            cmd_parts.append(str(value))
    
    return " ".join(cmd_parts)

def main():
    parser = argparse.ArgumentParser(description='TimeMixer Well Lifecycle Training')
    
    # 实验标识
    parser.add_argument('--model_id', type=str, required=True, help='Experiment ID')
    parser.add_argument('--comment', type=str, default='experiment', help='Comment for experiment')
    parser.add_argument('--description', type=str, default='', help='Experiment description')
    
    # 数据参数
    parser.add_argument('--seq_len', type=int, default=3000, help='Input sequence length')
    parser.add_argument('--label_len', type=int, default=256, help='Label length')
    parser.add_argument('--pred_len', type=int, default=256, help='Prediction length')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=16, help='Number of heads')
    parser.add_argument('--e_layers', type=int, default=6, help='Encoder layers')
    parser.add_argument('--d_layers', type=int, default=3, help='Decoder layers')
    parser.add_argument('--d_ff', type=int, default=1024, help='Feed forward dimension')
    
    # 训练参数
    parser.add_argument('--train_epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    # 创建配置
    config = create_experiment_config(args)
    
    # 保存配置到experiments目录
    config_path = f"experiments/{args.model_id}_config.json"
    save_config(config, config_path)
    
    # 构建并执行训练命令
    cmd = build_command(config)
    print(f"Training command: {cmd}")
    print(f"Config saved to: {config_path}")
    
    # 执行训练
    os.system(cmd)

if __name__ == "__main__":
    main()
