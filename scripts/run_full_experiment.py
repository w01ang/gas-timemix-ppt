#!/usr/bin/env python3
"""
TimeMixer Full Experiment Pipeline (No Smooth Transition)
完整的实验流程脚本（无平滑过渡）
"""

import os
import sys
import argparse
import subprocess
import datetime
from pathlib import Path

def run_command(cmd, step_name):
    """运行命令并处理结果"""
    print(f"\n{'='*60}")
    print(f"🔄 {step_name}")
    print(f"📝 Command: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=False)
        print(f"✅ {step_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {step_name} failed with exit code {e.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description='TimeMixer Full Experiment Pipeline (No Smooth)')
    
    # 实验标识
    parser.add_argument('--model_id', type=str, required=True, help='Experiment ID')
    parser.add_argument('--comment', type=str, default='no_smooth_experiment', help='Experiment comment')
    parser.add_argument('--description', type=str, default='No smooth transition experiment', help='Experiment description')
    
    # 模型参数
    parser.add_argument('--seq_len', type=int, default=3000, help='Input sequence length')
    parser.add_argument('--label_len', type=int, default=256, help='Label length')
    parser.add_argument('--pred_len', type=int, default=256, help='Prediction length')
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
    
    # 测试参数
    parser.add_argument('--test_wells', type=str, default='0,1,2,3,4,5,6,7,8,9', help='Comma-separated well indices to test')
    parser.add_argument('--ratios', type=str, default='10,20,30,40,50,60,70,80,90', help='Comma-separated split ratios (%)')
    
    # 流程控制
    parser.add_argument('--skip_training', action='store_true', help='Skip training step')
    parser.add_argument('--skip_testing', action='store_true', help='Skip testing step')
    parser.add_argument('--skip_plotting', action='store_true', help='Skip plotting step')
    parser.add_argument('--skip_archiving', action='store_true', help='Skip archiving step')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting full experiment pipeline for: {args.model_id}")
    print(f"📅 Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 步骤1: 训练
    if not args.skip_training:
        train_cmd = f"python scripts/train_experiment.py --model_id {args.model_id} --comment {args.comment} --description '{args.description}' --seq_len {args.seq_len} --label_len {args.label_len} --pred_len {args.pred_len} --d_model {args.d_model} --n_heads {args.n_heads} --e_layers {args.e_layers} --d_layers {args.d_layers} --d_ff {args.d_ff} --train_epochs {args.train_epochs} --batch_size {args.batch_size} --patience {args.patience} --learning_rate {args.learning_rate}"
        
        if not run_command(train_cmd, "Training Model"):
            print("❌ Training failed, stopping pipeline")
            return
    
    # 步骤2: 测试和可视化
    if not args.skip_testing:
        test_cmd = f"python scripts/test_and_visualize.py --model_id {args.model_id} --test_wells {args.test_wells} --ratios {args.ratios}"
        
        if not run_command(test_cmd, "Testing and Visualization"):
            print("❌ Testing failed, stopping pipeline")
            return
    
    # 步骤3: 指标可视化
    if not args.skip_plotting:
        plot_cmd = f"python scripts/plot_metrics.py --results_dir results_archive/{args.model_id}_no_smooth"
        
        if not run_command(plot_cmd, "Metrics Visualization"):
            print("❌ Plotting failed, stopping pipeline")
            return
    
    # 步骤4: 归档
    if not args.skip_archiving:
        archive_cmd = f"python scripts/archive_experiment.py --model_id {args.model_id}_no_smooth --archive_name {args.model_id}_no_smooth_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if not run_command(archive_cmd, "Archiving Results"):
            print("❌ Archiving failed, stopping pipeline")
            return
    
    print(f"\n🎉 Full experiment pipeline completed successfully!")
    print(f"📁 Results available in: results_archive/{args.model_id}_no_smooth")

if __name__ == "__main__":
    main()
