#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TimeMixer Well Lifecycle Prediction - Experiment Archiving Script
井生命周期预测实验归档脚本

Usage:
    python scripts/archive_experiment.py --model_id wellmix_dynamic_input_v2
    python scripts/archive_experiment.py --model_id my_experiment --archive_name my_experiment_v1
"""

import os
import sys
import json
import argparse
import shutil
import datetime
from pathlib import Path

def archive_experiment(model_id, archive_name=None):
    """归档实验"""
    if archive_name is None:
        archive_name = f"{model_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 创建归档目录
    archive_dir = f"results_archive/{archive_name}"
    os.makedirs(archive_dir, exist_ok=True)
    
    # 构建设置名称
    config_path = f"experiments/{model_id}_config.json"
    if not os.path.exists(config_path):
        print(f"Warning: Config file not found: {config_path}")
    else:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 复制配置文件
        shutil.copy2(config_path, os.path.join(archive_dir, f"{model_id}_config.json"))
        
        # 构建checkpoints路径
        setting = f'long_term_forecast_{model_id}_{config["comment"]}_{config["model"]}_{config["data"]}_sl{config["seq_len"]}_pl{config["pred_len"]}_dm{config["d_model"]}_nh{config["n_heads"]}_el{config["e_layers"]}_dl{config["d_layers"]}_df{config["d_ff"]}_fc{config.get("channel_independence", 1)}_ebtimeF_dtTrue_{config["des"]}_0'
        checkpoint_source = os.path.join("checkpoints", setting)
        checkpoint_dest = os.path.join(archive_dir, "checkpoints")
        
        # 复制checkpoints
        if os.path.exists(checkpoint_source):
            shutil.copytree(checkpoint_source, checkpoint_dest)
            print(f"Checkpoints copied from: {checkpoint_source}")
        else:
            print(f"Warning: Checkpoints not found: {checkpoint_source}")
        
        # 复制test_results
        test_results_source = os.path.join("test_results", setting)
        test_results_dest = os.path.join(archive_dir, "test_results")
        
        if os.path.exists(test_results_source):
            shutil.copytree(test_results_source, test_results_dest)
            print(f"Test results copied from: {test_results_source}")
        else:
            print(f"Warning: Test results not found: {test_results_source}")
    
    # 创建实验摘要
    summary = {
        "archive_name": archive_name,
        "model_id": model_id,
        "archive_time": datetime.datetime.now().isoformat(),
        "archive_dir": archive_dir,
        "description": config.get("experiment_description", "") if os.path.exists(config_path) else "",
        "key_parameters": {
            "seq_len": config.get("seq_len", "unknown"),
            "d_model": config.get("d_model", "unknown"),
            "n_heads": config.get("n_heads", "unknown"),
            "e_layers": config.get("e_layers", "unknown"),
            "train_epochs": config.get("train_epochs", "unknown"),
            "learning_rate": config.get("learning_rate", "unknown")
        } if os.path.exists(config_path) else {}
    }
    
    with open(os.path.join(archive_dir, "experiment_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Experiment archived to: {archive_dir}")
    print(f"Summary saved to: {os.path.join(archive_dir, 'experiment_summary.json')}")
    
    return archive_dir

def main():
    parser = argparse.ArgumentParser(description='TimeMixer Well Lifecycle Experiment Archiving')
    
    parser.add_argument('--model_id', type=str, required=True, help='Experiment ID to archive')
    parser.add_argument('--archive_name', type=str, default=None, help='Custom archive name')
    
    args = parser.parse_args()
    
    archive_experiment(args.model_id, args.archive_name)

if __name__ == "__main__":
    main()
