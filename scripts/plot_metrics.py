#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TimeMixer Well Lifecycle Prediction - Metrics Visualization Script
井生命周期预测指标可视化脚本

Usage:
    python scripts/plot_metrics.py --results_dir results_archive/my_experiment
    python scripts/plot_metrics.py --csv_file results_archive/my_experiment/analysis/per_well_ratio_metrics_extended.csv
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def create_visualizations(csv_file, output_dir):
    """创建指标可视化图表"""
    # 读取数据
    df = pd.read_csv(csv_file)
    
    # 设置字体
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # 1) 分割比例维度柱状图：sMAPE / NRMSE_mean / R2
    ratio_agg = df.groupby('split_ratio').agg({'sMAPE_%':'mean','NRMSE_mean':'mean','R2':'mean'}).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].bar(ratio_agg['split_ratio'], ratio_agg['sMAPE_%'], color='#4C78A8')
    axes[0].set_title('sMAPE by Split Ratio')
    axes[0].set_xlabel('Split Ratio (%)')
    axes[0].set_ylabel('sMAPE (%)')
    
    axes[1].bar(ratio_agg['split_ratio'], ratio_agg['NRMSE_mean'], color='#F58518')
    axes[1].set_title('NRMSE(mean) by Split Ratio')
    axes[1].set_xlabel('Split Ratio (%)')
    axes[1].set_ylabel('NRMSE (mean-normalized)')
    
    axes[2].bar(ratio_agg['split_ratio'], ratio_agg['R2'], color='#54A24B')
    axes[2].set_title('R^2 by Split Ratio')
    axes[2].set_xlabel('Split Ratio (%)')
    axes[2].set_ylabel('R^2')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bars_extended_by_ratio.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2) 井×分割比例热力图：sMAPE / R2
    for metric, cmap in [('sMAPE_%','magma'), ('R2','viridis')]:
        pivot = df.pivot_table(index='well_idx', columns='split_ratio', values=metric, aggfunc='mean')
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(pivot.values, cmap=cmap, aspect='auto')
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel('Split Ratio (%)')
        ax.set_ylabel('Well Index')
        ax.set_title(f'{metric} Heatmap (Well × Split Ratio)')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'heatmap_{metric}.pdf'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3) 箱线图：按井 sMAPE 分布 / 按井 R2 分布
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    df.boxplot(column='sMAPE_%', by='well_idx', ax=axes[0])
    axes[0].set_title('sMAPE Distribution by Well')
    axes[0].set_xlabel('Well Index')
    axes[0].set_ylabel('sMAPE (%)')
    axes[0].figure.suptitle('')
    
    df.boxplot(column='R2', by='well_idx', ax=axes[1])
    axes[1].set_title('R^2 Distribution by Well')
    axes[1].set_xlabel('Well Index')
    axes[1].set_ylabel('R^2')
    axes[1].figure.suptitle('')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'box_extended_by_well.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4) 基础指标对比图
    ratio_agg_basic = df.groupby('split_ratio').agg({'mae':'mean','rmse':'mean','corr':'mean'}).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].bar(ratio_agg_basic['split_ratio'], ratio_agg_basic['mae'], color='#4C78A8')
    axes[0].set_title('MAE by Split Ratio')
    axes[0].set_xlabel('Split Ratio (%)')
    axes[0].set_ylabel('MAE')
    
    axes[1].bar(ratio_agg_basic['split_ratio'], ratio_agg_basic['rmse'], color='#F58518')
    axes[1].set_title('RMSE by Split Ratio')
    axes[1].set_xlabel('Split Ratio (%)')
    axes[1].set_ylabel('RMSE')
    
    axes[2].bar(ratio_agg_basic['split_ratio'], ratio_agg_basic['corr'], color='#E45756')
    axes[2].set_title('Correlation by Split Ratio')
    axes[2].set_xlabel('Split Ratio (%)')
    axes[2].set_ylabel('Correlation')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bars_basic_by_ratio.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='TimeMixer Well Lifecycle Metrics Visualization')
    
    # 输入选项
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--results_dir', type=str, help='Results directory containing analysis/ subdirectory')
    group.add_argument('--csv_file', type=str, help='Direct path to metrics CSV file')
    
    # 输出目录
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for plots (default: same as input)')
    
    args = parser.parse_args()
    
    # 确定CSV文件路径
    if args.results_dir:
        csv_file = os.path.join(args.results_dir, 'analysis', 'per_well_ratio_metrics_extended.csv')
        if not os.path.exists(csv_file):
            print(f"Error: CSV file not found: {csv_file}")
            return
        output_dir = args.output_dir or os.path.join(args.results_dir, 'analysis')
    else:
        csv_file = args.csv_file
        if not os.path.exists(csv_file):
            print(f"Error: CSV file not found: {csv_file}")
            return
        output_dir = args.output_dir or os.path.dirname(csv_file)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建可视化
    create_visualizations(csv_file, output_dir)

if __name__ == "__main__":
    main()
