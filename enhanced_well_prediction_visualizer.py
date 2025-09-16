#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Well Lifecycle Prediction Visualizer
增强井生命周期预测可视化工具

基于TimeMixer模型的井产量预测可视化方案，支持：
1. 三色曲线图：输入段(蓝) + 真实输出段(绿) + 预测输出段(橙)
2. 更长步长预测：减少平滑度，更好捕捉真实值波动
3. 迭代多块预测：扩展预测区间
4. 数据导出：CSV和PDF格式

Author: AI Assistant
Date: 2024
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Optional

# 设置matplotlib中文字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

class EnhancedWellPredictionVisualizer:
    """增强井生命周期预测可视化器"""
    
    def __init__(self, model, device, args, test_data):
        """
        初始化可视化器
        
        Args:
            model: 训练好的TimeMixer模型
            device: 计算设备
            args: 模型参数
            test_data: 测试数据集
        """
        self.model = model
        self.device = device
        self.args = args
        self.test_data = test_data
        
    def enhanced_test_visualization(self, folder_path: str, one_pred: np.ndarray, one_true: np.ndarray) -> None:
        """
        执行增强的三色可视化
        
        Args:
            folder_path: 输出文件夹路径
            one_pred: 单井预测结果
            one_true: 单井真实值
        """
        try:
            # 获取完整序列
            full_series = self.test_data.well_series[0]
            half_len = len(full_series) // 2
            total_len = len(full_series)
            
            # 使用更长的步长（3倍pred_len）
            step_len = min(self.args.pred_len * 3, 384)
            
            print(f"开始增强可视化：步长={step_len}, 总长度={total_len}")
            
            # 准备初始编码窗口
            ctx = full_series[max(0, half_len - self.args.seq_len):half_len]
            if ctx.shape[0] < self.args.seq_len:
                pad = np.zeros((self.args.seq_len - ctx.shape[0],), dtype=np.float32)
                ctx = np.concatenate([pad, ctx], axis=0)
            
            # 标准化
            ctx_scaled = (ctx - self.test_data.scaler.mean_[0]) / self.test_data.scaler.scale_[0]
            window = ctx_scaled.reshape(-1, 1).astype(np.float32)
            
            # 迭代预测
            extended_pred = []
            remain = total_len - half_len
            step_count = 0
            
            while remain > 0:
                current_step = min(step_len, remain)
                step_count += 1
                print(f"  预测步骤 {step_count}: 当前步长={current_step}, 剩余={remain}")
                
                # 调整输入长度
                if window.shape[0] < self.args.seq_len:
                    pad_len = self.args.seq_len - window.shape[0]
                    window = np.concatenate([np.zeros((pad_len, 1)), window], axis=0)
                
                x_enc = torch.from_numpy(window[-self.args.seq_len:]).unsqueeze(0).float().to(self.device)
                x_mark_enc = torch.zeros((1, x_enc.shape[1], 3), dtype=torch.float32).to(self.device)
                y_mark = torch.zeros((1, self.args.label_len + current_step, 3), dtype=torch.float32).to(self.device)
                
                if self.args.down_sampling_layers == 0:
                    dec_inp = torch.zeros((1, self.args.label_len + current_step, 1), dtype=torch.float32).to(self.device)
                else:
                    dec_inp = None
                
                with torch.no_grad():
                    out = self.model(x_enc, x_mark_enc, dec_inp, y_mark)
                    if isinstance(out, tuple):
                        out = out[0]
                    out = out[:, -current_step:, :]
                    out_np = out.detach().cpu().numpy()[0, :, 0]
                
                # 反标准化
                out_inv = out_np * self.test_data.scaler.scale_[0] + self.test_data.scaler.mean_[0]
                extended_pred.append(out_inv)
                
                # 更新窗口
                out_scaled = out_np.reshape(-1, 1).astype(np.float32)
                window = np.concatenate([window[current_step:], out_scaled], axis=0)
                remain -= current_step
            
            extended_pred = np.concatenate(extended_pred, axis=0)
            extended_pred = extended_pred[:(total_len - half_len)]
            
            print(f"  预测完成：总预测长度={len(extended_pred)}")
            
            # 创建三色对比数据
            input_segment = full_series[:half_len]
            true_output_segment = full_series[half_len:half_len + len(extended_pred)]
            pred_output_segment = extended_pred
            
            # 确保长度一致
            min_len = min(len(true_output_segment), len(pred_output_segment))
            true_output_segment = true_output_segment[:min_len]
            pred_output_segment = pred_output_segment[:min_len]
            
            # 绘制三色对比图
            self._create_three_color_plot(
                input_segment, true_output_segment, pred_output_segment, 
                half_len, total_len, folder_path
            )
            
            # 保存CSV数据
            self._save_three_color_csv(
                input_segment, true_output_segment, pred_output_segment,
                half_len, total_len, folder_path
            )
            
            print(f"增强可视化完成：步长={step_len}, 总预测长度={len(extended_pred)}")
            
        except Exception as e:
            print(f'增强可视化失败: {e}')
            import traceback
            traceback.print_exc()
    
    def _create_three_color_plot(self, input_segment: np.ndarray, true_output_segment: np.ndarray, 
                                pred_output_segment: np.ndarray, half_len: int, total_len: int, 
                                folder_path: str) -> None:
        """创建三色对比图"""
        plt.figure(figsize=(15, 8))
        x = np.arange(total_len)
        
        # 输入段（蓝色）
        plt.plot(x[:half_len], input_segment, 'b-', linewidth=2.5, 
                label='Input Segment (Historical)', alpha=0.8)
        
        # 真实输出段（绿色）
        plt.plot(x[half_len:half_len + len(true_output_segment)], true_output_segment, 
                'g-', linewidth=2.5, label='True Output (Ground Truth)', alpha=0.8)
        
        # 预测输出段（橙色）
        plt.plot(x[half_len:half_len + len(pred_output_segment)], pred_output_segment, 
                'orange', linewidth=2.5, label='Predicted Output (Model)', alpha=0.8)
        
        # 添加分割线
        plt.axvline(x=half_len, color='red', linestyle='--', alpha=0.7, linewidth=2)
        plt.text(half_len + total_len*0.01, max(input_segment)*0.9, 'Prediction Start', 
                rotation=90, fontsize=10, color='red', alpha=0.8)
        
        plt.legend(fontsize=12)
        plt.title('Well Lifecycle Prediction: Input (Blue) + True Output (Green) + Predicted Output (Orange)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Time Steps (Days)', fontsize=12)
        plt.ylabel('Gas Production', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图片
        output_path = os.path.join(folder_path, 'one_well_enhanced_3color.pdf')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  三色图已保存: {output_path}")
    
    def _save_three_color_csv(self, input_segment: np.ndarray, true_output_segment: np.ndarray, 
                             pred_output_segment: np.ndarray, half_len: int, total_len: int, 
                             folder_path: str) -> None:
        """保存三色数据CSV"""
        df_enhanced = {
            'input_segment': np.concatenate([input_segment, np.full((total_len - half_len,), np.nan)]),
            'true_output': np.concatenate([np.full((half_len,), np.nan), 
                                         np.concatenate([true_output_segment, 
                                                        np.full((total_len - half_len - len(true_output_segment),), np.nan)])]),
            'pred_output': np.concatenate([np.full((half_len,), np.nan), 
                                         np.concatenate([pred_output_segment, 
                                                        np.full((total_len - half_len - len(pred_output_segment),), np.nan)])])
        }
        
        output_path = os.path.join(folder_path, 'one_well_enhanced_3color.csv')
        pd.DataFrame(df_enhanced).to_csv(output_path, index=False)
        print(f"  三色数据已保存: {output_path}")


def create_enhanced_visualizer(model, device, args, test_data) -> EnhancedWellPredictionVisualizer:
    """
    创建增强可视化器的便捷函数
    
    Args:
        model: 训练好的TimeMixer模型
        device: 计算设备
        args: 模型参数
        test_data: 测试数据集
        
    Returns:
        EnhancedWellPredictionVisualizer实例
    """
    return EnhancedWellPredictionVisualizer(model, device, args, test_data)


# 使用示例
if __name__ == "__main__":
    print("Enhanced Well Prediction Visualizer")
    print("增强井生命周期预测可视化工具")
    print("=" * 50)
    print("使用方法：")
    print("1. 导入: from enhanced_well_prediction_visualizer import create_enhanced_visualizer")
    print("2. 创建: visualizer = create_enhanced_visualizer(model, device, args, test_data)")
    print("3. 调用: visualizer.enhanced_test_visualization(folder_path, one_pred, one_true)")
    print("=" * 50)
