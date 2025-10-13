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
import csv

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from argparse import Namespace
import torch

class TrainingLogger:
    """训练日志记录器"""
    def __init__(self, log_dir, model_id):
        self.log_dir = log_dir
        self.model_id = model_id
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建日志文件路径
        self.loss_log_path = os.path.join(log_dir, 'training_loss.csv')
        self.summary_log_path = os.path.join(log_dir, 'training_summary.txt')
        
        # 初始化CSV日志
        with open(self.loss_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train_Loss', 'Vali_Loss', 'Test_Loss', 
                           'Learning_Rate', 'Time'])
        
        # 初始化文本日志
        with open(self.summary_log_path, 'w') as f:
            f.write(f"训练日志 - 模型ID: {model_id}\n")
            f.write(f"开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
    
    def log_epoch(self, epoch, train_loss, vali_loss, test_loss, lr, epoch_time):
        """记录每个epoch的损失"""
        # 写入CSV
        with open(self.loss_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f'{train_loss:.7f}', f'{vali_loss:.7f}', 
                           f'{test_loss:.7f}', f'{lr:.10f}', 
                           datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        
        # 写入文本日志
        with open(self.summary_log_path, 'a') as f:
            f.write(f"Epoch {epoch}:\n")
            f.write(f"  Train Loss: {train_loss:.7f}\n")
            f.write(f"  Vali Loss:  {vali_loss:.7f}\n")
            f.write(f"  Test Loss:  {test_loss:.7f}\n")
            f.write(f"  Learning Rate: {lr:.10f}\n")
            f.write(f"  Time: {epoch_time:.2f}s\n")
            f.write("-" * 80 + "\n")
    
    def log_final(self, best_epoch, best_vali_loss, total_time, stopped_early=False):
        """记录最终训练结果"""
        with open(self.summary_log_path, 'a') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("训练完成\n")
            f.write(f"结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总训练时间: {total_time:.2f}s ({total_time/60:.2f}分钟)\n")
            f.write(f"最佳Epoch: {best_epoch}\n")
            f.write(f"最佳验证损失: {best_vali_loss:.7f}\n")
            if stopped_early:
                f.write(f"早停触发: 是\n")
            else:
                f.write(f"早停触发: 否（完成所有epoch）\n")
            f.write("=" * 80 + "\n")

def train_sliding_window_model(args, logger=None):
    """训练滑动窗口模型（带日志记录）"""
    import time
    import numpy as np
    from utils.tools import EarlyStopping, adjust_learning_rate
    from torch.optim import lr_scheduler
    
    print(f"🚀 开始训练滑动窗口TimeMixer模型...")
    print(f"📊 配置参数:")
    print(f"   模型ID: {args.model_id}")
    print(f"   输入长度 (input_len): {args.seq_len}")
    print(f"   输出长度 (output_len): {args.pred_len}")
    print(f"   滑动步长 (step_len): {args.step_len}")
    print(f"   比例: {args.seq_len}:{args.pred_len} = {args.seq_len/args.pred_len:.2f}:1")
    print(f"   模型维度: {args.d_model}")
    print(f"   训练轮数: {args.train_epochs}")
    
    if logger:
        print(f"   日志保存: {logger.log_dir}")
    
    # 创建实验对象
    exp = Exp_Long_Term_Forecast(args)
    
    # 获取数据
    train_data, train_loader = exp._get_data(flag='train')
    vali_data, vali_loader = exp._get_data(flag='val')
    test_data, test_loader = exp._get_data(flag='test')
    
    # 创建checkpoint目录
    path = os.path.join(args.checkpoints, args.model_id)
    if not os.path.exists(path):
        os.makedirs(path)
    
    # 训练设置
    train_steps = len(train_loader)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    model_optim = exp._select_optimizer()
    criterion = exp._select_criterion()
    
    scheduler = lr_scheduler.OneCycleLR(
        optimizer=model_optim,
        steps_per_epoch=train_steps,
        pct_start=args.pct_start,
        epochs=args.train_epochs,
        max_lr=args.learning_rate
    )
    
    # 开始训练
    print(f"\n🔄 开始训练...")
    training_start_time = time.time()
    
    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []
        exp.model.train()
        epoch_start_time = time.time()
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            iter_count += 1
            model_optim.zero_grad()
            
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)
            
            # 准备decoder输入
            if args.down_sampling_layers == 0:
                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(exp.device)
            else:
                dec_inp = None
            
            # 前向传播
            outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            # 计算损失
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(exp.device)
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸导致NaN
            torch.nn.utils.clip_grad_norm_(exp.model.parameters(), max_norm=1.0)
            
            model_optim.step()
            
            # 调整学习率
            if args.lradj == 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, args, printout=False)
                scheduler.step()
            
            # 打印进度
            if (i + 1) % 100 == 0:
                print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                speed = (time.time() - epoch_start_time) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
        
        # Epoch结束
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch: {epoch + 1} cost time: {epoch_time:.2f}s")
        
        train_loss = np.average(train_loss)
        vali_loss = exp.vali(vali_data, vali_loader, criterion)
        test_loss = exp.vali(test_data, test_loader, criterion)
        
        current_lr = scheduler.get_last_lr()[0] if args.lradj == 'TST' else model_optim.param_groups[0]['lr']
        
        print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} "
              f"Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
        
        # 记录日志
        if logger:
            logger.log_epoch(epoch + 1, train_loss, vali_loss, test_loss, current_lr, epoch_time)
        
        # 早停检查
        early_stopping(vali_loss, exp.model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            if logger:
                logger.log_final(epoch + 1 - args.patience, early_stopping.val_loss_min, 
                               time.time() - training_start_time, stopped_early=True)
            break
        
        # 调整学习率
        if args.lradj != 'TST':
            adjust_learning_rate(model_optim, scheduler, epoch + 1, args, printout=True)
        else:
            print(f'Updating learning rate to {scheduler.get_last_lr()[0]}')
    
    # 训练完成
    total_training_time = time.time() - training_start_time
    
    if logger and not early_stopping.early_stop:
        logger.log_final(args.train_epochs, early_stopping.val_loss_min, 
                        total_training_time, stopped_early=False)
    
    # 加载最佳模型
    best_model_path = path + '/checkpoint.pth'
    exp.model.load_state_dict(torch.load(best_model_path))
    
    print(f"✅ 滑动窗口模型训练完成！")
    print(f"📁 模型保存在: checkpoints/{args.model_id}/")
    if logger:
        print(f"📝 训练日志保存在: {logger.log_dir}/")
    
    return exp.model

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
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout比例')
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
        dropout=args.dropout,
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
        loss='MAE',
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
    
    # 创建日志记录器
    log_dir = f"logs/{args.model_id}"
    logger = TrainingLogger(log_dir, args.model_id)
    print(f"📝 训练日志将保存到: {log_dir}/")
    
    # 将logger传递给model_args以便在训练中使用
    model_args.logger = logger
    model_args.log_dir = log_dir
    
    # 开始训练
    train_sliding_window_model(model_args, logger)

if __name__ == "__main__":
    main()

