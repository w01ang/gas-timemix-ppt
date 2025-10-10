# 滑动窗口训练与测试指南

## 📋 概述

本指南介绍如何使用滑动窗口方法训练和测试TimeMixer模型。与之前的多比例分割方法不同，滑动窗口方法具有以下特点：

### ✨ 核心改进

1. **固定长度输入输出**：
   - `input_len`: 输入序列长度（固定）
   - `output_len`: 输出序列长度（固定）
   - 所有样本的输入输出长度完全一致，无填充，无截断

2. **滑动窗口采样**：
   - 使用固定步长 `step_len` 在每口井上滑动采样
   - 默认步长 = output_len（无重叠）
   - 可设置较小步长增加样本数量（有重叠）

3. **统一的训练测试策略**：
   - 训练集、验证集、测试集使用相同的采样方法
   - 井划分: 70% 训练，10% 验证，20% 测试
   - 每口井内部使用滑动窗口生成多个样本

## 🚀 快速开始

### 1. 训练模型

```bash
python scripts/train_sliding_window.py \
  --model_id wellmix_sliding_640_160 \
  --input_len 640 \
  --output_len 160 \
  --step_len 160 \
  --train_epochs 100 \
  --batch_size 8 \
  --learning_rate 1e-4 \
  --use_gpu
```

**参数说明**：
- `--model_id`: 实验唯一标识
- `--input_len`: 输入序列长度（例如：640步）
- `--output_len`: 输出序列长度（例如：160步，比例 4:1）
- `--step_len`: 滑动步长（默认=output_len，无重叠；可设为更小值增加样本）
- `--train_epochs`: 训练轮数
- `--batch_size`: 批大小
- `--learning_rate`: 学习率
- `--use_gpu`: 启用GPU加速（MPS/CUDA）

### 2. 测试与可视化

```bash
python scripts/test_sliding_window.py \
  --model_id wellmix_sliding_640_160 \
  --max_wells 5
```

**参数说明**：
- `--model_id`: 要测试的模型ID（与训练时一致）
- `--max_wells`: 最多测试多少口井（默认：全部）
- `--output_dir`: 输出目录（默认：test_results/<model_id>）

## 📊 数据采样机制

### 滑动窗口示例

假设一口井长度为 4000 步，使用 `input_len=640`, `output_len=160`, `step_len=160`：

```
井序列: [0, 1, 2, ..., 3999]  (总长度 4000)

窗口1: input=[0:640]     output=[640:800]    (起始位置: 0)
窗口2: input=[160:800]   output=[800:960]    (起始位置: 160)
窗口3: input=[320:960]   output=[960:1120]   (起始位置: 320)
...
窗口N: input=[3680:4320] (超出) - 丢弃

总共生成约 24 个有效窗口
```

### 不同步长的影响

| step_len | 窗口重叠 | 样本数量 | 训练时间 | 适用场景 |
|----------|---------|---------|---------|---------|
| output_len (160) | 无重叠 | 基准 | 较快 | 标准训练 |
| output_len/2 (80) | 50%重叠 | 2倍 | 较慢 | 数据增强 |
| output_len*2 (320) | 有间隙 | 0.5倍 | 很快 | 快速实验 |

## 🔧 参数调优建议

### 1. 输入输出比例选择

| 比例 | input_len | output_len | 适用场景 |
|------|-----------|-----------|---------|
| 2:1  | 640       | 320       | 短期预测 |
| 4:1  | 640       | 160       | **推荐** |
| 8:1  | 640       | 80        | 长期预测 |

### 2. 井长度要求

确保大部分井的长度 ≥ `input_len + output_len`，否则无法生成有效窗口。

查看数据集统计：
```python
import pandas as pd
df = pd.read_csv('data/preprocessed_daily_gas_by_well.csv')
lengths = df.count()
print(f"最小长度: {lengths.min()}")
print(f"平均长度: {lengths.mean():.0f}")
print(f"最大长度: {lengths.max()}")
```

### 3. 模型容量设置

| 数据规模 | d_model | n_heads | e_layers | d_ff |
|---------|---------|---------|---------|------|
| 小 (<50井) | 64 | 4 | 2 | 256 |
| 中 (50-200井) | 128 | 8 | 3 | 512 |
| 大 (>200井) | 256 | 16 | 6 | 1024 |

## 📁 文件结构

训练后会生成以下文件：

```
gas-timemix/
├── checkpoints/
│   └── wellmix_sliding_640_160/
│       ├── checkpoint.pth          # 模型权重
│       └── training_log.txt        # 训练日志
├── experiments/
│   └── wellmix_sliding_640_160/
│       └── config.json             # 实验配置
└── test_results/
    └── wellmix_sliding_640_160/
        ├── results.csv             # 详细结果
        ├── well_statistics.csv     # 按井统计
        └── well_*.pdf              # 可视化图片
```

## 🎨 可视化说明

测试脚本生成的图表包含：

- **浅灰色线**: 完整井序列
- **蓝色段**: 输入序列（每个窗口）
- **绿色实线**: 真实输出
- **橙色虚线**: 预测输出
- **红色虚线**: 预测起点标记

每口井会生成一张包含所有滑动窗口的预测可视化图。

## 🆚 与旧方法的对比

| 特性 | 旧方法 (多比例分割) | 新方法 (滑动窗口) |
|------|-------------------|------------------|
| 输入长度 | 动态（填充/截断） | 固定（无填充） |
| 采样方式 | 10%-90%分割点 | 滑动窗口 |
| 样本数量 | 每井9个 | 每井N个（取决于step_len） |
| 训练测试一致性 | 不完全一致 | **完全一致** |
| 参数含义 | 混淆 | **清晰明确** |

## 💡 最佳实践

1. **首次实验**: 使用默认参数快速验证
   ```bash
   python scripts/train_sliding_window.py \
     --model_id test_640_160 \
     --input_len 640 \
     --output_len 160 \
     --train_epochs 10 \
     --use_gpu
   ```

2. **正式训练**: 增加训练轮数和模型容量
   ```bash
   python scripts/train_sliding_window.py \
     --model_id prod_640_160 \
     --input_len 640 \
     --output_len 160 \
     --train_epochs 100 \
     --d_model 256 \
     --n_heads 16 \
     --e_layers 6 \
     --use_gpu
   ```

3. **数据增强**: 使用重叠窗口
   ```bash
   python scripts/train_sliding_window.py \
     --model_id augmented_640_160 \
     --input_len 640 \
     --output_len 160 \
     --step_len 80 \  # 50%重叠
     --train_epochs 50 \
     --use_gpu
   ```

## ❓ 常见问题

### Q1: 如何选择合适的input_len和output_len？

A: 根据井的平均长度决定。建议 `input_len + output_len` ≤ 平均井长度的 50%，以确保每口井能生成足够多的窗口。

### Q2: step_len应该设置多大？

A: 
- 初始实验: `step_len = output_len`（无重叠，训练最快）
- 数据不足: `step_len = output_len / 2`（50%重叠，增加样本）
- 数据充足: `step_len = output_len * 2`（有间隙，减少冗余）

### Q3: 训练很慢怎么办？

A:
1. 减小batch_size（如果内存不足）
2. 增大step_len（减少样本数）
3. 减小模型容量（d_model, n_heads, e_layers）
4. 启用GPU加速（--use_gpu）

### Q4: 如何继续之前的训练？

A: 目前不支持断点续训，建议增加patience参数（早停耐心值）避免过拟合。

## 📞 技术支持

如遇问题，请检查：
1. 数据路径是否正确（--root_path, --data_path）
2. 井长度是否满足最小要求
3. GPU是否可用（检查torch.backends.mps.is_available()）
4. 训练日志中的错误信息

