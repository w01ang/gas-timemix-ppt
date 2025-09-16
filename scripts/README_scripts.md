# TimeMixer 实验管理脚本

## 📁 脚本目录结构

```
scripts/
├── train_experiment.py          # 模型训练脚本
├── test_and_visualize.py        # 测试和可视化脚本（无平滑过渡）
├── plot_metrics.py              # 指标可视化脚本
├── archive_experiment.py        # 实验归档脚本
├── run_full_experiment.py       # 完整实验流程脚本
└── README_scripts.md           # 本说明文件
```

## 🚀 快速开始

### 1. 完整实验流程（推荐）

```bash
# 激活环境
conda activate timemixer

# 运行完整实验
python scripts/run_full_experiment.py \
    --model_id my_experiment \
    --test_wells 0,1,2,3,4,5,6,7,8,9 \
    --ratios 10,20,30,40,50,60,70,80,90 \
    --seq_len 3000 \
    --pred_len 256 \
    --d_model 256 \
    --train_epochs 100
```

### 2. 分步执行

#### 步骤1：训练模型
```bash
python scripts/train_experiment.py \
    --model_id my_experiment \
    --comment "my_training" \
    --description "My experiment description" \
    --seq_len 3000 \
    --pred_len 256 \
    --d_model 256 \
    --train_epochs 100
```

#### 步骤2：测试和可视化
```bash
python scripts/test_and_visualize.py \
    --model_id my_experiment \
    --test_wells 0,1,2,3,4 \
    --ratios 50,60,70,80,90
```

#### 步骤3：指标可视化
```bash
python scripts/plot_metrics.py \
    --results_dir results_archive/my_experiment_no_smooth
```

#### 步骤4：归档结果
```bash
python scripts/archive_experiment.py \
    --model_id my_experiment_no_smooth \
    --archive_name my_experiment_$(date +%Y%m%d_%H%M%S)
```

## 📋 脚本详细说明

### 1. `train_experiment.py` - 模型训练
**功能：** 训练TimeMixer模型并保存检查点

**主要参数：**
- `--model_id`: 实验ID（必需）
- `--seq_len`: 输入序列长度（默认：3000）
- `--pred_len`: 预测长度（默认：256）
- `--d_model`: 模型维度（默认：256）
- `--train_epochs`: 训练轮数（默认：100）
- `--batch_size`: 批次大小（默认：8）
- `--learning_rate`: 学习率（默认：1e-4）

### 2. `test_and_visualize.py` - 测试和可视化
**功能：** 测试模型并生成增强可视化图表（无平滑过渡）

**主要参数：**
- `--model_id`: 实验ID（必需）
- `--test_wells`: 测试井索引，逗号分隔（默认：0,1,2,3,4,5,6,7,8,9）
- `--ratios`: 分割比例，逗号分隔（默认：10,20,30,40,50,60,70,80,90）
- `--output_dir`: 输出目录（可选）

**输出：**
- PDF图表：`well_{well_idx}_ratio_{ratio}_no_smooth.pdf`
- CSV数据：`well_{well_idx}_ratio_{ratio}_no_smooth.csv`
- 汇总结果：`detailed_results_no_smooth.csv`

### 3. `plot_metrics.py` - 指标可视化
**功能：** 生成各种评估指标的可视化图表

**主要参数：**
- `--results_dir`: 结果目录路径（必需）
- `--output_dir`: 输出目录（可选）

**输出：**
- 指标对比图：`metrics_comparison.pdf`
- 箱线图：`metrics_boxplot.pdf`
- 热力图：`metrics_heatmap.pdf`

### 4. `archive_experiment.py` - 实验归档
**功能：** 将实验相关文件归档到指定目录

**主要参数：**
- `--model_id`: 实验ID（必需）
- `--archive_name`: 归档名称（可选，默认：{model_id}_{timestamp}）

### 5. `run_full_experiment.py` - 完整流程
**功能：** 一键运行完整的实验流程

**主要参数：**
- `--model_id`: 实验ID（必需）
- `--test_wells`: 测试井索引
- `--ratios`: 分割比例
- `--skip_training`: 跳过训练步骤
- `--skip_testing`: 跳过测试步骤
- `--skip_plotting`: 跳过绘图步骤
- `--skip_archiving`: 跳过归档步骤

## 🔧 重要特性

### 无平滑过渡设计
- ✅ 预测值直接从模型输出开始
- ✅ 无人工平滑干预
- ✅ 保持模型原始预测能力
- ✅ 代码更简洁

### 动态输入长度
- ✅ 根据分割比例动态调整输入长度
- ✅ 最大输入长度：3000步
- ✅ 充分利用历史数据

### 多井多比例测试
- ✅ 支持多口井同时测试
- ✅ 支持多种分割比例（10%-90%）
- ✅ 自动生成对比分析

### 增强可视化
- ✅ 4色图表：早期历史（紫）、输入段（蓝）、真实输出（绿）、预测输出（橙）
- ✅ 完整生命周期展示
- ✅ 统计信息标注
- ✅ Times New Roman字体

## 📊 输出结果

### 目录结构
```
results_archive/
└── {model_id}_no_smooth/
    ├── detailed_results_no_smooth.csv      # 详细结果
    ├── by_well_summary_no_smooth.csv       # 按井汇总
    ├── overall_summary_no_smooth.csv       # 整体汇总
    ├── well_0_ratio_50_no_smooth.pdf       # 井0-50%分割图表
    ├── well_0_ratio_50_no_smooth.csv       # 井0-50%分割数据
    └── ...                                  # 其他井和比例的结果
```

### 评估指标
- **MAE**: 平均绝对误差
- **RMSE**: 均方根误差
- **MAPE**: 平均绝对百分比误差
- **跳跃大小**: 预测起始值与输入末尾值的差值

## 🛠️ 环境要求

```bash
# 激活环境
conda activate timemixer

# 检查依赖
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import matplotlib; print('Matplotlib:', matplotlib.__version__)"
python -c "import pandas; print('Pandas:', pandas.__version__)"
```

## 📝 使用示例

### 示例1：快速测试
```bash
conda activate timemixer
python scripts/run_full_experiment.py --model_id quick_test --test_wells 0,1,2 --ratios 50,60,70
```

### 示例2：完整实验
```bash
conda activate timemixer
python scripts/run_full_experiment.py \
    --model_id full_experiment \
    --test_wells 0,1,2,3,4,5,6,7,8,9 \
    --ratios 10,20,30,40,50,60,70,80,90 \
    --seq_len 3000 \
    --pred_len 256 \
    --d_model 256 \
    --n_heads 16 \
    --e_layers 6 \
    --d_layers 3 \
    --d_ff 1024 \
    --train_epochs 100 \
    --batch_size 8 \
    --learning_rate 1e-4
```

### 示例3：仅测试现有模型
```bash
conda activate timemixer
python scripts/test_and_visualize.py \
    --model_id existing_model \
    --test_wells 0,1,2,3,4 \
    --ratios 50,60,70,80,90
```

## 🔍 故障排除

### 常见问题

1. **PyTorch环境问题**
   ```bash
   conda activate timemixer
   python -c "import torch; print(torch.__version__)"
   ```

2. **模型文件不存在**
   - 确保先运行训练脚本
   - 检查checkpoints目录

3. **数据文件不存在**
   - 确保数据文件在正确路径
   - 检查数据文件格式

4. **内存不足**
   - 减少batch_size
   - 减少seq_len
   - 减少测试井数量

## 📞 支持

如有问题，请检查：
1. 环境是否正确激活
2. 数据文件是否存在
3. 模型是否已训练
4. 参数设置是否合理

---
**更新时间：** 2025-09-16  
**版本：** v2.0 (无平滑过渡版本)
