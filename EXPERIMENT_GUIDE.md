# TimeMixer 井生命周期预测实验指南

## 📋 项目概述

基于TimeMixer的井生命周期预测模型，实现8:2比例的时序预测（80%历史数据作为输入，预测后续20%数据）。

- **模型**: TimeMixer (多尺度时序预测)
- **任务**: 油气井产量长期预测
- **输入**: 2400步历史产量数据 (80%)
- **输出**: 600步未来产量预测 (20%)
- **硬件加速**: 支持MPS (Apple Silicon) / CUDA

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建conda环境
conda create -n timemixer python=3.10
conda activate timemixer

# 安装依赖
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib scikit-learn scipy seaborn einops reformer-pytorch sympy
```

### 2. 数据准备

数据文件结构：
```
data/
└── preprocessed_daily_gas_by_well.csv  # 每列一口井的日产量序列
```

### 3. 训练模型

```bash
# 激活环境并进入项目目录
source /Users/wangjr/miniconda3/bin/activate timemixer
cd gas-timemix

# 训练8:2比例模型
python scripts/train_8_2_ratio.py \
  --model_id wellmix_8_2_full \
  --total_length 3000 \
  --input_ratio 0.8 \
  --output_ratio 0.2 \
  --train_epochs 100 \
  --batch_size 8 \
  --use_gpu \
  --root_path /path/to/your/data \
  --data_path preprocessed_daily_gas_by_well.csv \
  2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S).log
```

### 4. 测试和可视化

```bash
# 测试并生成可视化图表
python scripts/test_and_visualize.py \
  --model_id wellmix_8_2_full \
  --test_wells 0,1,2,3,4,5,6,7,8,9 \
  --ratios 60,70,80 \
  --output_dir results_archive/wellmix_8_2_full_no_smooth \
  2>&1 | tee logs/test_$(date +%Y%m%d_%H%M%S).log
```

## 📊 完整训练指令记录

### 训练参数

```bash
python scripts/train_8_2_ratio.py \
  --model_id wellmix_8_2_full \
  --total_length 3000 \            # 总序列长度
  --input_ratio 0.8 \               # 输入比例: 80%
  --output_ratio 0.2 \              # 输出比例: 20%
  --train_epochs 100 \              # 训练轮数
  --batch_size 8 \                  # 批次大小
  --use_gpu \                       # 启用GPU加速
  --root_path /path/to/data \       # 数据根目录
  --data_path preprocessed_daily_gas_by_well.csv
```

**参数说明：**
- `--model_id`: 模型标识符，用于保存checkpoint
- `--total_length 3000`: 总序列长度
  - 输入: 3000 × 0.8 = 2400步
  - 输出: 3000 × 0.2 = 600步
- `--train_epochs 100`: 训练100轮
- `--batch_size 8`: 每批次8个样本
- `--use_gpu`: 启用GPU (MPS/CUDA)，若不可用自动降级到CPU

### 测试参数

```bash
python scripts/test_and_visualize.py \
  --model_id wellmix_8_2_full \              # 对应训练的模型ID
  --test_wells 0,1,2,3,4,5,6,7,8,9 \         # 测试的井编号
  --ratios 60,70,80 \                        # 分割比例(%)
  --output_dir results_archive/wellmix_8_2_full_no_smooth
```

**参数说明：**
- `--test_wells`: 逗号分隔的井编号列表
- `--ratios`: 分割比例，如70表示用前70%数据作为输入，预测后续数据
- `--output_dir`: 结果输出目录

## 📈 实验结果

### 训练结果

- **最终训练损失**: 0.0024
- **最终验证损失**: 0.0029 (从初始1.25降至0.0029，降低99.8%)
- **最终测试损失**: 0.7279
- **训练时长**: ~33分钟 (100 epochs，使用MPS)
- **单epoch耗时**: ~20秒

### 测试结果 (10口井 × 3个分割比例)

| 指标 | 数值 |
|------|------|
| 总样本数 | 30 |
| 平均MAE | 17,869.85 |
| 平均RMSE | 21,271.07 |
| 平均MAPE | 122.01% |
| 平均Jump | -5,459.52 |

### 生成的文件

```
gas-timemix/
├── logs/
│   ├── train_full_20250930_083629.log        # 训练日志
│   └── test_full_all_20250930_092946.log     # 测试日志
├── checkpoints/
│   └── wellmix_8_2_full/
│       └── checkpoint.pth                     # 模型权重
├── experiments/
│   └── wellmix_8_2_full/
│       └── config.json                        # 实验配置
└── results_archive/
    └── wellmix_8_2_full_no_smooth/
        ├── detailed_results_no_smooth.csv    # 详细结果
        ├── by_well_summary_no_smooth.csv     # 按井汇总
        ├── overall_summary_no_smooth.csv     # 整体汇总
        ├── well_0_ratio_60_no_smooth.pdf     # 可视化图表
        ├── well_0_ratio_60_no_smooth.csv     # 预测数据
        └── ... (共30个PDF + 30个CSV)
```

## 🎨 可视化说明

所有生成的图表使用统一的四色方案：

- 🟣 **紫色**: 早期历史数据（输入段之前）
- 🔵 **蓝色**: 输入段（模型使用的历史数据）
- 🟢 **绿色**: 真实输出段（实际产量）
- 🟠 **橙色**: 预测输出段（模型预测）

**标记线：**
- 🔴 **红色虚线**: 预测起点
- 🔵 **蓝色点线**: 输入起点

**图表特征：**
- 横坐标从0开始，显示完整井生命周期
- 包含所有历史数据，无空白段
- 图例清晰标注各段数据类型

## 🔍 监控训练进度

```bash
# 实时查看训练日志
tail -f logs/train_full_20250930_083629.log

# 查看最新训练epoch
grep "Epoch:" logs/train_full_20250930_083629.log | tail -5

# 检查训练进程是否运行
ps aux | grep "train_8_2_ratio.py" | grep -v grep
```

## 📊 查看结果

```bash
# 查看整体汇总
cat results_archive/wellmix_8_2_full_no_smooth/overall_summary_no_smooth.csv

# 查看按井汇总
cat results_archive/wellmix_8_2_full_no_smooth/by_well_summary_no_smooth.csv

# 打开可视化图表 (macOS)
open results_archive/wellmix_8_2_full_no_smooth/well_0_ratio_70_no_smooth.pdf

# 列出所有结果文件
ls -lh results_archive/wellmix_8_2_full_no_smooth/
```

## 🛠️ 一键运行脚本

创建 `run_full_experiment.sh`:

```bash
#!/bin/bash
# 完整实验流程脚本

# 激活环境
source /Users/wangjr/miniconda3/bin/activate timemixer
cd gas-timemix

# 步骤1: 训练模型
echo "🚀 Step 1: Training model..."
python scripts/train_8_2_ratio.py \
  --model_id wellmix_8_2_production \
  --total_length 3000 \
  --input_ratio 0.8 \
  --output_ratio 0.2 \
  --train_epochs 100 \
  --batch_size 8 \
  --use_gpu \
  --root_path /path/to/your/data \
  --data_path preprocessed_daily_gas_by_well.csv \
  2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S).log

# 步骤2: 测试和可视化
echo "🧪 Step 2: Testing and visualization..."
python scripts/test_and_visualize.py \
  --model_id wellmix_8_2_production \
  --test_wells 0,1,2,3,4,5,6,7,8,9 \
  --ratios 60,70,80 \
  --output_dir results_archive/wellmix_8_2_production_no_smooth \
  2>&1 | tee logs/test_$(date +%Y%m%d_%H%M%S).log

echo "✅ Experiment completed!"
echo "📊 Results saved to: results_archive/wellmix_8_2_production_no_smooth/"
```

使用方法：
```bash
chmod +x run_full_experiment.sh
./run_full_experiment.sh
```

## 🔧 故障排查

### 问题1: CUDA/MPS不可用
```bash
# 解决方案：使用CPU训练
python scripts/train_8_2_ratio.py \
  ... \
  # 不加 --use_gpu 参数
```

### 问题2: 内存不足
```bash
# 解决方案：减小批次大小
python scripts/train_8_2_ratio.py \
  ... \
  --batch_size 4  # 从8减到4
```

### 问题3: 数据长度不足
某些井的早期分割点（10%-50%）可能因为数据长度不足而被跳过。这是正常的，模型会自动跳过这些样本。

**建议使用的分割比例：** 60%, 70%, 80%

## 📝 实验配置文件

训练完成后会自动生成配置文件：`experiments/wellmix_8_2_full/config.json`

```json
{
  "model_id": "wellmix_8_2_full",
  "total_length": 3000,
  "input_ratio": 0.8,
  "output_ratio": 0.2,
  "seq_len": 2400,
  "pred_len": 600,
  "d_model": 256,
  "n_heads": 16,
  "e_layers": 6,
  "d_layers": 3,
  "d_ff": 1024,
  "train_epochs": 100,
  "batch_size": 8,
  "learning_rate": 0.0001,
  "use_gpu": true,
  "created_at": "2025-09-30T08:36:29"
}
```

## 📖 参考文献

- 原始项目: [gas-timemix](https://github.com/w01ang/gas-timemix)
- TimeMixer论文: [TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting](https://arxiv.org/abs/2405.14616)

## 📧 联系方式

如有问题，请通过GitHub Issues联系。

---

**版本**: v1.0  
**更新时间**: 2025-09-30  
**实验日期**: 2025-09-30
