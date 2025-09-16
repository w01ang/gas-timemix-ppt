# TimeMixer 井生命周期预测实验管理系统

## 📁 目录结构

```
TimeMixer/
├── scripts/                    # 实验脚本
│   ├── train_experiment.py     # 训练脚本
│   ├── test_and_visualize.py   # 测试与可视化脚本
│   ├── plot_metrics.py         # 指标可视化脚本
│   ├── archive_experiment.py   # 实验归档脚本
│   └── run_full_experiment.py  # 完整实验流程脚本
├── experiments/                 # 实验配置文件
│   └── {model_id}_config.json  # 各实验的配置
├── results_archive/            # 实验结果归档
│   └── {model_id}/            # 各实验的完整结果
│       ├── checkpoints/        # 模型检查点
│       ├── test_results/       # 测试结果
│       ├── analysis/           # 指标分析
│       └── experiment_summary.json  # 实验摘要
├── checkpoints/               # 当前训练检查点
└── test_results/              # 当前测试结果
```

## 🚀 快速开始

### 1. 完整实验流程（推荐）

```bash
# 运行完整实验（训练+测试+可视化+归档）
python scripts/run_full_experiment.py \
  --model_id my_experiment_v1 \
  --seq_len 3000 \
  --d_model 256 \
  --train_epochs 100 \
  --description "测试3000序列长度和256模型维度"
```

### 2. 分步执行

#### 训练模型
```bash
python scripts/train_experiment.py \
  --model_id my_experiment \
  --seq_len 3000 \
  --d_model 256 \
  --n_heads 16 \
  --e_layers 6 \
  --train_epochs 100 \
  --learning_rate 1e-4 \
  --description "测试新参数组合"
```

#### 测试和可视化
```bash
python scripts/test_and_visualize.py \
  --model_id my_experiment \
  --test_wells 0,1,2,3,4,5,6,7,8,9 \
  --ratios 10,20,30,40,50,60,70,80,90 \
  --transition_steps 20
```

#### 指标可视化
```bash
python scripts/plot_metrics.py \
  --results_dir results_archive/my_experiment
```

#### 归档实验
```bash
python scripts/archive_experiment.py \
  --model_id my_experiment \
  --archive_name my_experiment_v1
```

## 📊 输出结果

### 训练输出
- **配置文件**: `experiments/{model_id}_config.json`
- **模型检查点**: `checkpoints/{setting}/checkpoint.pth`
- **训练日志**: 终端实时输出

### 测试输出
- **预测图表**: `results_archive/{model_id}/multi_split_ratio_well{X}/well{X}_split{YY}_yellow.pdf`
- **预测数据**: `results_archive/{model_id}/multi_split_ratio_well{X}/well{X}_split{YY}_yellow.csv`
- **指标汇总**: `results_archive/{model_id}/analysis/per_well_ratio_metrics_extended.csv`

### 可视化输出
- **柱状图**: `bars_extended_by_ratio.pdf`, `bars_basic_by_ratio.pdf`
- **热力图**: `heatmap_sMAPE_%.pdf`, `heatmap_R2.pdf`
- **箱线图**: `box_extended_by_well.pdf`

## 🔧 参数调优指南

### 关键参数

| 参数 | 默认值 | 建议范围 | 说明 |
|------|--------|----------|------|
| `seq_len` | 3000 | 512-3000 | 输入序列长度 |
| `d_model` | 256 | 128-512 | 模型维度 |
| `n_heads` | 16 | 8-32 | 注意力头数 |
| `e_layers` | 6 | 2-8 | 编码器层数 |
| `d_layers` | 3 | 1-4 | 解码器层数 |
| `d_ff` | 1024 | 512-2048 | 前馈网络维度 |
| `learning_rate` | 1e-4 | 1e-5 to 1e-3 | 学习率 |
| `batch_size` | 8 | 4-32 | 批大小 |
| `train_epochs` | 100 | 50-200 | 训练轮数 |

### 调参建议

1. **序列长度**: 从512开始，逐步增加到3000
2. **模型容量**: 先调`d_model`，再调`n_heads`和层数
3. **学习率**: 从1e-4开始，根据收敛情况调整
4. **批大小**: 根据GPU内存调整，通常8-16效果较好

## 📈 评估指标

### 基础指标
- **MAE**: 平均绝对误差
- **RMSE**: 均方根误差
- **相关系数**: 预测与真实值的相关性

### 扩展指标
- **sMAPE**: 对称平均绝对百分比误差
- **NRMSE**: 归一化均方根误差（按均值或极差）
- **R²**: 决定系数
- **MdAE**: 中位数绝对误差
- **MBE**: 平均偏差误差
- **MAAPE**: 平均反正切绝对百分比误差

## 🎯 实验管理

### 命名规范
- **model_id**: `{项目}_{版本}_{日期}` (如: `wellmix_v2_20241215`)
- **comment**: 简短描述 (如: `dynamic_input`)
- **description**: 详细说明 (如: `测试3000序列长度和256模型维度`)

### 历史记录
- 每次实验自动生成时间戳
- 配置文件包含完整参数记录
- 归档目录包含所有相关文件
- 实验摘要提供快速概览

### 对比分析
```bash
# 查看所有实验
ls results_archive/

# 比较不同实验的指标
python scripts/plot_metrics.py --csv_file results_archive/exp1/analysis/per_well_ratio_metrics_extended.csv
python scripts/plot_metrics.py --csv_file results_archive/exp2/analysis/per_well_ratio_metrics_extended.csv
```

## 🔍 故障排除

### 常见问题

1. **内存不足**
   - 减小`batch_size`
   - 减小`seq_len`
   - 减小`d_model`

2. **训练不收敛**
   - 降低`learning_rate`
   - 增加`patience`
   - 检查数据预处理

3. **预测跳跃**
   - 调整`transition_steps`（建议10-20）
   - 检查数据分割逻辑

### 日志查看
- 训练日志：终端实时输出
- 配置文件：`experiments/{model_id}_config.json`
- 实验摘要：`results_archive/{model_id}/experiment_summary.json`

## 📞 支持

如有问题，请检查：
1. 配置文件是否正确
2. 数据路径是否存在
3. 模型检查点是否完整
4. 输出目录权限是否正确
