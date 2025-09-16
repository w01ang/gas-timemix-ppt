# TimeMixer 快速开始指南

## 🚀 一键运行完整实验

```bash
# 激活环境
conda activate timemixer

# 运行完整实验（推荐）
python scripts/run_full_experiment.py \
    --model_id my_experiment \
    --test_wells 0,1,2,3,4,5,6,7,8,9 \
    --ratios 10,20,30,40,50,60,70,80,90
```

## 📋 分步执行

### 1. 训练模型
```bash
python scripts/train_experiment.py --model_id my_experiment
```

### 2. 测试和可视化
```bash
python scripts/test_and_visualize.py --model_id my_experiment --test_wells 0,1,2,3,4
```

### 3. 生成指标图表
```bash
python scripts/plot_metrics.py --results_dir results_archive/my_experiment_no_smooth
```

### 4. 归档结果
```bash
python scripts/archive_experiment.py --model_id my_experiment_no_smooth
```

## ⚙️ 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_id` | 实验ID（必需） | - |
| `--test_wells` | 测试井索引 | 0,1,2,3,4,5,6,7,8,9 |
| `--ratios` | 分割比例(%) | 10,20,30,40,50,60,70,80,90 |
| `--seq_len` | 输入序列长度 | 3000 |
| `--pred_len` | 预测长度 | 256 |
| `--d_model` | 模型维度 | 256 |

## 📊 输出结果

结果保存在 `results_archive/{model_id}_no_smooth/` 目录：
- PDF图表：`well_{well_idx}_ratio_{ratio}_no_smooth.pdf`
- CSV数据：`well_{well_idx}_ratio_{ratio}_no_smooth.csv`
- 汇总结果：`detailed_results_no_smooth.csv`

## 🔧 特性

- ✅ **无平滑过渡**：预测值直接从模型输出开始
- ✅ **动态输入长度**：根据分割比例调整输入长度
- ✅ **多井多比例测试**：支持多口井和多种分割比例
- ✅ **增强可视化**：4色图表展示完整生命周期

## 🆘 常见问题

1. **环境问题**：确保使用 `conda activate timemixer`
2. **模型不存在**：先运行训练脚本
3. **内存不足**：减少 `batch_size` 或 `test_wells` 数量

---
**版本：** v2.0 (无平滑过渡版本)  
**更新时间：** 2025-09-16
