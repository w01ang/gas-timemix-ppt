# TimeMixer - Enhanced Well Lifecycle Prediction

基于TimeMixer的井生命周期预测模型，支持不定长输入序列和定长输出序列的8:2比例预测。

## 🚀 主要特性

- ✅ **不定长输入序列**: 支持动态长度的时序数据输入
- ✅ **定长输出序列**: 生成固定长度的预测结果
- ✅ **8:2比例预测**: 输入:输出 = 4:1 (接近8:2)
- ✅ **无平滑过渡**: 预测值直接从模型输出开始
- ✅ **多井多比例训练**: 支持多种分割比例的训练策略
- ✅ **增强可视化**: 4色图表展示完整生命周期
- ✅ **实验管理**: 完整的训练、测试、可视化、归档流程

## 📁 项目结构

```
TimeMixer/
├── scripts/                          # 实验管理脚本
│   ├── train_experiment.py           # 模型训练脚本
│   ├── train_8_2_ratio.py           # 8:2比例专用训练脚本
│   ├── test_and_visualize.py        # 测试和可视化脚本
│   ├── plot_metrics.py              # 指标可视化脚本
│   ├── archive_experiment.py        # 实验归档脚本
│   ├── run_full_experiment.py       # 完整实验流程脚本
│   ├── README_scripts.md            # 脚本使用说明
│   └── QUICK_START.md               # 快速开始指南
├── data_provider/                    # 数据加载器
│   └── data_loader.py               # 支持不定长输入的井数据加载器
├── exp/                             # 实验模块
│   └── exp_long_term_forecasting.py # 长期预测实验类
├── models/                          # 模型定义
├── utils/                           # 工具函数
└── README.md                        # 项目说明
```

## 🛠️ 环境要求

```bash
# 创建conda环境
conda create -n timemixer python=3.10
conda activate timemixer

# 安装依赖
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib scikit-learn
pip install scipy seaborn
```

## 🚀 快速开始

### 1. 8:2比例模型训练

```bash
# 激活环境
conda activate timemixer

# 训练8:2比例模型
python scripts/train_8_2_ratio.py \
    --model_id wellmix_8_2 \
    --total_length 1000 \
    --input_ratio 0.8 \
    --output_ratio 0.2 \
    --train_epochs 100
```

### 2. 完整实验流程

```bash
# 运行完整实验
python scripts/run_full_experiment.py \
    --model_id my_experiment \
    --test_wells 0,1,2,3,4,5,6,7,8,9 \
    --ratios 10,20,30,40,50,60,70,80,90 \
    --seq_len 800 \
    --pred_len 200
```

### 3. 分步执行

```bash
# 步骤1: 训练模型
python scripts/train_experiment.py --model_id my_model

# 步骤2: 测试和可视化
python scripts/test_and_visualize.py --model_id my_model --test_wells 0,1,2,3,4

# 步骤3: 生成指标图表
python scripts/plot_metrics.py --results_dir results_archive/my_model_no_smooth

# 步骤4: 归档结果
python scripts/archive_experiment.py --model_id my_model_no_smooth
```

## 📊 核心功能

### 不定长输入序列支持

- **动态输入长度**: 根据数据长度自动调整输入序列长度
- **最大输入长度**: 3000步
- **最小输入长度**: 100步
- **填充策略**: 输入不足时零填充

### 定长输出序列

- **固定输出长度**: 通过`pred_len`参数控制
- **一致性保证**: 所有预测结果长度一致
- **可配置性**: 支持任意长度的输出序列

### 8:2比例预测

- **输入比例**: 80%的历史数据作为输入
- **输出比例**: 20%的未来数据作为预测目标
- **灵活配置**: 支持自定义比例设置

## 🔧 配置参数

| 参数 | 说明 | 默认值 | 8:2比例示例 |
|------|------|--------|-------------|
| `seq_len` | 输入序列长度 | 3000 | 800 |
| `pred_len` | 预测长度 | 256 | 200 |
| `d_model` | 模型维度 | 256 | 256 |
| `n_heads` | 注意力头数 | 16 | 16 |
| `e_layers` | 编码器层数 | 6 | 6 |
| `d_layers` | 解码器层数 | 3 | 3 |
| `d_ff` | 前馈网络维度 | 1024 | 1024 |

## 📈 输出结果

### 可视化图表
- **4色图表**: 早期历史(紫)、输入段(蓝)、真实输出(绿)、预测输出(橙)
- **完整生命周期**: 展示井的完整生产周期
- **预测窗口放大**: 详细展示预测区域

### 评估指标
- **MAE**: 平均绝对误差
- **RMSE**: 均方根误差
- **MAPE**: 平均绝对百分比误差
- **跳跃分析**: 预测起始值与输入末尾值的差异

### 结果文件
```
results_archive/
└── {model_id}_no_smooth/
    ├── detailed_results_no_smooth.csv      # 详细结果
    ├── by_well_summary_no_smooth.csv       # 按井汇总
    ├── overall_summary_no_smooth.csv       # 整体汇总
    ├── well_0_ratio_50_no_smooth.pdf       # 井0-50%分割图表
    ├── well_0_ratio_50_no_smooth.csv       # 井0-50%分割数据
    └── ...                                  # 其他结果文件
```

## 🎯 使用场景

- **油井产量预测**: 基于历史产量数据预测未来产量
- **时间序列预测**: 支持任意长度的时序数据预测
- **生产规划**: 为生产决策提供数据支持
- **异常检测**: 通过预测偏差识别异常情况

## 📝 主要改进

1. **支持不定长输入**: 修改数据加载器支持动态输入长度
2. **8:2比例预测**: 实现输入:输出=4:1的预测比例
3. **无平滑过渡**: 去除人工平滑，保持模型原始预测能力
4. **增强可视化**: 4色图表展示完整生命周期
5. **实验管理**: 完整的训练、测试、可视化、归档流程
6. **多井多比例**: 支持多种分割比例的训练策略

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

## 📄 许可证

本项目基于原始TimeMixer项目进行修改和增强。

## 📞 联系方式

如有问题，请通过GitHub Issues联系。

---
**版本**: v2.0 (无平滑过渡版本)  
**更新时间**: 2025-09-16
