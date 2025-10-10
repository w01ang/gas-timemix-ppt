# TimeMixer - Enhanced Well Lifecycle Prediction

基于TimeMixer的井生命周期预测模型，支持固定长度输入输出的滑动窗口训练方法。

## 🚀 主要特性

- ✅ **滑动窗口训练**: 固定长度输入输出，无填充无截断
- ✅ **清晰的参数定义**: input_len和output_len含义明确
- ✅ **训练测试一致**: 使用相同的数据采样策略
- ✅ **灵活的步长控制**: 支持自定义滑动步长
- ✅ **完整的实验管理**: 训练、测试、可视化、归档一体化
- ✅ **增强可视化**: 多窗口预测结果展示
- ✅ **向后兼容**: 保留旧方法支持

## 🆕 最新更新 (2025-10-10)

实现了**全新的滑动窗口训练方法**，解决了旧方法中的诸多问题：
- 无填充/截断，所有样本长度完全一致
- 参数含义清晰（input_len就是输入长度，output_len就是输出长度）
- 训练和测试使用相同策略，评估更可靠

详见：[更新日志](CHANGELOG.md) | [方法对比](METHOD_COMPARISON.md) | [使用指南](SLIDING_WINDOW_GUIDE.md)

## 📁 项目结构

```
gas-timemix/
├── scripts/                          # 实验脚本
│   ├── train_sliding_window.py      # 🆕 滑动窗口训练（推荐）
│   ├── test_sliding_window.py       # 🆕 滑动窗口测试（推荐）
│   ├── train_8_2_ratio.py           # 旧：多比例训练（兼容）
│   └── test_and_visualize.py        # 旧：多比例测试（兼容）
├── data_provider/                    # 数据加载器
│   ├── data_loader.py               # ✨ 支持滑动窗口的数据加载器
│   └── data_factory.py              # ✨ 数据工厂（支持step_len）
├── exp/                             # 实验模块
│   └── exp_long_term_forecasting.py # 长期预测实验类
├── models/                          # TimeMixer模型
├── utils/                           # 工具函数
├── SLIDING_WINDOW_GUIDE.md          # 🆕 滑动窗口使用指南
├── METHOD_COMPARISON.md             # 🆕 新旧方法对比
├── CHANGELOG.md                     # 🆕 更新日志
└── README.md                        # 本文件
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

### 方法1: 滑动窗口训练（推荐）⭐

```bash
# 激活环境
conda activate timemixer

# 训练模型
python scripts/train_sliding_window.py \
    --model_id wellmix_640_160 \
    --input_len 640 \
    --output_len 160 \
    --step_len 160 \
    --train_epochs 100 \
    --use_gpu

# 测试模型
python scripts/test_sliding_window.py \
    --model_id wellmix_640_160 \
    --max_wells 10
```

**参数说明**：
- `input_len`: 输入序列长度（固定640步）
- `output_len`: 输出序列长度（固定160步）
- `step_len`: 滑动步长（默认=output_len，无重叠）
- 比例 = 640:160 = 4:1

### 方法2: 旧方法（兼容）

```bash
# 训练
python scripts/train_8_2_ratio.py \
    --model_id wellmix_old \
    --total_length 800 \
    --input_ratio 0.8 \
    --output_ratio 0.2 \
    --train_epochs 100

# 测试
python scripts/test_and_visualize.py \
    --model_id wellmix_old \
    --ratios 10,20,30,40,50,60,70,80,90
```

### 选择哪种方法？

| 场景 | 推荐方法 |
|------|---------|
| 新实验 | ✅ 滑动窗口（参数清晰，效果好）|
| 与旧实验对比 | 旧方法（保持一致性）|
| 不确定 | ✅ 滑动窗口（更符合标准实践）|

详细对比请参考：[方法对比文档](METHOD_COMPARISON.md)

## 📊 核心功能

### 滑动窗口训练（新方法）

- **固定长度输入**: 所有样本输入长度完全一致（如640步）
- **固定长度输出**: 所有样本输出长度完全一致（如160步）
- **无填充无截断**: 所有数据都是真实值，无人工填充
- **统一采样**: 训练、验证、测试使用相同策略
- **可控密度**: 通过step_len调整样本数量

### 灵活的比例控制

- **标准比例**: 4:1 (input_len=640, output_len=160)
- **短期预测**: 2:1 (input_len=640, output_len=320)
- **长期预测**: 8:1 (input_len=640, output_len=80)
- **自定义**: 任意input_len和output_len组合

### 完整的实验流程

- **配置保存**: 自动保存所有实验参数
- **模型检查点**: 保存最佳模型权重
- **详细日志**: 训练过程完整记录
- **结果可视化**: 自动生成图表和CSV
- **按井统计**: 详细的per-well分析

## 🔧 配置参数

### 滑动窗口参数

| 参数 | 说明 | 推荐值 | 示例 |
|------|------|--------|------|
| `input_len` | 输入序列长度（固定） | 640 | 640 |
| `output_len` | 输出序列长度（固定） | 160 | 160 |
| `step_len` | 滑动窗口步长 | =output_len | 160 |

### 模型参数

| 参数 | 说明 | 默认值 | 大模型 |
|------|------|--------|--------|
| `d_model` | 模型维度 | 256 | 512 |
| `n_heads` | 注意力头数 | 16 | 32 |
| `e_layers` | 编码器层数 | 6 | 8 |
| `d_layers` | 解码器层数 | 3 | 4 |
| `d_ff` | 前馈网络维度 | 1024 | 2048 |

### 训练参数

| 参数 | 说明 | 默认值 | 快速实验 | 正式训练 |
|------|------|--------|---------|---------|
| `train_epochs` | 训练轮数 | 100 | 10 | 100-200 |
| `batch_size` | 批大小 | 8 | 16 | 8 |
| `learning_rate` | 学习率 | 1e-4 | 1e-3 | 1e-4 |
| `patience` | 早停耐心 | 20 | 5 | 20 |

## 📈 输出结果

### 可视化图表（滑动窗口）
- **多窗口展示**: 显示所有滑动窗口的预测结果
- **颜色编码**: 
  - 浅灰色：完整井序列
  - 蓝色：输入段（每个窗口）
  - 绿色：真实输出
  - 橙色虚线：预测输出
  - 红色虚线：预测起点标记

### 评估指标
- **MAE**: 平均绝对误差
- **RMSE**: 均方根误差
- **MAPE**: 平均绝对百分比误差
- **按窗口**: 每个滑动窗口的独立指标
- **按井汇总**: 每口井所有窗口的平均指标

### 结果文件结构
```
test_results/
└── {model_id}/
    ├── results.csv                  # 详细结果（每个窗口一行）
    ├── well_statistics.csv          # 按井统计
    ├── well_0_sliding_windows.pdf   # 井0的所有窗口可视化
    ├── well_1_sliding_windows.pdf   # 井1的所有窗口可视化
    └── ...

experiments/
└── {model_id}/
    └── config.json                  # 实验配置

checkpoints/
└── {model_id}/
    ├── checkpoint.pth               # 模型权重
    └── training_log.txt             # 训练日志
```

## 🎯 使用场景

- **油井产量预测**: 基于历史产量数据预测未来产量
- **时间序列预测**: 支持任意长度的时序数据预测
- **生产规划**: 为生产决策提供数据支持
- **异常检测**: 通过预测偏差识别异常情况

## 📝 主要特点

### 新方法（滑动窗口）优势
1. ✅ **参数清晰**: input_len和output_len含义明确
2. ✅ **无数据损失**: 无填充、无截断，全是真实数据
3. ✅ **训练测试一致**: 使用相同的采样策略
4. ✅ **样本质量统一**: 所有样本长度完全一致
5. ✅ **充分利用数据**: 长井生成更多有效样本
6. ✅ **可控样本密度**: 通过step_len灵活调整

### 向后兼容
- 保留旧方法（多比例分割）的所有脚本
- 新旧方法可以共存
- 可以在同一数据集上对比两种方法

### 完整的工具链
- 🚀 训练：自动保存配置和模型
- 🧪 测试：批量预测和指标计算
- 📊 可视化：自动生成图表
- 📁 管理：实验配置和结果归档

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

## 📄 许可证

本项目基于原始TimeMixer项目进行修改和增强。

## 📞 联系方式

如有问题，请通过GitHub Issues联系。

## 📚 详细文档

- **[滑动窗口使用指南](SLIDING_WINDOW_GUIDE.md)**: 新方法完整教程
- **[方法对比](METHOD_COMPARISON.md)**: 新旧方法详细对比
- **[更新日志](CHANGELOG.md)**: 所有更新记录
- **[实验指南](EXPERIMENT_GUIDE.md)**: 旧方法使用说明（兼容）

## ❓ 常见问题

**Q: 应该使用哪种方法？**  
A: 推荐使用滑动窗口方法（`train_sliding_window.py`），参数清晰，效果更好。

**Q: 如何选择input_len和output_len？**  
A: 建议 `input_len + output_len ≤ 平均井长度的50%`，常用比例4:1（如640:160）。

**Q: step_len应该设多大？**  
A: 初始实验用`step_len = output_len`（无重叠）；数据不足可用`output_len/2`（50%重叠）。

**Q: 训练很慢怎么办？**  
A: 启用GPU（`--use_gpu`），增大step_len，或减小模型容量。

**Q: 旧模型还能用吗？**  
A: 可以，旧脚本和模型完全保留，与新方法独立。

---
**版本**: v3.0 (滑动窗口版本)  
**更新时间**: 2025-10-10
