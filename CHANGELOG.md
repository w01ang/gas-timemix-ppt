# 更新日志

## 2025-10-10: 滑动窗口方法重构 🎉

### 重大更新

实现了全新的滑动窗口训练方法，解决了旧方法中参数含义不清、训练测试不一致的问题。

### 🆕 新增功能

#### 1. 滑动窗口数据加载器
- **文件**: `data_provider/data_loader.py`
- **新增参数**: `step_len` - 滑动窗口步长
- **改进**:
  - 使用固定长度的输入和输出窗口
  - 无填充、无截断，所有样本长度完全一致
  - 按井划分数据集：70% 训练，10% 验证，20% 测试
  - 支持自定义步长控制样本密度

#### 2. 新训练脚本
- **文件**: `scripts/train_sliding_window.py`
- **核心参数**:
  - `--input_len`: 输入序列长度（固定）
  - `--output_len`: 输出序列长度（固定）
  - `--step_len`: 滑动步长（默认=output_len）
- **使用示例**:
  ```bash
  python scripts/train_sliding_window.py \
    --model_id wellmix_640_160 \
    --input_len 640 \
    --output_len 160 \
    --step_len 160 \
    --train_epochs 100 \
    --use_gpu
  ```

#### 3. 新测试脚本
- **文件**: `scripts/test_sliding_window.py`
- **功能**:
  - 对测试集每口井生成所有滑动窗口的预测
  - 计算每个窗口的MAE/RMSE/MAPE指标
  - 生成可视化图表（显示所有窗口）
  - 输出详细结果CSV和统计汇总
- **使用示例**:
  ```bash
  python scripts/test_sliding_window.py \
    --model_id wellmix_640_160 \
    --max_wells 10
  ```

#### 4. 数据工厂更新
- **文件**: `data_provider/data_factory.py`
- **改进**: 支持传递`step_len`参数到WELLS数据集

#### 5. 新增文档
- **`SLIDING_WINDOW_GUIDE.md`**: 滑动窗口方法完整使用指南
- **`METHOD_COMPARISON.md`**: 新旧方法详细对比
- **`CHANGELOG.md`**: 本文件，记录所有更新

### 🔧 核心改进

#### 问题1: 参数含义不清
**旧方法**:
```bash
--total_length 800 --input_ratio 0.8
# seq_len=640，但实际输入长度从400到3200不等
```

**新方法**:
```bash
--input_len 640 --output_len 160
# 所有输入长度精确为640步，所有输出长度精确为160步
```

#### 问题2: 填充和截断导致信息损失
**旧方法**:
- 早期分割点：大量零填充（如10%分割，400步数据被填充到640步）
- 后期分割点：截断长输入（如80%分割，3200步被截断到640步）

**新方法**:
- 无填充、无截断
- 所有样本都是完整的真实数据

#### 问题3: 训练测试不一致
**旧方法**:
- 训练使用9个固定比例分割点（10%-90%）
- 测试可以使用任意比例
- 数据分布可能不同

**新方法**:
- 训练、验证、测试使用完全相同的滑动窗口策略
- 唯一区别是井的划分
- 数据分布一致

### 📊 性能对比

使用640输入，160输出，100口井平均长度3500步：

| 指标 | 旧方法 | 新方法 |
|------|--------|--------|
| 训练样本数 | 720 | ~1190 |
| 测试样本数 | 180 | ~340 |
| 输入质量 | 不一致（填充/截断） | 一致（真实数据） |
| 训练测试一致性 | ❌ 不一致 | ✅ 完全一致 |
| 参数清晰度 | ❌ 混淆 | ✅ 清晰 |

### 🔄 向后兼容

- ✅ 旧脚本 (`train_8_2_ratio.py`, `test_and_visualize.py`) 仍可使用
- ✅ 旧模型仍可加载和测试
- ✅ 新旧方法可以共存（使用不同model_id）

### 📁 文件结构

```
gas-timemix/
├── data_provider/
│   ├── data_loader.py          # ✨ 更新：支持step_len
│   └── data_factory.py         # ✨ 更新：传递step_len
├── scripts/
│   ├── train_8_2_ratio.py      # 旧方法（保留）
│   ├── test_and_visualize.py   # 旧方法测试（保留）
│   ├── train_sliding_window.py # 🆕 新方法训练
│   └── test_sliding_window.py  # 🆕 新方法测试
├── SLIDING_WINDOW_GUIDE.md     # 🆕 使用指南
├── METHOD_COMPARISON.md        # 🆕 方法对比
├── CHANGELOG.md                # 🆕 本文件
└── EXPERIMENT_GUIDE.md         # 原实验指南（旧方法）
```

### 🎯 使用建议

**新实验（推荐）**:
```bash
# 训练
python scripts/train_sliding_window.py \
  --model_id my_new_model \
  --input_len 640 \
  --output_len 160 \
  --train_epochs 100 \
  --use_gpu

# 测试
python scripts/test_sliding_window.py \
  --model_id my_new_model
```

**旧实验（兼容）**:
```bash
# 训练
python scripts/train_8_2_ratio.py \
  --model_id my_old_model \
  --total_length 800 \
  --input_ratio 0.8 \
  --train_epochs 100

# 测试
python scripts/test_and_visualize.py \
  --model_id my_old_model
```

### 🐛 Bug修复

1. **NumPy 2.0兼容**: 将 `np.Inf` 替换为 `np.inf`
2. **设备一致性**: 确保所有张量在预测时都移动到模型设备
3. **配置加载**: 测试脚本加载配置时补全所有必需参数
4. **空结果处理**: 处理测试时结果为空的情况

### 📚 详细文档

- **快速开始**: 参见 `SLIDING_WINDOW_GUIDE.md`
- **方法对比**: 参见 `METHOD_COMPARISON.md`
- **旧方法**: 参见 `EXPERIMENT_GUIDE.md`

---

## 历史版本

### 初始版本 (2025-10-10之前)

- 基础TimeMixer实现
- 多比例分割训练方法
- 基本的测试和可视化
- 实验管理和配置保存

---

## 未来计划

- [ ] 添加断点续训功能
- [ ] 支持多变量预测（features='M'）
- [ ] 添加更多数据增强方法
- [ ] 实现自动超参数搜索
- [ ] 添加模型集成功能
- [ ] 支持分布式训练

---

## 贡献者

- 数据加载器重构
- 滑动窗口实现
- 完整文档编写
- Bug修复和优化

---

## 反馈

如有问题或建议，请查看相关文档或提交issue。

