# 训练日志使用指南

## 📝 概述

训练脚本 `train_sliding_window.py` 会自动记录训练过程中的所有损失指标到日志文件中。

## 📁 日志文件结构

训练时会在 `logs/<model_id>/` 目录下生成两个日志文件：

```
logs/
└── <model_id>/
    ├── training_loss.csv         # CSV格式的详细损失记录
    └── training_summary.txt      # 文本格式的训练摘要
```

## 📊 日志文件详解

### 1. training_loss.csv

CSV格式的详细损失记录，每个epoch一行，包含以下列：

| 列名 | 说明 | 示例 |
|------|------|------|
| Epoch | 训练轮次 | 1, 2, 3, ... |
| Train_Loss | 训练损失 | 0.3386983 |
| Vali_Loss | 验证损失 | 0.3731581 |
| Test_Loss | 测试损失 | 0.4059942 |
| Learning_Rate | 当前学习率 | 0.0000687410 |
| Time | 记录时间 | 2025-10-10 15:30:45 |

**示例内容**：
```csv
Epoch,Train_Loss,Vali_Loss,Test_Loss,Learning_Rate,Time
1,0.3386983,0.3731581,0.4059942,0.0000687410,2025-10-10 15:30:45
2,0.2468214,0.3570666,0.3897177,0.0000000022,2025-10-10 15:42:12
...
```

**用途**：
- 可以导入Excel/Python进行可视化分析
- 绘制损失曲线
- 分析学习率变化趋势

### 2. training_summary.txt

文本格式的训练摘要，包含：

**训练开始信息**：
```
训练日志 - 模型ID: wellmix_640_160
开始时间: 2025-10-10 15:30:00
================================================================================
```

**每个Epoch的详细记录**：
```
Epoch 1:
  Train Loss: 0.3386983
  Vali Loss:  0.3731581
  Test Loss:  0.4059942
  Learning Rate: 0.0000687410
  Time: 12.82s
--------------------------------------------------------------------------------
```

**训练结束信息**：
```
================================================================================
训练完成
结束时间: 2025-10-10 18:45:23
总训练时间: 11523.45s (192.06分钟)
最佳Epoch: 87
最佳验证损失: 0.2345678
早停触发: 是
================================================================================
```

## 🚀 使用示例

### 训练模型（自动记录日志）

```bash
python scripts/train_sliding_window.py \
  --model_id my_model_640_160 \
  --input_len 640 \
  --output_len 160 \
  --train_epochs 100 \
  --use_gpu
```

训练完成后，日志会自动保存在：
- `logs/my_model_640_160/training_loss.csv`
- `logs/my_model_640_160/training_summary.txt`

### 查看日志

**查看文本摘要**：
```bash
cat logs/my_model_640_160/training_summary.txt
```

**查看CSV（前10行）**：
```bash
head -11 logs/my_model_640_160/training_loss.csv
```

**查看最后几个epoch**：
```bash
tail -20 logs/my_model_640_160/training_summary.txt
```

## 📈 分析损失曲线

### 使用Python绘制损失曲线

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取日志
df = pd.read_csv('logs/my_model_640_160/training_loss.csv')

# 绘制损失曲线
plt.figure(figsize=(12, 6))

plt.plot(df['Epoch'], df['Train_Loss'], label='Train Loss', marker='o')
plt.plot(df['Epoch'], df['Vali_Loss'], label='Validation Loss', marker='s')
plt.plot(df['Epoch'], df['Test_Loss'], label='Test Loss', marker='^')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curves')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=300)
plt.show()
```

### 使用Excel分析

1. 打开 `training_loss.csv`
2. 选中数据列
3. 插入 → 图表 → 折线图
4. 可以分析：
   - 损失下降趋势
   - 是否过拟合（train loss下降但vali loss上升）
   - 最佳epoch位置
   - 学习率变化

## 🔍 日志信息解读

### 损失值含义

假设数据经过标准化（均值0，标准差1）：

```
Train Loss = 0.3386983
```

这表示：
- **MSE损失** = 0.34（均方误差）
- **RMSE** ≈ √0.34 ≈ 0.58（均方根误差，约0.58个标准差）

如果原始数据标准差为1000：
```
实际RMSE ≈ 0.58 × 1000 = 580 单位
```

### 验证损失 vs 测试损失

- **验证损失（Vali Loss）**：用于早停决策
- **测试损失（Test Loss）**：用于评估最终性能
- 理想情况：三者应该接近
- 过拟合：Train Loss << Vali Loss

### 学习率变化

使用 OneCycleLR 调度器：
- 前期（0-20%）：学习率逐渐上升
- 中期（20%-80%）：学习率保持高位
- 后期（80%-100%）：学习率快速下降

## ⚠️ 常见问题

### Q1: 日志文件在哪里？

A: 在 `logs/<model_id>/` 目录下，与模型ID同名。

### Q2: 如何查看训练是否早停？

A: 查看 `training_summary.txt` 最后几行，会显示"早停触发: 是/否"。

### Q3: 损失突然变大怎么办？

A: 
1. 检查学习率是否过大
2. 查看是否数据异常
3. 考虑降低学习率或增加batch_size

### Q4: 如何对比多个模型的训练过程？

A: 使用Python脚本读取多个CSV文件，绘制在同一图表上：

```python
import pandas as pd
import matplotlib.pyplot as plt

models = ['model1', 'model2', 'model3']

plt.figure(figsize=(12, 6))
for model_id in models:
    df = pd.read_csv(f'logs/{model_id}/training_loss.csv')
    plt.plot(df['Epoch'], df['Vali_Loss'], label=model_id, marker='o')

plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Model Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## 📚 相关文件

- **训练脚本**: `scripts/train_sliding_window.py`
- **日志目录**: `logs/`
- **模型目录**: `checkpoints/`
- **实验配置**: `experiments/<model_id>/config.json`

## 💡 最佳实践

1. **定期备份日志**：训练完成后备份日志文件
2. **及时分析**：训练过程中定期查看损失变化
3. **对比实验**：使用不同model_id区分不同实验
4. **记录说明**：在实验配置的description字段添加实验说明
5. **可视化监控**：使用脚本实时绘制损失曲线

---

**提示**：所有日志会自动生成，无需手动创建或配置！

