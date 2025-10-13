# 修复 NaN 损失问题指南

训练时验证损失变成NaN是梯度爆炸或数值不稳定的标志。

---

## 🔍 诊断流程

### 1. 确定NaN出现时机

**在第一个epoch就NaN？**
- ✅ 是 → 学习率过高或数据问题
- ❌ 否 → 梯度爆炸或数值不稳定

**损失值变化趋势：**
```
正常: 1.5 → 1.2 → 1.0 → 0.8 → ...
异常: 1.5 → 3.2 → 10.5 → Inf → NaN  ← 梯度爆炸
异常: 1.5 → 1.2 → NaN                ← 突然崩溃
```

---

## 🚨 立即解决方案（按优先级）

### 方案1：大幅降低学习率 ⭐⭐⭐⭐⭐

**最常见的原因和最快的解决方法！**

```bash
# 如果当前使用 --learning_rate 1e-4 或更高
# 立即降低到:

python scripts/train_sliding_window.py \
  --model_id slide_fix_nan_lr1e5 \
  --input_len 3000 \
  --output_len 1000 \
  --step_len 50 \
  --batch_size 16 \
  --learning_rate 1e-5 \
  --train_epochs 60 \
  --use_gpu
```

**如果还是NaN，继续降低：**
```bash
--learning_rate 5e-6  # 更保守
```

---

### 方案2：添加梯度裁剪 ⭐⭐⭐⭐⭐

修改训练脚本，添加梯度裁剪防止梯度爆炸：

```python
# 在 train_sliding_window.py 的训练循环中
# 找到 loss.backward() 之后，添加：

import torch.nn as nn

# loss.backward() 之后添加
nn.utils.clip_grad_norm_(exp.model.parameters(), max_norm=1.0)

model_optim.step()
```

**完整修改示例：**

```python
# 在训练循环中 (大约第160行附近)
loss.backward()

# 【添加这一行】梯度裁剪
torch.nn.utils.clip_grad_norm_(exp.model.parameters(), max_norm=1.0)

model_optim.step()
```

---

### 方案3：检查并清理数据 ⭐⭐⭐⭐

```python
# 创建数据检查脚本
import pandas as pd
import numpy as np

# 读取数据
data_path = '/Users/wangjr/Documents/yk/timemixer/data/preprocessed_daily_gas_by_well.csv'
df = pd.read_csv(data_path)

print("数据异常值检查:")
print("-" * 50)

# 检查NaN
nan_count = df['OT'].isna().sum()
print(f"NaN值数量: {nan_count}")

# 检查Inf
inf_count = np.isinf(df['OT']).sum()
print(f"Inf值数量: {inf_count}")

# 检查异常大的值
max_val = df['OT'].max()
min_val = df['OT'].min()
mean_val = df['OT'].mean()
std_val = df['OT'].std()

print(f"\n数据统计:")
print(f"  最大值: {max_val}")
print(f"  最小值: {min_val}")
print(f"  均值: {mean_val:.2f}")
print(f"  标准差: {std_val:.2f}")

# 检查是否有异常大的值
threshold = mean_val + 10 * std_val
outliers = (df['OT'] > threshold).sum()
print(f"\n超过均值+10倍标准差的异常值: {outliers}")

# 如果发现异常，清理数据
if nan_count > 0 or inf_count > 0 or outliers > 100:
    print("\n⚠️  发现异常数据！建议清理。")
    
    # 清理方案
    df_clean = df.copy()
    
    # 1. 删除NaN和Inf
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna()
    
    # 2. 裁剪异常值
    upper_bound = mean_val + 5 * std_val
    lower_bound = max(0, mean_val - 5 * std_val)  # 产量不能为负
    df_clean['OT'] = df_clean['OT'].clip(lower_bound, upper_bound)
    
    # 保存清理后的数据
    output_path = '/Users/wangjr/Documents/yk/timemixer/data/preprocessed_daily_gas_by_well_cleaned.csv'
    df_clean.to_csv(output_path, index=False)
    print(f"\n✅ 清理后的数据已保存到: {output_path}")
    print(f"   原始数据行数: {len(df)}")
    print(f"   清理后行数: {len(df_clean)}")
else:
    print("\n✅ 数据正常，无异常值")
```

保存为 `scripts/check_data.py` 并运行：
```bash
cd /Users/wangjr/Documents/yk/timemixer/timemixer-ppt/gas-timemix
python scripts/check_data.py
```

---

### 方案4：使用更稳定的损失函数 ⭐⭐⭐

从MAE改为Huber Loss（对异常值更鲁棒）：

```python
# 在 exp/exp_long_term_forecasting.py 的 _select_criterion 中
def _select_criterion(self):
    if self.args.loss == 'Huber':
        criterion = nn.HuberLoss(delta=1.0)  # 添加Huber Loss
    elif self.args.loss == 'MSE':
        criterion = nn.MSELoss()
    elif self.args.loss == 'MAE':
        criterion = nn.L1Loss()
    else:
        criterion = nn.L1Loss()
    return criterion
```

然后训练时使用：
```bash
python scripts/train_sliding_window.py \
  --model_id slide_huber_loss \
  --loss Huber \
  --learning_rate 5e-5 \
  ...
```

---

### 方案5：降低模型复杂度 ⭐⭐⭐

如果以上都不行，可能是模型太复杂导致不稳定：

```bash
python scripts/train_sliding_window.py \
  --model_id slide_simple_stable \
  --input_len 3000 \
  --output_len 1000 \
  --step_len 50 \
  --batch_size 32 \
  --d_model 64 \
  --n_heads 4 \
  --e_layers 2 \
  --d_layers 1 \
  --d_ff 256 \
  --dropout 0.2 \
  --learning_rate 5e-5 \
  --train_epochs 60 \
  --use_gpu
```

---

## 🛠️ 完整修复流程

### 第一步：快速修复（2分钟）

```bash
# 立即用更低的学习率重新训练
python scripts/train_sliding_window.py \
  --model_id slide_fix_nan \
  --input_len 3000 \
  --output_len 1000 \
  --step_len 50 \
  --batch_size 16 \
  --learning_rate 1e-5 \
  --train_epochs 60 \
  --use_gpu
```

### 第二步：添加梯度裁剪（5分钟）

修改 `scripts/train_sliding_window.py`：

```python
# 找到第160行左右的训练循环
# 在 loss.backward() 之后添加：

loss.backward()
torch.nn.utils.clip_grad_norm_(exp.model.parameters(), max_norm=1.0)  # 【新增】
model_optim.step()
```

### 第三步：检查数据（5分钟）

创建并运行数据检查脚本（见方案3）

### 第四步：重新训练（30分钟）

使用修改后的脚本和清理后的数据重新训练

---

## 📊 预防措施

### 训练时监控指标

添加到训练循环中：

```python
# 在每个batch后检查
if torch.isnan(loss):
    print(f"⚠️  检测到NaN! Batch {i}, Loss: {loss.item()}")
    print(f"   学习率: {scheduler.get_last_lr()[0]}")
    # 保存出问题的batch数据
    torch.save({
        'batch_x': batch_x,
        'batch_y': batch_y,
        'model_state': exp.model.state_dict(),
    }, 'nan_debug.pth')
    raise ValueError("训练出现NaN，已保存调试信息")

# 监控梯度
if i % 10 == 0:
    total_norm = 0
    for p in exp.model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    if total_norm > 10.0:
        print(f"⚠️  警告: 梯度范数过大 {total_norm:.2f}")
```

### 推荐的稳定训练配置

```bash
python scripts/train_sliding_window.py \
  --model_id slide_ultra_stable \
  --input_len 3000 \
  --output_len 1000 \
  --step_len 50 \
  --batch_size 32 \
  --d_model 128 \
  --n_heads 8 \
  --e_layers 3 \
  --d_layers 1 \
  --d_ff 512 \
  --dropout 0.2 \
  --learning_rate 5e-5 \
  --patience 25 \
  --train_epochs 80 \
  --use_gpu
```

**关键配置点：**
- ✅ 学习率：5e-5（保守）
- ✅ Batch size：32（更稳定）
- ✅ 模型中等大小（d_model=128）
- ✅ Dropout：0.2（防止过拟合）

---

## 🔬 调试技巧

### 1. 找出哪个epoch出现NaN

```bash
# 查看训练日志
tail -20 logs/YOUR_MODEL_ID/training_summary.txt
```

### 2. 检查模型权重

```python
import torch

checkpoint = torch.load('checkpoints/YOUR_MODEL_ID/checkpoint.pth')
for key, value in checkpoint.items():
    if torch.isnan(value).any():
        print(f"发现NaN权重: {key}")
    if torch.isinf(value).any():
        print(f"发现Inf权重: {key}")
```

### 3. 逐步降低学习率测试

```bash
# 测试哪个学习率是安全的
for lr in 1e-4 5e-5 1e-5 5e-6; do
  echo "测试学习率: ${lr}"
  python scripts/train_sliding_window.py \
    --model_id test_lr_${lr} \
    --learning_rate ${lr} \
    --train_epochs 5 \
    --use_gpu
  
  # 检查是否成功
  if [ $? -eq 0 ]; then
    echo "✅ 学习率 ${lr} 可用"
  else
    echo "❌ 学习率 ${lr} 导致NaN"
  fi
done
```

---

## 💡 常见问题

### Q1: 为什么学习率1e-4会导致NaN？
A: OneCycleLR会在训练中期将学习率提升到峰值（约2-3倍），所以实际最高学习率可能达到2e-4或3e-4，对于大模型太高了。

### Q2: 梯度裁剪的max_norm该设多少？
A: 
- 1.0：非常保守，适合不稳定的训练
- 5.0：中等，适合大多数情况
- 10.0：宽松，适合稳定的训练

### Q3: 修改后需要重新训练吗？
A: 是的，一旦出现NaN，模型权重已损坏，必须重新开始。

### Q4: 如何知道是哪一层导致的NaN？
A: 在训练循环中添加：
```python
for name, param in exp.model.named_parameters():
    if param.grad is not None and torch.isnan(param.grad).any():
        print(f"NaN梯度在: {name}")
```

---

## 🎯 推荐行动方案

**如果你的模型刚开始就NaN：**

1. 立即降低学习率到 **1e-5**
2. 检查数据是否有异常值
3. 使用较小的模型（d_model=64）

**如果训练到中途才NaN：**

1. 添加梯度裁剪（max_norm=1.0）
2. 降低学习率到 **5e-5**
3. 增加batch size到32

**如果以上都不行：**

1. 使用Huber Loss替代MAE
2. 大幅简化模型（d_model=32）
3. 检查是否是数据问题

---

## ✅ 验证修复是否成功

训练开始后，观察前10个epoch：

```
Epoch 1: Train=1.5, Vali=1.3, Test=1.4  ✅
Epoch 2: Train=1.2, Vali=1.1, Test=1.2  ✅
Epoch 3: Train=1.0, Vali=0.9, Test=1.0  ✅
...

如果看到平滑下降，说明修复成功！
```

**不正常的情况：**
```
Epoch 1: Train=5.2, Vali=3.8  ⚠️  初始损失过高
Epoch 2: Train=NaN            ❌  立即崩溃
```

---

**立即尝试方案1（降低学习率）！这解决了90%的NaN问题！** 🚀

