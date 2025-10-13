# TimeMixer 调参指南

## 📋 完整参数列表

### 必需参数
| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `--model_id` | str | 实验唯一标识符 | `slide_optimized_v4` |
| `--input_len` | int | 输入序列长度 | `3000` |
| `--output_len` | int | 输出序列长度 | `1000` |

### 数据参数
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--root_path` | str | `/Users/wangjr/Documents/yk/timemixer/data` | 数据集根目录 |
| `--data_path` | str | `preprocessed_daily_gas_by_well.csv` | 数据集文件名 |
| `--step_len` | int | `output_len` | 滑动窗口步长 |

### 模型架构参数
| 参数 | 类型 | 默认值 | 建议范围 | 说明 |
|------|------|--------|----------|------|
| `--d_model` | int | 256 | 32-128 | 模型维度（越小越防过拟合） |
| `--n_heads` | int | 16 | 2-8 | 注意力头数（必须能整除d_model） |
| `--e_layers` | int | 6 | 1-3 | 编码器层数 |
| `--d_layers` | int | 3 | 1-2 | 解码器层数 |
| `--d_ff` | int | 1024 | 128-512 | 前馈网络维度（通常=d_model×4） |
| `--dropout` | float | 0.1 | 0.1-0.5 | Dropout比例（越大越防过拟合） |

### 训练参数
| 参数 | 类型 | 默认值 | 建议范围 | 说明 |
|------|------|--------|----------|------|
| `--train_epochs` | int | 100 | 50-100 | 训练轮数 |
| `--batch_size` | int | 8 | 8-32 | 批大小（越大越稳定） |
| `--patience` | int | 20 | 10-25 | 早停耐心值 |
| `--learning_rate` | float | 1e-4 | 5e-5 to 1e-4 | 初始学习率 |
| `--use_gpu` | flag | False | - | 启用GPU/MPS加速 |

---

## 🎯 调参策略

### 问题1：过拟合（训练损失<<验证损失）

**症状：**
```
Epoch 30:
  训练损失: 0.05   ← 很低
  验证损失: 0.45   ← 很高
  差距: 0.40       ← 太大！
```

**解决方案（按优先级）：**

1. **降低模型复杂度** ⭐⭐⭐⭐⭐
   ```bash
   --d_model 64      # 从 256 降到 64
   --n_heads 4       # 从 16 降到 4
   --e_layers 2      # 从 6 降到 2
   --d_layers 1      # 从 3 降到 1
   --d_ff 256        # 从 1024 降到 256
   ```

2. **增加正则化** ⭐⭐⭐⭐
   ```bash
   --dropout 0.25    # 从 0.1 提升到 0.25-0.5
   ```

3. **增加batch size** ⭐⭐⭐
   ```bash
   --batch_size 16   # 从 8 提升到 16-32
   ```

4. **降低学习率** ⭐⭐
   ```bash
   --learning_rate 5e-5   # 从 1e-4 降到 5e-5
   ```

5. **减小输入长度** ⭐⭐
   ```bash
   --input_len 2000  # 从 3000 降到 2000
   ```

### 问题2：欠拟合（训练和验证损失都很高）

**症状：**
```
Epoch 30:
  训练损失: 0.50   ← 很高
  验证损失: 0.52   ← 也很高
  差距: 0.02       ← 很小但效果差
```

**解决方案：**

1. **增加模型复杂度** ⭐⭐⭐⭐⭐
   ```bash
   --d_model 128     # 增加到 128
   --n_heads 8       # 增加到 8
   --e_layers 3      # 增加到 3
   ```

2. **提高学习率** ⭐⭐⭐⭐
   ```bash
   --learning_rate 2e-4  # 提升到 2e-4
   ```

3. **增加训练轮数** ⭐⭐⭐
   ```bash
   --train_epochs 150
   --patience 30
   ```

4. **减小dropout** ⭐⭐
   ```bash
   --dropout 0.05    # 降低到 0.05
   ```

### 问题3：训练不稳定（验证损失剧烈波动）

**症状：**
```
验证损失标准差 > 0.3
验证损失在 0.3 和 1.5 之间跳动
```

**解决方案：**

1. **增加batch size** ⭐⭐⭐⭐⭐
   ```bash
   --batch_size 32   # 更大的batch
   ```

2. **降低学习率** ⭐⭐⭐⭐
   ```bash
   --learning_rate 5e-5
   ```

3. **减小step_len（增加样本多样性）** ⭐⭐⭐
   ```bash
   --step_len 50     # 创建更多重叠样本
   ```

4. **降低模型复杂度** ⭐⭐
   ```bash
   --d_model 64
   ```

---

## 🚀 推荐配置方案

### 方案A：稳健型（针对当前过拟合问题）⭐⭐⭐⭐⭐

```bash
python scripts/train_sliding_window.py \
  --model_id slide_optimized_v4 \
  --input_len 3000 \
  --output_len 1000 \
  --step_len 50 \
  --batch_size 16 \
  --d_model 64 \
  --n_heads 4 \
  --e_layers 2 \
  --d_layers 1 \
  --d_ff 256 \
  --dropout 0.25 \
  --learning_rate 8e-5 \
  --patience 15 \
  --train_epochs 50 \
  --use_gpu
```

**预期效果：**
- 测试损失：0.36-0.38
- 过拟合差距：< 0.15
- 验证损失波动：< 0.2

### 方案B：轻量型（如果方案A还过拟合）

```bash
python scripts/train_sliding_window.py \
  --model_id slide_lighter_v4 \
  --input_len 3000 \
  --output_len 1000 \
  --step_len 50 \
  --batch_size 32 \
  --d_model 48 \
  --n_heads 4 \
  --e_layers 2 \
  --d_layers 1 \
  --d_ff 192 \
  --dropout 0.3 \
  --learning_rate 1e-4 \
  --patience 20 \
  --train_epochs 60 \
  --use_gpu
```

### 方案C：平衡型（调整输入输出比例）

```bash
python scripts/train_sliding_window.py \
  --model_id slide_balanced_in2400_out600 \
  --input_len 2400 \
  --output_len 600 \
  --step_len 300 \
  --batch_size 16 \
  --d_model 64 \
  --n_heads 4 \
  --e_layers 2 \
  --d_layers 1 \
  --d_ff 256 \
  --dropout 0.25 \
  --learning_rate 8e-5 \
  --patience 15 \
  --train_epochs 50 \
  --use_gpu
```

### 方案D：极简型（保底方案）

```bash
python scripts/train_sliding_window.py \
  --model_id slide_minimal_v4 \
  --input_len 1600 \
  --output_len 400 \
  --step_len 200 \
  --batch_size 32 \
  --d_model 32 \
  --n_heads 2 \
  --e_layers 2 \
  --d_layers 1 \
  --d_ff 128 \
  --dropout 0.4 \
  --learning_rate 1e-4 \
  --patience 25 \
  --train_epochs 80 \
  --use_gpu
```

---

## 📊 如何评估训练效果

### 健康的训练曲线

```
Epoch 20:
  Train Loss: 0.15
  Vali Loss:  0.18
  Test Loss:  0.17
  Gap: 0.03 ✅
```

**特征：**
- ✅ 训练/验证损失差距 < 0.1
- ✅ 验证损失平滑下降
- ✅ 测试损失接近验证损失
- ✅ 验证损失标准差 < 0.1

### 过拟合的训练曲线

```
Epoch 30:
  Train Loss: 0.05
  Vali Loss:  0.45
  Test Loss:  0.42
  Gap: 0.40 ❌
```

**特征：**
- ❌ 训练/验证损失差距 > 0.3
- ❌ 训练损失持续下降，验证损失反弹
- ❌ 验证损失剧烈波动
- ❌ 最佳epoch出现在早期（前20%轮次）

### 欠拟合的训练曲线

```
Epoch 50:
  Train Loss: 0.45
  Vali Loss:  0.47
  Test Loss:  0.46
  Gap: 0.02 ⚠️
```

**特征：**
- ⚠️ 差距小但所有损失都很高
- ⚠️ 损失下降缓慢或停滞
- ⚠️ 接近训练结束仍在改善

---

## 🔬 高级调参技巧

### 1. 参数组合规则

**n_heads 必须能整除 d_model：**
```bash
# ✅ 正确
--d_model 64 --n_heads 4    # 64 / 4 = 16
--d_model 48 --n_heads 4    # 48 / 4 = 12

# ❌ 错误
--d_model 64 --n_heads 5    # 64 / 5 = 12.8 (不整除)
```

**d_ff 通常是 d_model 的 2-4 倍：**
```bash
--d_model 64 --d_ff 256     # 4倍
--d_model 48 --d_ff 192     # 4倍
--d_model 32 --d_ff 128     # 4倍
```

### 2. step_len 与样本数量

```bash
# 假设井长 = 5000，input_len = 3000，output_len = 1000

--step_len 1000  # 每口井生成 ~2 个样本（无重叠）
--step_len 500   # 每口井生成 ~4 个样本（50%重叠）
--step_len 100   # 每口井生成 ~20 个样本（90%重叠）
```

**建议：**
- 数据少时：`step_len = output_len / 2` （50%重叠）
- 数据充足时：`step_len = output_len` （无重叠）

### 3. 渐进式调参流程

```
1. 从极简模型开始
   ↓ (d_model=32, dropout=0.4)
2. 观察是否欠拟合
   ↓
3. 如果欠拟合 → 逐步增加复杂度
   ↓ (d_model=48 → 64 → 128)
4. 直到出现轻微过拟合
   ↓
5. 增加dropout并微调
   ↓ (dropout=0.25)
6. 找到最佳平衡点 ✅
```

---

## 📈 实验记录模板

建议创建一个实验日志表格：

| 实验ID | d_model | dropout | batch_size | 最佳测试损失 | 过拟合差距 | 备注 |
|--------|---------|---------|------------|--------------|-----------|------|
| v1 | 256 | 0.1 | 8 | 0.422 | 0.609 | 严重过拟合 |
| v2 | 256 | 0.1 | 16 | 0.390 | 0.479 | 仍过拟合 |
| v3 | 64 | 0.1 | 16 | 0.397 | 0.412 | 改善但不够 |
| v4 | 64 | 0.25 | 16 | ? | ? | 待测试 |

---

## 💡 常见问题

### Q1: 为什么降低d_model就能防止过拟合？
A: d_model决定了模型的参数量。参数越多，模型"记忆"训练数据的能力越强，导致过拟合。

### Q2: dropout应该设置多大？
A: 
- 轻度过拟合：0.15-0.2
- 中度过拟合：0.25-0.3
- 严重过拟合：0.3-0.5

### Q3: 如何选择input_len和output_len？
A:
- 比例建议：4:1 到 8:1
- input_len: 包含足够历史信息（1600-3000步）
- output_len: 实际需要预测的长度（400-1000步）

### Q4: 训练多少个epoch合适？
A: 使用早停机制，通常20-50个epoch足够。如果在前10个epoch就触发早停，说明模型太复杂。

### Q5: 如何判断是否需要更多数据？
A: 如果用最简单的模型（d_model=32）仍然过拟合，说明数据量不足。

---

## 🎯 快速决策树

```
训练完成后，观察日志：

训练/验证损失差距 > 0.3？
  └─ 是 → 过拟合 → 降低复杂度 or 增加dropout
  └─ 否 → 继续

验证损失 > 0.5？
  └─ 是 → 检查是否欠拟合
       └─ 训练损失也高？
            └─ 是 → 欠拟合 → 增加复杂度
            └─ 否 → 过拟合 → 降低复杂度
  └─ 否 → 效果不错！

验证损失标准差 > 0.3？
  └─ 是 → 不稳定 → 增加batch_size or 降低学习率
  └─ 否 → 继续

测试损失 < 0.4？
  └─ 是 → 成功！🎉
  └─ 否 → 继续优化
```

---

**最后更新：** 2025-10-13
**版本：** v1.0

