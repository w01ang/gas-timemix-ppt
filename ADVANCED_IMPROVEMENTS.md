# TimeMixer 深度改进方法指南

根据当前4个版本的训练结果，提供系统性改进方案。

---

## 📊 当前最佳基线

**v2 (step50_bs16)** 是目前表现最好的版本：
- 测试损失：**0.390** ✅
- Epoch 17时达到最佳
- 使用默认模型（d_model=256, n_heads=16, e_layers=6）
- 仍有过拟合（差距0.236）

---

## 🚀 改进方向（按优先级排序）

### 1️⃣ 优化现有最佳模型（⭐⭐⭐⭐⭐ 最推荐）

**策略：在v2基础上微调，而不是大幅降低模型容量**

#### 方案A：轻微降低复杂度 + 增强正则化

```bash
python scripts/train_sliding_window.py \
  --model_id slide_v2_optimized \
  --input_len 3000 \
  --output_len 1000 \
  --step_len 50 \
  --batch_size 16 \
  --d_model 192 \
  --n_heads 12 \
  --e_layers 4 \
  --d_layers 2 \
  --d_ff 768 \
  --dropout 0.15 \
  --learning_rate 8e-5 \
  --patience 20 \
  --train_epochs 80 \
  --use_gpu
```

**改进点：**
- 保持较大模型容量（从256→192，仅降低25%）
- 轻微增加dropout（0.1→0.15）
- 保持v2的step_len=50和batch_size=16
- 预期测试损失：**0.35-0.37**

#### 方案B：v2 + L2正则化

需要修改训练脚本添加weight decay：

```python
# 在 exp/exp_long_term_forecasting.py 的 _select_optimizer 中
model_optim = optim.Adam(
    self.model.parameters(),
    lr=self.args.learning_rate,
    weight_decay=1e-5  # 添加L2正则化
)
```

然后运行：
```bash
python scripts/train_sliding_window.py \
  --model_id slide_v2_l2reg \
  --input_len 3000 \
  --output_len 1000 \
  --step_len 50 \
  --batch_size 16 \
  --dropout 0.15 \
  --train_epochs 80 \
  --use_gpu
```

---

### 2️⃣ 数据增强技术（⭐⭐⭐⭐⭐）

#### A. 噪声注入

在数据加载时添加小量噪声：

```python
# 在 data_provider/data_loader.py 的 __getitem__ 中
if self.flag == 'train':
    noise = np.random.normal(0, 0.01, seq_x.shape)
    seq_x = seq_x + noise
```

#### B. 时间窗口抖动

随机调整滑动窗口的起始位置：

```python
# 在生成样本时
if self.flag == 'train':
    jitter = np.random.randint(-10, 10)  # ±10步抖动
    start_idx = max(0, start_idx + jitter)
```

#### C. Mixup增强

混合不同井的数据：

```python
# 在训练循环中
if np.random.rand() < 0.3:  # 30%概率
    lambda_mix = np.random.beta(0.2, 0.2)
    idx = torch.randperm(batch_x.size(0))
    batch_x = lambda_mix * batch_x + (1 - lambda_mix) * batch_x[idx]
    batch_y = lambda_mix * batch_y + (1 - lambda_mix) * batch_y[idx]
```

---

### 3️⃣ 改进学习率策略（⭐⭐⭐⭐）

#### A. 使用CosineAnnealing替代OneCycleLR

```python
# 在训练脚本中
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

scheduler = CosineAnnealingWarmRestarts(
    model_optim,
    T_0=10,  # 每10个epoch重启
    T_mult=2,  # 周期倍增
    eta_min=1e-6
)
```

#### B. 分段学习率

```bash
# 第一阶段：高学习率快速收敛
python scripts/train_sliding_window.py \
  --model_id slide_stage1 \
  --learning_rate 1e-4 \
  --train_epochs 20

# 第二阶段：低学习率精细调优
# 加载stage1的检查点继续训练
python scripts/train_sliding_window.py \
  --model_id slide_stage2 \
  --learning_rate 2e-5 \
  --train_epochs 40 \
  --load_checkpoint checkpoints/slide_stage1/checkpoint.pth
```

---

### 4️⃣ 模型架构改进（⭐⭐⭐⭐）

#### A. 添加残差连接增强

在TimeMixer基础上添加更多skip connections

#### B. 使用不同的注意力机制

尝试以下变体：
- Flash Attention（更高效）
- Linear Attention（降低复杂度）
- Local Attention（只关注局部窗口）

#### C. 层次化预测

先预测粗粒度趋势，再预测细节：

```
Encoder → Coarse Prediction (100步)
       → Fine Prediction (1000步基于粗预测)
```

---

### 5️⃣ 集成学习（⭐⭐⭐⭐）

#### A. 多模型集成

训练5个不同初始化的模型：

```bash
for seed in 1 2 3 4 5; do
  python scripts/train_sliding_window.py \
    --model_id slide_ensemble_seed${seed} \
    --input_len 3000 \
    --output_len 1000 \
    --step_len 50 \
    --batch_size 16 \
    --train_epochs 60 \
    --use_gpu
done
```

预测时取平均：
```python
predictions = []
for seed in range(1, 6):
    model = load_model(f'slide_ensemble_seed{seed}')
    pred = model.predict(x)
    predictions.append(pred)

final_pred = np.mean(predictions, axis=0)
```

**预期提升：5-10%**

#### B. 不同配置集成

```bash
# 模型1：大模型
--d_model 256 --e_layers 6

# 模型2：中模型
--d_model 128 --e_layers 4

# 模型3：小模型
--d_model 64 --e_layers 2

# 加权集成
final_pred = 0.5 * pred1 + 0.3 * pred2 + 0.2 * pred3
```

---

### 6️⃣ 改进损失函数（⭐⭐⭐）

#### A. 加权MAE损失

对远期预测给予更高权重：

```python
def weighted_mae_loss(pred, target):
    # 对后500步的预测权重更高
    weights = torch.linspace(1.0, 2.0, pred.size(1))
    weights = weights.to(pred.device)
    loss = torch.abs(pred - target) * weights
    return loss.mean()
```

#### B. 组合损失

```python
def combined_loss(pred, target):
    mae_loss = F.l1_loss(pred, target)
    mse_loss = F.mse_loss(pred, target)
    # 趋势损失：预测和真实的一阶差分
    trend_loss = F.l1_loss(pred[:, 1:] - pred[:, :-1], 
                           target[:, 1:] - target[:, :-1])
    return mae_loss + 0.1 * mse_loss + 0.2 * trend_loss
```

#### C. 分位数损失

预测多个分位数，提供不确定性估计：

```python
def quantile_loss(pred, target, quantiles=[0.1, 0.5, 0.9]):
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - pred[:, :, i]
        losses.append(torch.max(q * errors, (q - 1) * errors))
    return torch.cat(losses).mean()
```

---

### 7️⃣ 特征工程（⭐⭐⭐）

#### A. 添加时间特征

```python
# 在数据加载时添加
features = [
    'day_of_week',      # 星期几
    'day_of_month',     # 月中第几天
    'week_of_year',     # 年中第几周
    'cumulative_prod',  # 累计产量
    'prod_rate_change', # 产量变化率
]
```

#### B. 滑动窗口统计特征

```python
# 为每个输入序列计算统计特征
stats = {
    'mean': seq.mean(),
    'std': seq.std(),
    'min': seq.min(),
    'max': seq.max(),
    'trend': (seq[-100:].mean() - seq[:100].mean()) / len(seq)
}
```

#### C. 井特征嵌入

为每口井学习独特的嵌入向量：

```python
class WellEmbedding(nn.Module):
    def __init__(self, num_wells, d_model):
        super().__init__()
        self.embedding = nn.Embedding(num_wells, d_model)
    
    def forward(self, well_ids, seq):
        well_emb = self.embedding(well_ids)  # [batch, d_model]
        # 将井嵌入添加到序列中
        return seq + well_emb.unsqueeze(1)
```

---

### 8️⃣ 数据预处理优化（⭐⭐⭐）

#### A. 更好的归一化策略

```python
# 当前使用StandardScaler
# 尝试：
from sklearn.preprocessing import RobustScaler  # 对异常值更鲁棒

# 或者分段归一化
def segment_normalize(data, segment_size=500):
    normalized = []
    for i in range(0, len(data), segment_size):
        segment = data[i:i+segment_size]
        normalized.append((segment - segment.mean()) / segment.std())
    return np.concatenate(normalized)
```

#### B. 数据平滑

```python
from scipy.signal import savgol_filter

# 在训练前平滑原始数据
smoothed = savgol_filter(data, window_length=21, polyorder=3)
```

#### C. 异常值处理

```python
# 检测并修正异常值
def clip_outliers(data, n_std=3):
    mean = data.mean()
    std = data.std()
    lower = mean - n_std * std
    upper = mean + n_std * std
    return np.clip(data, lower, upper)
```

---

### 9️⃣ 多任务学习（⭐⭐⭐）

同时预测多个相关任务：

```python
class MultiTaskTimeMixer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.encoder = TimeMixerEncoder(d_model)
        self.production_head = nn.Linear(d_model, 1)
        self.trend_head = nn.Linear(d_model, 1)
        self.decline_head = nn.Linear(d_model, 1)
    
    def forward(self, x):
        features = self.encoder(x)
        prod = self.production_head(features)
        trend = self.trend_head(features)  # 预测趋势方向
        decline = self.decline_head(features)  # 预测递减率
        return prod, trend, decline

# 损失函数
loss = prod_loss + 0.2 * trend_loss + 0.1 * decline_loss
```

---

### 🔟 超参数优化（⭐⭐⭐）

#### A. 使用Optuna自动搜索

```python
import optuna

def objective(trial):
    # 定义搜索空间
    d_model = trial.suggest_categorical('d_model', [64, 128, 192, 256])
    n_heads = trial.suggest_categorical('n_heads', [4, 8, 12, 16])
    dropout = trial.suggest_float('dropout', 0.05, 0.3)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    
    # 训练并返回验证损失
    model = train_model(d_model, n_heads, dropout, lr)
    return model.best_vali_loss

# 运行优化
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print('最佳参数:', study.best_params)
```

#### B. 网格搜索关键参数

```bash
# 搜索最佳dropout和learning_rate
for dropout in 0.1 0.15 0.2 0.25; do
  for lr in 5e-5 8e-5 1e-4; do
    python scripts/train_sliding_window.py \
      --model_id slide_grid_d${dropout}_lr${lr} \
      --dropout ${dropout} \
      --learning_rate ${lr} \
      --train_epochs 40
  done
done
```

---

## 🎯 推荐实施路线图

### 阶段1：快速改进（1-2天）

```
1. 方案1A：微调v2模型 → 预期提升到0.35-0.37
2. 添加dropout到0.15
3. 尝试不同learning_rate (5e-5, 8e-5)
```

### 阶段2：中期优化（3-5天）

```
4. 实施数据增强（噪声注入）
5. 添加L2正则化
6. 尝试不同学习率策略（CosineAnnealing）
```

### 阶段3：深度优化（1-2周）

```
7. 训练5模型集成 → 预期提升5-10%
8. 改进损失函数（加权MAE）
9. 添加特征工程
10. 使用Optuna自动调参
```

---

## 📊 预期改进效果

| 方法 | 难度 | 预期提升 | 时间成本 |
|------|------|---------|---------|
| 方案1A（微调v2） | 低 | 5-8% | 0.5天 |
| 数据增强 | 中 | 3-5% | 1天 |
| L2正则化 | 低 | 2-3% | 0.5天 |
| 5模型集成 | 低 | 5-10% | 2天 |
| 改进损失函数 | 中 | 3-7% | 1天 |
| 特征工程 | 高 | 5-15% | 3天 |
| 超参数优化 | 中 | 5-10% | 2天 |
| **组合效果** | - | **15-30%** | 1-2周 |

---

## 🔥 立即行动建议

### 今天就可以做的3件事：

1. **运行方案1A**（最快见效）
```bash
python scripts/train_sliding_window.py \
  --model_id slide_v2_optimized \
  --input_len 3000 \
  --output_len 1000 \
  --step_len 50 \
  --batch_size 16 \
  --d_model 192 \
  --n_heads 12 \
  --e_layers 4 \
  --d_layers 2 \
  --d_ff 768 \
  --dropout 0.15 \
  --learning_rate 8e-5 \
  --patience 20 \
  --train_epochs 80 \
  --use_gpu
```

2. **并行训练3个不同随机种子的模型**（准备集成）
```bash
# 终端1
python scripts/train_sliding_window.py --model_id slide_v2_seed1 --input_len 3000 --output_len 1000 --step_len 50 --batch_size 16 --train_epochs 60 --use_gpu

# 终端2  
python scripts/train_sliding_window.py --model_id slide_v2_seed2 --input_len 3000 --output_len 1000 --step_len 50 --batch_size 16 --train_epochs 60 --use_gpu

# 终端3
python scripts/train_sliding_window.py --model_id slide_v2_seed3 --input_len 3000 --output_len 1000 --step_len 50 --batch_size 16 --train_epochs 60 --use_gpu
```

3. **修改训练脚本添加L2正则化**
```python
# 编辑 exp/exp_long_term_forecasting.py
# 找到 _select_optimizer 函数，添加 weight_decay=1e-5
```

---

## 💡 关键洞察

1. **v2已经很好了**：不要过度降低模型容量
2. **集成是最稳妥的提升方式**：5-10%提升几乎是保证的
3. **数据是瓶颈**：如果井数<20，再好的模型也有限
4. **稳定性很重要**：宁愿牺牲1%性能换取稳定训练

---

**立即开始方案1A！这是投入产出比最高的改进！** 🚀

