# TimeMixer 多尺度季节-趋势配置说明

## 📊 核心参数

### 1. 季节-趋势分解参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `moving_avg` | 1000 | 移动平均窗口大小，用于季节-趋势分解 |

**作用：** 
- 决定如何分离趋势和季节性成分
- `Trend = MovingAvg(X, window=moving_avg)`
- `Season = X - Trend`

**推荐值：**
- 对于3000步输入：`moving_avg=1000` (约1/3输入长度)
- 对于其他输入长度：约为输入长度的 20%-40%

### 2. 多尺度下采样参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `down_sampling_layers` | 2 | 下采样层数，决定产生多少个尺度 |
| `down_sampling_window` | 2 | 下采样窗口大小 |
| `down_sampling_method` | 'avg' | 下采样方法 (avg/max/conv) |

**作用：**
- 产生不同粒度的时间序列
- 层数N产生N+1个尺度
- 每层将序列长度缩减为 `1/down_sampling_window`

**推荐值：**
- `down_sampling_layers=2` → 产生3个尺度 (3000 → 1500 → 750)
- `down_sampling_window=2` → 每层缩减1/2 (温和)
- `down_sampling_window=3` → 每层缩减1/3 (激进)

---

## 🎯 尺度层级示例

### 配置: moving_avg=1000, layers=2, window=2

**输入序列长度: 3000步**

```
尺度0 (原始细节):
  长度: 3000步
  Trend: MovingAvg(X₀, 1000)
  Season: X₀ - Trend
  捕捉: 日级波动、周级模式

  ↓ AvgPool(window=2)

尺度1 (中期模式):
  长度: 1500步 (每2步取平均)
  Trend: MovingAvg(X₁, 1000)  
  Season: X₁ - Trend
  捕捉: 月级季节、季度变化

  ↓ AvgPool(window=2)

尺度2 (长期趋势):
  长度: 750步 (每4步取平均)
  Trend: MovingAvg(X₂, 1000)
  Season: X₂ - Trend
  捕捉: 年度衰减、生命周期
```

**产生3个尺度，覆盖短中长期模式**

---

## 🚀 使用方法

### 1. 使用默认配置（推荐）

```bash
python scripts/train_sliding_window.py \
  --model_id slide_3scales \
  --input_len 3000 \
  --output_len 1000 \
  --batch_size 16 \
  --train_epochs 60
```

默认自动使用：
- `moving_avg=1000`
- `down_sampling_layers=2`
- `down_sampling_window=2`

### 2. 自定义配置

```bash
python scripts/train_sliding_window.py \
  --model_id slide_custom \
  --input_len 3000 \
  --output_len 1000 \
  --moving_avg 500 \
  --down_sampling_layers 3 \
  --down_sampling_window 2 \
  --batch_size 16 \
  --train_epochs 60
```

### 3. 查看配置信息

运行训练时会自动显示：

```
📊 滑动窗口配置:
   输入长度 (input_len): 3000
   输出长度 (output_len): 1000
   ...

🎨 季节-趋势分解配置:
   移动平均窗口 (moving_avg): 1000
   下采样层数 (down_sampling_layers): 2
   下采样窗口 (down_sampling_window): 2
   产生的尺度层级: 3000 → 1500 → 750 (3个尺度)
   季节周期占比: 1000/3000 = 33.3%
```

---

## 💡 参数调优指南

### 场景1: 输入长度变化

| 输入长度 | 推荐 moving_avg | 说明 |
|---------|----------------|------|
| 1000 | 300-400 | 30%-40% |
| 2000 | 600-800 | 30%-40% |
| 3000 | 1000 | 33% ✓ |
| 5000 | 1500-2000 | 30%-40% |

### 场景2: 需要更多尺度

```bash
# 4个尺度: 3000 → 1500 → 750 → 375
--down_sampling_layers 3 \
--down_sampling_window 2
```

**注意：** 尺度太多可能导致：
- 最粗尺度信息损失过多
- 计算量显著增加
- 训练不稳定

### 场景3: 更激进的下采样

```bash
# 3个尺度: 3000 → 1000 → 333
--down_sampling_layers 2 \
--down_sampling_window 3
```

**适合：** 长期依赖性强的场景

---

## 📐 数学原理

### 1. 季节-趋势分解

```python
# 对于每个尺度 i
Trend_i = MovingAvg(X_i, window=moving_avg)
Season_i = X_i - Trend_i
```

### 2. 多尺度下采样

```python
# 产生N+1个尺度
X_0 = 原始序列
X_1 = AvgPool(X_0, window)
X_2 = AvgPool(X_1, window)
...
X_N = AvgPool(X_{N-1}, window)
```

### 3. 模型处理流程

```
输入 X (3000步)
  ↓
多尺度分解 → [X₀, X₁, X₂]
  ↓
季节-趋势 → [(Trend₀, Season₀), (Trend₁, Season₁), (Trend₂, Season₂)]
  ↓
编码器 → Transformer(Season_i) + Cross-Attention
  ↓
解码器 → 每个尺度独立预测 + 上采样
  ↓
融合 → 最终预测 (1000步)
```

---

## ⚠️ 注意事项

1. **moving_avg 不宜过小**
   - 太小：无法有效分离趋势和季节
   - 推荐：至少为输入长度的20%

2. **down_sampling_layers 不宜过多**
   - 太多：最粗尺度信息损失严重
   - 推荐：2-3层

3. **训练和测试配置必须一致**
   - 所有测试脚本会自动使用训练时的配置
   - 修改训练配置后需重新训练

4. **观察训练日志**
   - 如果loss异常，可能是moving_avg设置不当
   - 建议先用默认配置，再微调

---

## 📚 相关文档

- [滑动窗口使用指南](SLIDING_WINDOW_GUIDE.md)
- [参数调优指南](TUNING_GUIDE.md)
- [NaN问题修复](FIX_NAN_LOSS.md)
- [可视化说明](timemixer_decomposition_explanation.png)

---

## 🔬 可视化示例

已生成可视化图 `timemixer_decomposition_explanation.png`，展示了：
1. 原始序列
2. 提取的趋势
3. 提取的季节性
4. 尺度1下采样结果
5. 尺度2下采样结果

查看该图可以直观理解多尺度分解过程。

---

## 📝 更新日志

### v4.0 (2025-10-13)
- ✅ 添加命令行参数支持 (`--moving_avg`, `--down_sampling_layers`, `--down_sampling_window`)
- ✅ 更新默认配置为推荐值 (moving_avg=1000, layers=2)
- ✅ 添加配置信息自动显示
- ✅ 同步更新所有训练和测试脚本

### v3.0 之前
- 使用固定配置 (moving_avg=49, layers=1)

