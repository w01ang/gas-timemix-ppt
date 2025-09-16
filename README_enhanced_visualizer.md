# Enhanced Well Prediction Visualizer
# 增强井生命周期预测可视化工具

## 功能特点

### 1. 三色曲线图
- **蓝色**：输入段（历史数据，前半段）
- **绿色**：真实输出段（后半段的真实值）
- **橙色**：预测输出段（后半段的预测值）
- **红色虚线**：标记预测起点

### 2. 更长步长预测
- 使用 `pred_len * 3` 的步长进行迭代预测
- 减少预测的平滑度，更好捕捉真实值的剧烈波动
- 支持最大384步的预测步长

### 3. 迭代多块预测
- 每次预测后滑动窗口继续预测
- 总预测长度可达640步或更多
- 自动处理不同长度的井数据

### 4. 数据导出
- PDF格式的高质量图表
- CSV格式的结构化数据
- 支持中英文标签和Times New Roman字体

## 使用方法

### 基本用法

```python
from enhanced_well_prediction_visualizer import create_enhanced_visualizer

# 创建可视化器
visualizer = create_enhanced_visualizer(model, device, args, test_data)

# 执行可视化
visualizer.enhanced_test_visualization(folder_path, one_pred, one_true)
```

### 在TimeMixer测试中集成

```python
# 在 exp_long_term_forecasting.py 的 test() 方法中添加
from enhanced_well_prediction_visualizer import create_enhanced_visualizer

# 在测试循环中
visualizer = create_enhanced_visualizer(self.model, self.device, self.args, test_data)
visualizer.enhanced_test_visualization(folder_path, one_pred, one_true)
```

## 输出文件

### 主要输出
- `one_well_enhanced_3color.pdf` - 三色对比图（主要结果）
- `one_well_enhanced_3color.csv` - 三色数据CSV

### 数据结构
CSV文件包含三列：
- `input_segment`: 输入段数据（后半段为NaN）
- `true_output`: 真实输出段数据（前半段为NaN）
- `pred_output`: 预测输出段数据（前半段为NaN）

## 技术参数

### 可配置参数
- `step_len`: 预测步长（默认：pred_len * 3，最大384）
- `max_pred_len`: 最大预测长度（默认：640）
- `figure_size`: 图表尺寸（默认：15x8）
- `line_width`: 线条宽度（默认：2.5）

### 依赖要求
- PyTorch
- NumPy
- Pandas
- Matplotlib
- 支持Times New Roman字体的系统

## 示例输出

```
开始增强可视化：步长=384, 总长度=1024
  预测步骤 1: 当前步长=384, 剩余=512
  预测步骤 2: 当前步长=128, 剩余=128
  预测完成：总预测长度=512
  三色图已保存: ./test_results/.../one_well_enhanced_3color.pdf
  三色数据已保存: ./test_results/.../one_well_enhanced_3color.csv
增强可视化完成：步长=384, 总预测长度=512
```

## 注意事项

1. 确保在正确的conda环境中运行
2. 模型需要支持动态预测长度
3. 数据需要预先标准化
4. 输出目录需要有写入权限

## 更新历史

- v1.0: 初始版本，支持三色可视化和长步长预测
- 基于TimeMixer框架的井生命周期预测任务
