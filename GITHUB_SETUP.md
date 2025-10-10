# GitHub 仓库创建与推送指南

## 步骤 1: 在GitHub上创建新仓库

1. 访问 https://github.com/new
2. 填写仓库信息：
   - **Repository name**: `gas-timemix-ppt`
   - **Description**: `TimeMixer井生命周期预测实验 - 8:2比例时序预测模型`
   - **Visibility**: Public 或 Private（根据需要选择）
   - ⚠️ **不要**勾选 "Initialize this repository with a README"
   - ⚠️ **不要**添加 .gitignore 或 license（已有本地文件）
3. 点击 "Create repository"

## 步骤 2: 推送到新仓库

在终端执行以下命令：

```bash
cd /Users/wangjr/Documents/yk/timemixer/timemixer-ppt/gas-timemix

# 移除原来的远程仓库
git remote remove origin

# 添加新的远程仓库（替换YOUR_USERNAME为你的GitHub用户名）
git remote add origin https://github.com/YOUR_USERNAME/gas-timemix-ppt.git

# 推送代码到新仓库
git push -u origin main
```

## 步骤 3: 验证推送

访问你的新仓库页面：
```
https://github.com/YOUR_USERNAME/gas-timemix-ppt
```

应该能看到：
- ✅ EXPERIMENT_GUIDE.md（实验指南）
- ✅ README.md（项目说明）
- ✅ 代码文件（scripts/、models/、exp/等）
- ✅ .gitignore（已排除大文件）

## 可选：添加数据样本

如果需要分享数据样本（注意不要推送大文件）：

```bash
# 1. 创建一个小的数据样本
cd data
head -100 preprocessed_daily_gas_by_well.csv > sample_data.csv

# 2. 更新 .gitignore，允许 sample_data.csv
echo "!data/sample_data.csv" >> ../.gitignore

# 3. 提交并推送
cd ..
git add data/sample_data.csv .gitignore
git commit -m "Add sample data for testing"
git push
```

## 故障排查

### 问题1: 推送时要求认证

**解决方案A: 使用Personal Access Token (推荐)**
```bash
# 1. 生成token: https://github.com/settings/tokens
# 2. 勾选 "repo" 权限
# 3. 推送时使用token作为密码
```

**解决方案B: 使用SSH**
```bash
# 1. 检查SSH密钥
ls ~/.ssh/id_rsa.pub

# 2. 如果没有，生成SSH密钥
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

# 3. 添加到GitHub: https://github.com/settings/keys

# 4. 修改远程仓库URL为SSH
git remote set-url origin git@github.com:YOUR_USERNAME/gas-timemix-ppt.git

# 5. 推送
git push -u origin main
```

### 问题2: 推送被拒绝（non-fast-forward）

```bash
# 强制推送（谨慎使用）
git push -u origin main --force
```

### 问题3: 文件过大

```bash
# 查看最大的文件
find . -type f -size +10M

# 如果有大文件意外提交，从历史中移除
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch path/to/large/file' \
  --prune-empty --tag-name-filter cat -- --all
```

## 完成后的仓库结构

```
gas-timemix-ppt/
├── EXPERIMENT_GUIDE.md          ← 实验指南（新增）
├── README.md                     ← 项目说明
├── LICENSE                       ← Apache 2.0许可证
├── .gitignore                    ← 已排除大文件
├── requirements.txt              ← Python依赖
├── scripts/                      ← 训练和测试脚本
│   ├── train_8_2_ratio.py
│   ├── test_and_visualize.py
│   └── ...
├── models/                       ← TimeMixer模型定义
├── exp/                          ← 实验类
├── layers/                       ← 模型层组件
├── data_provider/                ← 数据加载器
├── utils/                        ← 工具函数
├── checkpoints/                  ← 模型权重（被.gitignore）
├── experiments/                  ← 实验配置（被.gitignore）
├── results_archive/              ← 结果输出（被.gitignore）
└── logs/                         ← 训练日志（被.gitignore）
```

## 克隆和使用

其他人可以这样使用你的仓库：

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/gas-timemix-ppt.git
cd gas-timemix-ppt

# 创建环境
conda create -n timemixer python=3.10
conda activate timemixer
pip install -r requirements.txt

# 准备数据
mkdir -p data
# 将数据文件放到 data/ 目录

# 按照 EXPERIMENT_GUIDE.md 运行实验
```

---

**提示**: 记得将 `YOUR_USERNAME` 替换为你的实际GitHub用户名！

