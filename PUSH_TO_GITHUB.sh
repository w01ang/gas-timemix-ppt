#!/bin/bash
# 推送到GitHub新仓库的脚本

echo "📋 准备推送到 GitHub..."
echo ""
echo "⚠️  请先在浏览器中完成以下步骤："
echo "   1. 访问 https://github.com/new"
echo "   2. Repository name: gas-timemix-ppt"
echo "   3. Description: TimeMixer井生命周期预测实验 - 8:2比例时序预测模型"
echo "   4. 选择 Public 或 Private"
echo "   5. ⚠️ 不要勾选 'Initialize with README'"
echo "   6. 点击 'Create repository'"
echo ""
read -p "按回车键继续（确认已创建仓库）..." 

echo ""
echo "📝 请输入你的GitHub用户名:"
read GITHUB_USERNAME

if [ -z "$GITHUB_USERNAME" ]; then
    echo "❌ 错误: 用户名不能为空"
    exit 1
fi

echo ""
echo "🔄 更新远程仓库配置..."
cd /Users/wangjr/Documents/yk/timemixer/timemixer-ppt/gas-timemix

# 移除原来的远程仓库
git remote remove origin 2>/dev/null || true

# 添加新的远程仓库
git remote add origin https://github.com/${GITHUB_USERNAME}/gas-timemix-ppt.git

echo ""
echo "📦 提交最新的GitHub设置文档..."
git add GITHUB_SETUP.md PUSH_TO_GITHUB.sh
git commit -m "Add GitHub setup guide and push script" 2>/dev/null || echo "No changes to commit"

echo ""
echo "🚀 推送到GitHub..."
echo "仓库地址: https://github.com/${GITHUB_USERNAME}/gas-timemix-ppt.git"
echo ""

# 推送到main分支
if git push -u origin main; then
    echo ""
    echo "✅ 成功推送到GitHub!"
    echo "🔗 访问你的仓库: https://github.com/${GITHUB_USERNAME}/gas-timemix-ppt"
    echo ""
    echo "📚 其他人可以这样克隆和使用:"
    echo "   git clone https://github.com/${GITHUB_USERNAME}/gas-timemix-ppt.git"
    echo "   cd gas-timemix-ppt"
    echo "   conda create -n timemixer python=3.10"
    echo "   conda activate timemixer"
    echo "   pip install -r requirements.txt"
    echo "   # 参考 EXPERIMENT_GUIDE.md 运行实验"
else
    echo ""
    echo "❌ 推送失败!"
    echo ""
    echo "可能的原因："
    echo "1. 需要认证 - 使用 Personal Access Token 或 SSH"
    echo "2. 仓库未创建 - 请先在 GitHub 上创建仓库"
    echo "3. 网络问题 - 检查网络连接"
    echo ""
    echo "📖 详细解决方案请参考: GITHUB_SETUP.md"
    exit 1
fi
