#!/bin/bash
# ===========================================
# Reinforcement Learning Universal Env Setup
# ===========================================

# 1️⃣ 检查是否已安装 Python3
if ! command -v python3 &> /dev/null
then
    echo "❌ Python3 未安装，请先安装 Python 3.8~3.10。"
    exit
fi

# 2️⃣ 创建虚拟环境
echo "🚀 创建虚拟环境..."
python3 -m venv rl_env

# 3️⃣ 激活虚拟环境
source rl_env/bin/activate

# 4️⃣ 升级 pip
pip install --upgrade pip

# 5️⃣ 安装核心依赖
echo "📦 正在安装通用包..."
pip install numpy==1.24.4 matplotlib==3.8.0 gym==0.26.2 gymnasium==0.29.1 torch==2.2.0 stable-baselines3==2.3.0 tqdm==4.66.1

# 6️⃣ 可选安装可视化/额外工具
pip install seaborn pandas

# 7️⃣ 生成 requirements.txt
echo "🧾 生成 requirements.txt..."
pip freeze > requirements.txt

echo "✅ 虚拟环境已创建完毕！"
echo "➡️ 使用方式："
echo "   source rl_env/bin/activate"
