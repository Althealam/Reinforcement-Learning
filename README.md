# 🧠 Reinforcement Learning Framework | 强化学习框架

A structured, bilingual collection of reinforcement learning algorithms implemented from scratch, organized by methodology.

---

## 1️⃣ Value-based Methods | 基础价值迭代类

**Dynamic Programming (DP) | 动态规划**
- Policy Evaluation ☑️
- Policy Iteration ☑️
- Value Iteration ☑️  

**Monte Carlo Methods | 蒙特卡罗方法**

**Temporal Difference (TD) | 时序差分法**
- TD(0) ☑️  
- TD(n) ☑️
- TD(λ) ☑️  

**Q-learning Family | Q-learning 系列**
- Q-learning (off-policy) ☑️ 
- SARSA (on-policy) ☑️  
- Expected SARSA ☑️ 
- Double Q-learning ☑️ 

**Deep Value Networks | 深度价值网络**
- DQN ☑️ 
- Double DQN  
- Dueling DQN  
- Rainbow DQN  

---

## 2️⃣ Policy-based Methods | 策略梯度类

- **REINFORCE (Vanilla Policy Gradient) | 纯策略梯度**  
- **Baseline Methods | 基线法**  
- **Actor-Critic Methods | Actor-Critic**
  - A2C / A3C  
  - DDPG (Continuous Action Space | 连续动作空间)  
  - TD3  
  - SAC (Soft Actor-Critic)

---

## 3️⃣ Model-based Methods | 模型驱动类

- Dyna-Q  
- World Models  
- MBPO (Model-Based Policy Optimization)

---

## 4️⃣ Advanced / Modern RL | 高级优化类

- PPO (Proximal Policy Optimization)  
- TRPO (Trust Region Policy Optimization)  
- GAE (Generalized Advantage Estimation)  
- DPO / GRPO (Reward Model Alignment)

---

## 5️⃣ Multi-Agent & Hierarchical RL | 多智能体 / 层级类

- Multi-Agent Q-learning  
- MADDPG  
- HRL (Options Framework)

---

## 6️⃣ Application Domains | 应用场景类

- 🎮 **Game Environments**: CartPole / LunarLander / Mario / Atari  
  （游戏环境）  
- 🧩 **Recommender Systems / Advertising Bidding Optimization**  
  （推荐系统 / 广告出价优化）  
- 🤖 **Robotics / Control** （机器人 / 控制）  
- 💰 **Finance / Trading Strategies** （金融 / 交易策略）  
- ✍️ **Text Generation / LLM Fine-Tuning (RLHF / PPO)** （文本生成 / LLM 调优）

---

## 📘 Notes

- All implementations follow the same environment and logging conventions.  
- Each subfolder contains an independent example (`.py` file) with documentation and visualization.  
- This repository aims to serve as a clean educational reference for RL learners and practitioners.

---

⭐ **Author:** Althea Lam  
📚 **Language:** Python 3.9+  
🏗️ **Frameworks:** NumPy · Gym · PyTorch  
📄 **License:** MIT  
