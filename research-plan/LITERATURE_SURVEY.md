# 文献综述：无人机集群智能任务规划前沿研究

> 调研日期：2026-04-08 | 覆盖时间范围：2023–2026

---

## 目录

1. [无人机集群协同任务规划（深度学习方法）](#1-无人机集群协同任务规划深度学习方法)
2. [多智能体强化学习用于无人机协调](#2-多智能体强化学习用于无人机协调)
3. [大语言模型用于多智能体决策](#3-大语言模型用于多智能体决策)
4. [图注意力网络用于多机器人任务分配](#4-图注意力网络用于多机器人任务分配)
5. [基于 Transformer 的无人机路径规划](#5-基于-transformer-的无人机路径规划)
6. [专题分析](#6-专题分析)
7. [总结与研究方案启示](#7-总结与研究方案启示)

---

## 1. 无人机集群协同任务规划（深度学习方法）

### 1.1 综述文献

**[S1] Alqudsi & Makaraci, "UAV swarms: research, challenges, and future directions," J. Eng. Appl. Sci., vol. 72, no. 12, Jan. 2025.**
- 综合综述了无人机集群的协同路径规划、任务分配、编队控制和安全问题
- 重点分析了 AI/ML 在集群决策中的集成方式，包括 DRL、群体智能算法
- 指出未来方向：利用 AI/ML 提升集群决策能力，DL 算法处理大规模数据并从经验中学习

**[S2] Dimos et al., "A Survey on UxV Swarms and the Role of AI as a Technological Enabler," Drones, vol. 9, no. 10, Oct. 2025.**
- 以 AI 为核心视角审视无人系统集群，覆盖路径规划、编队控制、任务规划
- 讨论了 DRL（DDPG、DQN、MARL）在集群任务规划中的应用
- 提出混合方法（DRL + 传统优化）是当前最有前景的方向

### 1.2 代表性工作

**[P1] Guo et al., "Multi-UAV Cooperative Multi-objective Task Allocation Based on DRL," ICGNC 2024, LNEE vol. 1348, Springer, Mar. 2025.**
- 方法：结合 GNN + 注意力机制建模策略函数，基于 RL 进行多目标任务分配
- 贡献：策略可泛化到不同数量的敌方目标节点；提高了训练效率和稳定性
- 🔑 直接来自 ICGNC 会议，与本研究高度相关
- 💻 单 GPU 可训练（GNN + Attention 策略网络规模适中）

**[P2] Yang et al., "Role-Structured Multi-Agent Pursuit–Evasion with Potential Game Constraints for Heterogeneous Airship–UAV Systems," Drones, vol. 10, no. 4, Mar. 2026.**
- 方法：势博弈约束的角色结构化 MARL，多头注意力处理异构智能体 token，两阶段任务分配求解器
- 贡献：将多智能体交互分解为追捕者内部势博弈 + 对抗目标的一般和博弈
- 来自北航无人系统研究所，Gazebo 仿真验证策略可迁移
- 💻 CTDE 架构，单 GPU 可训练

---

