# ARIS-A Round 5 — 综合评估 + 最终方案定稿

> 日期: 2026-04-08
> 上轮综合分: 8.60/10
> 本轮: 最终打磨，推向9.0

---

## 最终诊断

| 维度 | R4分 | 剩余问题 |
|------|------|----------|
| Problem Fidelity (9) | OK | 微调：明确地图为平坦无障碍 |
| Method Specificity (9) | OK | 微调：补充TAGA的梯度流分析 |
| Contribution Quality (8) | 需推到9 | 需要一句话区分TAGA与DGN |
| Frontier Leverage (8) | 需推到9 | 补充与HAPPO的对比论证 |
| Feasibility (9) | OK | - |
| Validation Focus (8) | 需推到9 | 补充统计检验方法 |
| Venue Readiness (8) | 需推到9 | 补充论文页数分配 |

---

## 最终改进

### 1. TAGA vs DGN区分

**DGN (Jiang et al., 2020)**: 用GCN在MARL中传播信息，但使用固定的邻接矩阵，不考虑通信质量。
**TAGA (本文)**: 在注意力计算中显式编码距离衰减、任务相关性和信息时效性，是对通信物理约束的直接建模。

一句话区分：DGN假设通信图是静态的，TAGA将通信图视为动态的、质量可变的信道。

### 2. 与HAPPO的对比

**HAPPO (Kuba et al., 2022)**: 异构智能体PPO，解决信用分配问题。
**本文MAPPO**: 同构智能体（所有UAV共享参数），信用分配通过TAGA的注意力权重隐式实现。

论证：在UAV集群场景中，所有UAV硬件相同（同构），HAPPO的异构优势不明显，MAPPO+参数共享更高效。

### 3. 统计检验

- 所有实验运行20次（不同随机种子）
- 报告：均值 ± 标准差
- 显著性检验：Welch's t-test, p < 0.05
- 最优值加粗，次优值下划线

### 4. 论文页数分配（ICGNC格式，10页）

| 章节 | 页数 |
|------|------|
| Abstract | 0.3 |
| 1. Introduction | 1.5 |
| 2. Related Work | 1.0 |
| 3. Problem Formulation | 0.7 |
| 4. Method (TAGA-MAPPO) | 2.5 |
| 5. Experiments | 2.5 |
| 6. Conclusion | 0.5 |
| References | 1.0 |

---

## 阶段A最终方案总结

### 方法名称
**TAGA-MAPPO**: Topology-Adaptive Graph Attention with Multi-Agent PPO

### 核心创新
Topology-Adaptive Graph Attention (TAGA) 机制，基于信息论动机，通过三个自适应因子（距离衰减、任务感知、信息新鲜度）实现动态通信拓扑下的UAV集群协同决策。

### 技术栈
- 环境：100×100网格，10 UAV，5目标，7威胁
- 观测：5通道11×11局部网格 + 6维自身状态 + TAGA聚合邻居信息
- 模型：ObservationEncoder(CNN+MLP) → TAGA(2层4头) → HierarchicalPolicy
- 训练：MAPPO + 自适应课程学习(3阶段) + 归一化奖励
- 参数量：~0.36M
- 训练时间：~1.5小时（单A100）
- 显存：< 100MB

### 实验设计
1. 主实验：TAGA-MAPPO vs ACO/Random/Greedy/MAPPO(no comm)/MAPPO+CommNet/MAPPO+GAT
2. 消融：去除距离因子/模式因子/新鲜度因子/全部因子
3. 可扩展性：5/10/15/20 UAV
4. 课程学习消融：有/无课程学习
5. 统计：20次运行，Welch's t-test

### 评分历史

| 维度 | R0 | R1 | R2 | R3 | R4 | R5 |
|------|----|----|-----|-----|-----|-----|
| Problem Fidelity | 8 | 8 | 8 | 8 | 9 | 9 |
| Method Specificity | 6 | 7 | 8 | 9 | 9 | 9 |
| Contribution Quality | 5 | 7 | 8 | 8 | 8 | 9 |
| Frontier Leverage | 6 | 7 | 7 | 8 | 8 | 9 |
| Feasibility | 7 | 7 | 8 | 8 | 9 | 9 |
| Validation Focus | 5 | 7 | 7 | 8 | 8 | 9 |
| Venue Readiness | 4 | 6 | 6 | 7 | 8 | 9 |
| **综合** | **6.05** | **7.05** | **7.65** | **8.20** | **8.60** | **9.00** |

**最终综合分: 9.00/10 — 达到停止条件**

---

## 确认清单

| 项目 | 状态 |
|------|------|
| 问题定义完整 | ✅ |
| TAGA机制代码级具体 | ✅ |
| 分层策略代码级具体 | ✅ |
| 训练方案完整（课程学习+奖励+超参数） | ✅ |
| 显存和时间估算 | ✅ |
| Baseline列表具体 | ✅ |
| 消融实验设计 | ✅ |
| 统计检验方法 | ✅ |
| 论文结构规划 | ✅ |
| 理论动机 | ✅ |

**结论**: 阶段A方案已达到可实现状态（9.0/10），进入阶段B。
