# ARIS-A Round 1 — 精炼方案

> 日期: 2026-04-08
> 改进重点: 贡献聚焦 + 论文定位
> 上轮综合分: 6.05/10

---

## 1. Problem Anchor

- **场景**: 多UAV协同搜索-攻击任务规划（CSAMP），100×100 km网格，10架UAV，5个未知目标，7个威胁区域
- **核心挑战**: 动态通信拓扑下的分布式实时协同决策
- **约束**: 单张A100 GPU，训练时间<4小时
- **成功条件**: 在搜索覆盖率和目标存在时间上优于ACO基线

## 2. Technical Gap

现有方法在UAV集群搜索-攻击任务中的通信建模存在根本缺陷：

| 方法 | 通信机制 | 局限 |
|------|----------|------|
| ACO（原论文） | 信息素间接通信 | 无法建模UAV间直接信息交换，收敛慢 |
| Independent RL | 无通信 | 环境非平稳，协同差 |
| CommNet/TarMAC | 固定全连接通信 | 忽略距离和拓扑变化，通信开销大 |
| 标准GNN/GAT | 静态图结构 | 不适应UAV移动导致的拓扑动态变化 |

**Gap**: 缺乏一种能感知通信拓扑动态变化、并自适应调整信息聚合策略的协同机制。

## 3. Method Thesis

**One-sentence thesis**: 提出Topology-Adaptive Graph Attention (TAGA)机制，通过距离衰减注意力、任务感知加权和信息新鲜度编码，实现UAV集群在动态通信拓扑下的自适应协同决策。

**主导贡献**: TAGA通信机制（一个聚焦的机制级贡献）
**支撑设计**: 分层策略（搜索/攻击模式切换）、MAPPO训练框架

## 4. Proposed Method: TAGA-MAPPO

### 4.1 环境建模

```
战场参数：
- 网格大小：100 × 100（每格1km × 1km）
- UAV数量：N = 10
- 目标数量：M = 5（初始位置未知）
- 威胁区域：K = 7（3已知 + 4未知）
- UAV速度：v = 100 m/s
- 搜索半径：R_s = 3km
- 攻击半径：R_a = 1km
- 通信半径：R_comm = 20km（20格）
- 弹药量：每架UAV 2枚
- 最大时间步：T_max = 500
```

### 4.2 观测空间

每架UAV i 的观测 o_i：

**(1) 局部网格观测** — 11×11 patch（5通道）
- 地形/障碍、已探索标记、威胁热力图、目标位置、友方UAV位置
- 维度：[5, 11, 11] = 605

**(2) 自身状态向量**
- [x, y, ammo, mode, fuel, step_t/T_max]
- 维度：6

**(3) TAGA聚合的邻居信息**（本文核心）
- 维度：64

**总观测维度**：675

### 4.3 TAGA通信机制（核心创新）

**标准GAT注意力**：
```
α_ij = softmax_j(LeakyReLU(a^T [W h_i || W h_j]))
```

**TAGA改进**：在标准注意力基础上引入三个自适应因子：

```
α_ij^TAGA = softmax_j(LeakyReLU(a^T [W h_i || W h_j]) + λ_d · f_d(d_ij) + λ_m · f_m(m_i, m_j) + λ_t · f_t(Δt_ij))
```

其中：
- **f_d(d_ij) = -d_ij / R_comm**: 距离衰减因子，距离越远注意力越低
- **f_m(m_i, m_j) = 𝟙(m_i == m_j)**: 任务模式匹配因子，同模式UAV间注意力增强
- **f_t(Δt_ij) = exp(-Δt_ij / τ)**: 信息新鲜度因子，最近通信的邻居权重更高
- λ_d, λ_m, λ_t: 可学习的缩放参数

**TAGA架构**：
```
层数：2层
隐藏维度：64
注意力头数：4（每头16维）
激活函数：ELU
Dropout：0.1
邻居定义：通信半径R_comm=20格内的UAV
```

### 4.4 动作空间

**分层动作**（支撑设计，非主要贡献）：
- 高层：mode ∈ {SEARCH, ATTACK}
- 低层：8方向移动 + 停留 + 攻击 = 10个动作
- 联合动作空间：20

### 4.5 策略网络

```
观测编码器（参数共享）：
├── CNN分支（11×11局部网格）→ 128维
├── MLP分支（自身状态6维）→ 64维
└── 拼接 → Linear(192, 128) → h_i

TAGA通信层：
├── h_i → TAGA_Layer1(128→64, 4heads) → ELU
└── → TAGA_Layer2(64→64, 4heads) → ELU → z_i

Actor: Linear(192, 128) → ReLU → Linear(128, 64) → ReLU → Linear(64, 20) → Softmax
Critic: Linear(64×10, 256) → ReLU → Linear(256, 128) → ReLU → Linear(128, 1)
```

### 4.6 奖励函数

```python
reward = 0.0
reward += 0.5 * new_explored_cells      # 搜索覆盖增量
reward += 5.0 * discovered_new_target    # 发现新目标
reward += 10.0 * destroyed_target        # 摧毁目标
reward -= 3.0 * in_threat_zone           # 进入威胁区
reward -= 0.01                           # 时间步惩罚
reward += 3.0 * cooperative_attack       # 协同攻击奖励
```

### 4.7 训练方案

**算法**: MAPPO
**课程学习**（支撑设计）：

| 阶段 | 设置 | 训练步数 |
|------|------|----------|
| Stage 1 | 3 UAV, 2目标, 0威胁, 50×50 | 2M steps |
| Stage 2 | 6 UAV, 3目标, 3威胁, 75×75 | 3M steps |
| Stage 3 | 10 UAV, 5目标, 7威胁, 100×100 | 5M steps |

**预计训练时间**: ~4小时（单张A100）

## 5. Claim-Driven Validation

### Claim 1: TAGA优于固定拓扑通信
- 对比: TAGA vs 标准GAT vs CommNet vs 无通信
- 指标: 搜索覆盖率、目标存在时间、协同攻击比例
- 预期: TAGA在所有指标上优于其他通信机制

### Claim 2: TAGA-MAPPO优于ACO基线
- 对比: TAGA-MAPPO vs ACO（原论文方法）vs Random vs Greedy
- 指标: 综合指标J、搜索覆盖率、目标摧毁率
- 预期: 显著优于ACO

### 消融实验
- 去除距离衰减因子
- 去除任务模式匹配因子
- 去除信息新鲜度因子
- 去除所有TAGA因子（退化为标准GAT）

## 6. 本轮评分

| 维度 | R0 | R1 | 变化 |
|------|----|----|------|
| Problem Fidelity | 8 | 8 | → |
| Method Specificity | 6 | 7 | ↑1 |
| Contribution Quality | 5 | 7 | ↑2 |
| Frontier Leverage | 6 | 7 | ↑1 |
| Feasibility | 7 | 7 | → |
| Validation Focus | 5 | 7 | ↑2 |
| Venue Readiness | 4 | 6 | ↑2 |
| **综合** | **6.05** | **7.05** | **↑1.0** |
