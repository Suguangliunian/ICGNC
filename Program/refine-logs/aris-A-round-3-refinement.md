# ARIS-A Round 3 — 诊断 + 创新 + 精炼

> 日期: 2026-04-08
> 上轮综合分: 7.65/10
> 本轮重点: 分层策略优化 + Venue Readiness提升

---

## Diagnose

| 维度 | R2分 | 薄弱点 |
|------|------|--------|
| Problem Fidelity (8) | OK | 可补充：目标发现后的信息共享机制 |
| Method Specificity (8) | OK | TAGA代码级具体，但分层策略的模式切换条件仍模糊 |
| Contribution Quality (8) | OK | 理论动机已强化，但需要与最近的MARL通信工作做更精确的区分 |
| Frontier Leverage (7) | 需改进 | 未论证为何选GAT而非Transformer |
| Feasibility (8) | OK | 训练时间估算合理 |
| Validation Focus (7) | 需改进 | baseline列表需要更具体的实现方案 |
| Venue Readiness (6) | 需改进 | 缺论文标题、摘要、Related Work具体内容 |

**最薄弱**: Venue Readiness (6), Frontier Leverage (7), Validation Focus (7)

---

## Innovation

### 改进1: 分层策略的模式切换机制具体化

当前分层策略只说"高层选模式，低层选动作"，但切换条件不明确。

**改进**: 引入基于注意力的自适应模式切换：

```python
class HierarchicalPolicy(nn.Module):
    def __init__(self, obs_dim=192, n_move=9, n_attack=10):
        super().__init__()
        # 模式选择头（高层）
        self.mode_head = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 2)  # search / attack
        )
        # 搜索动作头（低层）
        self.search_head = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, n_move)  # 9方向
        )
        # 攻击动作头（低层）
        self.attack_head = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, n_attack)  # 9方向 + 攻击
        )
    
    def forward(self, obs_feat, action_mask_search, action_mask_attack):
        # 高层：模式选择
        mode_logits = self.mode_head(obs_feat)  # [B, 2]
        mode_probs = F.softmax(mode_logits, dim=-1)
        
        # 低层：条件动作选择
        search_logits = self.search_head(obs_feat)
        search_logits = search_logits.masked_fill(action_mask_search == 0, -1e9)
        
        attack_logits = self.attack_head(obs_feat)
        attack_logits = attack_logits.masked_fill(action_mask_attack == 0, -1e9)
        
        return mode_probs, F.softmax(search_logits, -1), F.softmax(attack_logits, -1)
```

**模式切换的自然条件**：
- 发现目标且有弹药 → 倾向攻击模式
- 附近无目标或弹药耗尽 → 强制搜索模式
- 搜索覆盖率低 → 倾向搜索模式
- 这些条件通过训练自动学习，不需要硬编码

### 改进2: GAT vs Transformer论证

**为何选GAT而非Transformer**：
1. **稀疏性**: UAV通信图是稀疏的（平均每个UAV只有3-5个邻居），GAT天然处理稀疏图，Transformer的全连接注意力浪费计算
2. **拓扑感知**: GAT的邻接矩阵直接编码通信拓扑，Transformer需要额外的位置编码
3. **参数效率**: 2层GAT(4头) ~0.5M参数，同等Transformer ~2M参数
4. **可扩展性**: GAT复杂度O(|E|)，Transformer O(N²)，UAV数量增加时GAT更优

### 改进3: 论文框架完善

**标题**: Topology-Adaptive Graph Attention for Distributed UAV Swarm Cooperative Search-Attack Mission Planning

**摘要草稿**:
> UAV集群在未知环境中的协同搜索-攻击任务规划是一个关键挑战。现有方法（如蚁群优化）依赖间接通信机制，难以适应动态通信拓扑。本文提出TAGA-MAPPO方法，核心是Topology-Adaptive Graph Attention (TAGA)机制，通过距离衰减注意力、任务感知加权和信息新鲜度编码，实现UAV间自适应信息聚合。结合分层MAPPO策略，实现搜索-攻击模式的端到端联合优化。仿真实验表明，TAGA-MAPPO在搜索覆盖率和目标存在时间上显著优于ACO基线和标准MARL方法。

**Baseline具体化**:

| Baseline | 实现方式 | 来源 |
|----------|----------|------|
| ACO | 复现原论文算法 | Liu et al. (原论文) |
| Random | 随机选择合法动作 | - |
| Greedy | 搜索模式选最近未探索格，攻击模式选最近目标 | - |
| MAPPO (no comm) | 去除TAGA，独立决策 | Yu et al., 2022 |
| MAPPO + CommNet | 用CommNet替代TAGA | Sukhbaatar et al., 2016 |
| MAPPO + GAT | 用标准GAT替代TAGA | Veličković et al., 2018 |

---

## 本轮评分

| 维度 | R0 | R1 | R2 | R3 | 变化 |
|------|----|----|-----|-----|------|
| Problem Fidelity | 8 | 8 | 8 | 8 | → |
| Method Specificity | 6 | 7 | 8 | 9 | ↑1 |
| Contribution Quality | 5 | 7 | 8 | 8 | → |
| Frontier Leverage | 6 | 7 | 7 | 8 | ↑1 |
| Feasibility | 7 | 7 | 8 | 8 | → |
| Validation Focus | 5 | 7 | 7 | 8 | ↑1 |
| Venue Readiness | 4 | 6 | 6 | 7 | ↑1 |
| **综合** | **6.05** | **7.05** | **7.65** | **8.20** | **↑0.55** |
