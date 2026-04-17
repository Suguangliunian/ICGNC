# ARIS-A Round 2 — 诊断报告

> 日期: 2026-04-08
> 对象: Round 1 精炼方案 (TAGA-MAPPO)
> 上轮综合分: 7.05/10

---

## Diagnose: 7维度评分

| 维度 | R1分 | 薄弱点 |
|------|------|--------|
| Problem Fidelity (8) | OK | 通信半径已量化(R_comm=20格)，但丢包/延迟未建模 |
| Method Specificity (7) | 需改进 | TAGA公式已给出但缺少梯度流分析；CNN编码器的感受野与搜索半径不匹配；动作mask机制未定义 |
| Contribution Quality (7) | 需改进 | TAGA三个因子的设计动机需要更强的理论支撑；距离衰减+任务匹配+新鲜度的组合为何是最优的？ |
| Frontier Leverage (7) | 需改进 | 未考虑Transformer-based替代方案的对比论证 |
| Feasibility (7) | OK | 训练时间估算需更精确 |
| Validation Focus (7) | 需改进 | 消融实验设计了但缺少定量预期 |
| Venue Readiness (6) | 需改进 | 论文标题和摘要未拟定 |

**最薄弱维度**: Method Specificity (7) 和 Contribution Quality (7)

---

## 核心问题

1. **TAGA的理论动机不够强**: 三个因子（距离、模式、新鲜度）的组合缺乏理论依据，看起来像是"拼凑"
2. **CNN编码器设计粗糙**: 11×11 patch用3×3卷积，感受野只有5×5，无法覆盖搜索半径(3格)对应的7×7区域
3. **动作合法性处理缺失**: 边界检查、威胁区规避、弹药耗尽后的行为未定义
