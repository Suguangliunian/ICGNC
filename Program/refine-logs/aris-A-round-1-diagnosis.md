# ARIS-A Round 1 — 诊断报告

> 日期: 2026-04-08
> 对象: MADRL-GAT 基线方案 (round-0-initial-proposal.md)

---

## Diagnose: 7维度评分

| 维度 | 分数 | 评价 |
|------|------|------|
| 1. Problem Fidelity | 8/10 | 场景定义完整（100×100网格，10UAV，5目标，7威胁），但通信模型未明确（通信距离、丢包率） |
| 2. Method Specificity | 6/10 | GAT架构有基本参数，但邻居定义"通信范围内"未量化；策略网络有结构但缺少关键实现细节（如动作mask、无效动作处理）；奖励函数缺少归一化和稀疏奖励处理 |
| 3. Contribution Quality | 5/10 | 贡献点过多（GAT通信+分层策略+课程学习+实验验证=4个），缺乏一个主导贡献；GAT用于MARL通信并非新颖（QMIX、CommNet已有类似工作） |
| 4. Frontier Leverage | 6/10 | MAPPO+GAT是合理的现代组合，但缺乏与最新MARL进展的对比（如HAPPO、MAPPO-Lagrangian）；未考虑Transformer-based通信 |
| 5. Feasibility | 7/10 | 单A100可训练，但10M步训练量的时间估算不够精确；课程学习3阶段的切换条件未定义 |
| 6. Validation Focus | 5/10 | 评估指标定义了但缺少baseline对比方案的具体实现；缺少消融实验设计 |
| 7. Venue Readiness | 4/10 | 缺少Related Work定位；贡献散焦；论文结构未规划 |

**综合分数**: 0.15×8 + 0.25×6 + 0.25×5 + 0.15×6 + 0.10×7 + 0.05×5 + 0.05×4 = **6.05/10**

---

## 薄弱点排序

| 优先级 | 维度 | 问题 | 严重程度 |
|--------|------|------|----------|
| 1 | Contribution Quality (5) | 贡献点过多且散焦，GAT+MARL不够新颖 | CRITICAL |
| 2 | Venue Readiness (4) | 缺少论文定位和结构规划 | CRITICAL |
| 3 | Validation Focus (5) | 缺少消融实验和具体baseline | IMPORTANT |
| 4 | Method Specificity (6) | 关键实现细节缺失 | IMPORTANT |
| 5 | Frontier Leverage (6) | 未与最新MARL方法对比 | MINOR |

---

## 核心诊断结论

1. **最大问题：贡献散焦**。当前方案试图同时贡献GAT通信、分层策略、课程学习三个点，但每个都不够深入。需要聚焦到一个主导贡献。

2. **GAT用于MARL通信不够新颖**。CommNet (2016)、TarMAC (2019)、QMIX+Attention已有大量工作。需要找到差异化角度——建议聚焦于"动态拓扑下的自适应通信"而非简单的GAT聚合。

3. **缺少与原论文ACO的直接对比设计**。作为替代方案，必须有ACO作为baseline的直接对比实验。

## 下轮方向

Round 2 应聚焦于：
- 将贡献收敛到一个主导点：**动态通信拓扑下的自适应注意力协同机制**
- 明确通信模型参数
- 设计与ACO的直接对比实验
