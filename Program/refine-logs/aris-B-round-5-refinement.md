# ARIS-B Round 5 — 综合评估 + 最终方案定稿

> 日期: 2026-04-08
> 上轮综合分: 9.05/10
> 本轮: 最终打磨，确认所有细节

---

## 最终方案总结: TAGA-LLM

### 方法名称
**TAGA-LLM**: Topology-Adaptive Graph Attention informed LLM for UAV Swarm Mission Planning

### 核心创新
将TAGA（Topology-Adaptive Graph Attention）的通信拓扑感知思想融入LLM的prompt设计，用Qwen3.5-4B + LoRA微调替代传统优化算法，实现UAV集群的在线自适应搜索-攻击任务规划。

### 与原论文ACO的差异

| 维度 | 原论文ACO | TAGA-LLM |
|------|----------|----------|
| 决策引擎 | 信息素+启发式规则 | Qwen3.5-4B + LoRA |
| 通信建模 | 信息素间接通信 | TAGA-aware prompt编码 |
| 任务切换 | 覆盖率阈值硬切换 | CoT推理自适应切换 |
| 泛化能力 | 需重新调参 | 预训练知识+场景多样化训练 |
| 可解释性 | 无 | CoT推理链 |
| 指令调控 | 无 | 5种任务指令 |

### 技术栈

```
环境: 100×100网格, 10 UAV, 5目标, 7威胁, R_comm=20格
模型: Qwen3.5-4B + LoRA(r=16, alpha=32)
Prompt: TAGA-aware压缩编码 (~195 tokens)
训练: SFT on TAGA-MAPPO专家轨迹, 45K条, 2 epochs
推理: 4-bit量化, 批量推理, ~70ms/step
指令: 5种任务指令(BAL/SCH/ATK/EVD/COO)
```

### 资源预算

| 阶段 | 时间 | 硬件 | 显存 |
|------|------|------|------|
| TAGA-MAPPO训练（阶段A） | 1.5h | A100 | <100MB |
| 专家数据生成 | 30min | CPU+GPU | ~3GB |
| LoRA微调 | 37min | A100 | ~16GB |
| 评估推理 | 15min | A100 | ~3GB |
| **总计** | **~2.9h** | A100 80GB | max 16GB |

### 实验设计（9个实验）

| # | 实验 | 对比 | 主指标 |
|---|------|------|--------|
| E1 | 主实验 | TAGA-LLM vs ACO/Random/Greedy | Coverage, Success, Time |
| E2 | vs MARL | TAGA-LLM vs TAGA-MAPPO/MAPPO | Coverage, Success, Time |
| E3 | 指令调控 | 5种指令对比 | 轨迹热力图 |
| E4 | 消融-TAGA | 有/无TAGA-aware prompt | Coverage, Success |
| E5 | 消融-CoT | 有/无CoT推理 | 动作准确率 |
| E6 | 泛化 | 训练/未见场景 | ΔMetrics |
| E7 | 延迟 | 批量推理延迟 | ms/step |
| E8 | 可扩展性 | 5/10/15/20 UAV | Coverage, Time |
| E9 | 鲁棒性 | 通信中断10%/20%/30% | 性能下降 |

### 论文结构（ICGNC格式，10页）

| 章节 | 页数 | 内容 |
|------|------|------|
| Abstract | 0.3 | 问题+方法+结果 |
| 1. Introduction | 1.5 | 背景+动机+贡献 |
| 2. Related Work | 1.0 | ACO/MARL/LLM for decision |
| 3. Problem Formulation | 0.7 | 环境+指标+约束 |
| 4. Method | 2.5 | TAGA-aware prompt + LoRA + CoT + 指令 |
| 5. Experiments | 2.5 | E1-E9 |
| 6. Conclusion | 0.5 | 总结+Future Work |
| References | 1.0 | ~20篇 |

### 评分历史

| 维度 | B-R0 | B-R1 | B-R2 | B-R3 | B-R4 | B-R5 |
|------|------|------|------|------|------|------|
| Problem Fidelity | 9 | 9 | 9 | 9 | 9 | 9 |
| Method Specificity | 9 | 9 | 9 | 9 | 9 | 9 |
| Contribution Quality | 9 | 9 | 9 | 9 | 9 | 9 |
| Frontier Leverage | 9 | 9 | 9 | 9 | 9 | 9 |
| Feasibility | 9 | 9 | 9 | 9 | 9 | 9 |
| Validation Focus | 9 | 8 | 8 | 9 | 9 | 9 |
| Venue Readiness | 8 | 8 | 8 | 8 | 9 | 9 |
| **综合** | **8.9** | **8.8** | **8.85** | **8.95** | **9.05** | **9.05** |

**最终综合分: 9.05/10 — 达到停止条件**

### 确认清单

| 项目 | 状态 |
|------|------|
| TAGA-aware prompt设计 | ✅ |
| 压缩token编码 | ✅ |
| 训练数据生成pipeline | ✅ |
| 反事实数据生成 | ✅ |
| LoRA训练配置 | ✅ |
| Loss mask策略 | ✅ |
| 批量推理引擎 | ✅ |
| 指令调控机制 | ✅ |
| 9个实验设计 | ✅ |
| 资源预算 | ✅ |
| 论文结构 | ✅ |

**结论**: 阶段B方案已达到可实现状态（9.05/10），进入Pipeline代码生成。
