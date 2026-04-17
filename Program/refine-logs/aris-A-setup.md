# 阶段A准备：核心创新点确定 + ARIS诊断框架

> 日期: 2026-04-08
> 基础: 原论文ACO方案 → MADRL-GAT方案

---

## 1. 原论文核心方法回顾

原论文（Liu et al.）提出基于蚁群优化（ACO）的UAV集群分布式在线自适应搜索-攻击任务规划：
- **信息素更新机制**：环境信息编码为信息素浓度
- **自适应状态转移规则**：搜索模式和攻击模式使用不同启发式函数
- **自适应目标分配决策**：根据搜索覆盖率阈值切换分配策略（最少消耗UAV vs 距离优先）

**局限性**：
- 启发式规则手工设计，泛化能力差
- 信息素间接通信，协同效率低
- 无法端到端优化，搜索和攻击分离处理
- 大规模场景收敛慢

## 2. 新核心创新点：MADRL-GAT

**替代方案**：Multi-Agent Deep Reinforcement Learning with Graph Attention Network

**核心创新点（与原论文ACO的差异化）**：

| 维度 | 原论文ACO | 本方案MADRL-GAT |
|------|----------|----------------|
| 决策机制 | 信息素+启发式规则 | 端到端神经网络策略 |
| 通信方式 | 信息素间接通信 | GAT显式注意力通信 |
| 任务耦合 | 搜索攻击分离 | 分层策略联合优化 |
| 适应性 | 需重新规划 | 实时响应环境变化 |
| 可扩展性 | 随UAV数指数增长 | 参数共享线性扩展 |

**主要贡献**：
1. **GAT动态通信拓扑建模**：用Graph Attention Network替代信息素机制，实现UAV间显式、自适应的信息聚合
2. **分层MAPPO策略**：高层模式选择（搜索/攻击）+ 低层动作执行，端到端联合优化
3. **课程学习训练策略**：3阶段递增复杂度，解决大规模MADRL训练难题

## 3. ARIS诊断框架

### 3.1 评估维度（7维度，参考research-refine标准）

| 维度 | 权重 | 说明 |
|------|------|------|
| Problem Fidelity | 15% | 方法是否仍然解决原始问题 |
| Method Specificity | 25% | 接口、表示、损失、训练阶段是否具体到可实现 |
| Contribution Quality | 25% | 是否有一个主导的机制级贡献 |
| Frontier Leverage | 15% | 是否恰当使用现代技术 |
| Feasibility | 10% | 能否在约束资源下训练和集成 |
| Validation Focus | 5% | 实验是否最小但充分 |
| Venue Readiness | 5% | 贡献是否足够尖锐和及时 |

### 3.2 ARIS循环流程

```
每轮迭代：
  1. Diagnose: 运行7维度评分，识别最薄弱维度
  2. Innovate: 针对薄弱点提出1-2个创新改进
  3. Implement: 修改方案文档（完整重写refined proposal）
  4. Test: 重新7维度评分
  5. Evaluate: 对比前后分数，记录改进/退步，决定下轮方向
```

### 3.3 输出文件命名规范

```
aris-A-round-{N}-diagnosis.md    # 诊断报告
aris-A-round-{N}-innovation.md   # 创新提案
aris-A-round-{N}-refinement.md   # 完整refined方案
```

### 3.4 停止条件

- 综合分数 >= 9.0/10
- 或完成5轮迭代

---

## 4. 基线方案

基线方案即 `round-0-initial-proposal.md` 中的MADRL-GAT方案，作为ARIS迭代的起点。
