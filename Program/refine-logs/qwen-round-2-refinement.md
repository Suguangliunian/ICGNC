# Round 2 Refinement — 训练效率优化 + 核心创新强化

> 日期: 2026-04-08
> 改进重点: 修正训练时间、加入指令调控机制、明确核心卖点
> 上轮综合分: 6.7/10

---

## 改进摘要

1. **修正训练时间估算**：压缩数据量和序列长度，确保2小时内完成训练
2. **加入指令调控（Instruction-Steered）机制**：LLM独特优势的核心展示
3. **加入DRL baseline**：PPO作为学习方法的对比基准
4. **明确论文故事线**：可解释、可调控、可泛化的LLM决策引擎

---

## 核心改进

### 改进1: 训练时间修正

**问题诊断**：
- 原估算120k数据 × 3 epochs × 500 tokens ≈ 180M tokens
- A100上Qwen3.5-4B + LoRA的吞吐量 ≈ 2000-3000 tokens/s
- 实际需要 ~25小时，严重超时

**修正方案**：

| 参数 | 原方案 | 修正方案 | 说明 |
|------|--------|----------|------|
| 数据量 | 120k | 30k | 质量>数量，只保留top-25%轨迹 |
| 序列长度 | ~500 tokens | ~300 tokens | 压缩prompt格式 |
| Epochs | 3 | 2 | 减少过拟合风险 |
| Batch size | 8 | 16 | 增大batch提高GPU利用率 |
| Grad accum | 4 | 2 | 配合batch size调整 |

**修正后训练量**：
- 30k × 2 epochs × 300 tokens = 18M tokens
- 18M / 3000 tokens/s ≈ 6000s ≈ **1.7小时** ✓

**Prompt压缩策略**：

```
<|system|>UAV-{id} in {L}x{W} grid. Step {t}.
<|state|>pos=({x},{y}) mode={m} ammo={a}
<|view|>
+3:UUUSSTU
+2:USSSSU
+1:SSSSSSS
 0:SS*SSSS
-1:SSSUUUU
-2:UUUUUUU
-3:UUUUUUT
<|summary|>cov=45% best_dir=NW:0.9 nearest_unsrch=3
<|team|>U3:(12,8)N,srch,d5|U1:(20,15)E,atk,d12
<|targets|>T1:(30,25)n2a1d18|T2:(45,10)n1a0d8
<|decided|>U1:NW,srch|U2:E,srch
<|hist|>N,N,NE,E,srch
<|task|>{instruction}
```

**压缩后token数估算**：~250-300 tokens ✓

### 改进2: 指令调控机制 (Instruction-Steered Decision Making)

**核心创新**：通过修改 `<|task|>` 字段的自然语言指令，动态调整UAV策略偏好，无需重新训练。

**预定义指令集**：

| 指令ID | 指令内容 | 预期行为 |
|--------|----------|----------|
| I1 | "Maximize search coverage. Ignore targets unless directly adjacent." | 纯搜索优先 |
| I2 | "Attack all known targets immediately. Coordinate with teammates." | 攻击优先 |
| I3 | "Balance search and attack. Prioritize unassigned targets." | 平衡策略（默认） |
| I4 | "Avoid threat zones. Prioritize safety over coverage." | 安全优先 |
| I5 | "Focus on area ({x1},{y1})-({x2},{y2}). Other areas are low priority." | 区域聚焦 |

**训练数据中的指令多样化**：
- 每条ACO轨迹数据，根据当时的状态和动作，自动匹配最合适的指令
- 例如：如果ACO在某步选择了搜索动作且附近有目标，标注为I1（搜索优先）
- 额外生成一些"反事实"数据：同一状态下，不同指令对应不同动作

**指令匹配规则**：
```python
def assign_instruction(state, action):
    if action.mode == "search" and state.nearby_alive_targets:
        return "I1"  # 搜索优先（即使有目标也选择搜索）
    elif action.mode == "attack":
        return "I2"  # 攻击优先
    elif state.in_threat_zone and action.avoids_threat:
        return "I4"  # 安全优先
    else:
        return "I3"  # 平衡策略
```

### 改进3: 输出格式优化

**带CoT的紧凑输出**：
```
<|think|>cov_low→srch|NW_best(0.9)|T2_near_unassigned|U3_heading_N_no_conflict
<|act|>NW,srch,none
```

- `<|think|>` 用压缩的关键词链表示推理过程（节省token）
- `<|act|>` 用最简格式表示动作

**训练时loss mask**：
- `<|think|>` 部分：loss权重 0.3
- `<|act|>` 部分：loss权重 1.0
- 输入prompt部分：loss权重 0.0

### 改进4: 完整Baseline对比

| 方法 | 类型 | 说明 |
|------|------|------|
| ACO (原论文) | 传统优化 | 原始蚁群优化方法 |
| Random | 随机 | 随机动作基线 |
| Greedy | 规则 | 贪心搜索+最近目标攻击 |
| PPO | DRL | 标准强化学习方法 |
| BC-MLP | 行为克隆 | MLP做行为克隆（同数据） |
| **Ours (LLM-CoT)** | LLM | Qwen3.5-4B + LoRA + CoT |
| **Ours (LLM-Instruct)** | LLM | + 指令调控 |

**PPO baseline配置**：
- 网络：MLP (256, 256) 或 GRU (128)
- 状态：与LLM相同的特征（数值化）
- 奖励：搜索覆盖率增量 + 目标消灭奖励 - 时间惩罚
- 训练：100k环境步，~1小时

### 改进5: 更新后的实验设计

| 实验 | 目的 | 配置 | 运行次数 |
|------|------|------|----------|
| E1: 主实验 | 全方法对比 | 80×80, 6UAV, 5目标 | 20次 |
| E2: 规模泛化 | 地图/UAV数泛化 | 120×120, 10UAV, 8目标 | 10次 |
| E3: 指令调控 | LLM独特优势 | 同一场景，5种指令 | 10次×5指令 |
| E4: CoT消融 | CoT价值 | 有CoT vs 无CoT | 20次 |
| E5: 数据量消融 | 数据效率 | 5k/10k/20k/30k | 各10次 |
| E6: 推理延迟 | 实时性 | 测量决策时间 | 100次 |
| E7: 可解释性案例 | 定性分析 | 展示CoT推理过程 | 5个案例 |

---

## 更新后的论文故事线

**Title**: Instruction-Steered LLM as Interpretable Decision Engine for Cooperative UAV Search-Attack Mission Planning

**核心卖点**（三个独特优势）：
1. **可解释性**：CoT推理让每个决策都有可追溯的理由
2. **可调控性**：通过自然语言指令动态调整策略，无需重训练
3. **可泛化性**：预训练知识提供跨场景迁移能力

**故事线**：
传统方法（ACO）→ 学习方法（DRL）→ 基础模型方法（LLM）的演进
- ACO：性能好但不可解释、不可调控
- DRL：可学习但需要大量交互、不可解释
- LLM：可解释、可调控、可泛化，且通过LoRA微调在有限资源下可行

---

## 更新后的Compute Budget

| 阶段 | 时间 | 说明 |
|------|------|------|
| 数据生成+处理 | 25min | 场景生成+ACO仿真+筛选+CoT标注 |
| LoRA微调 | 1.7h | 30k数据, 2 epochs, ~300 tokens/sample |
| PPO baseline训练 | 1h | 100k环境步 |
| 全部评估 | 45min | 7个实验 |
| **总计** | **~3.5h** | 在4h预算内 ✓ |

注：PPO训练可与LoRA微调并行（如果有第二张GPU），否则串行总计约3.5h。
