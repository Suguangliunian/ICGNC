# Round 1 Refinement — 方法具体化

> 日期: 2026-04-08
> 改进重点: 方法具体化，确保可实现
> 上轮综合分: 5.7/10

---

## 改进摘要

本轮针对Round 0的主要缺陷进行了以下改进：
1. 设计了局部观测窗口机制，将prompt长度控制在512 tokens以内
2. 加入了Chain-of-Thought (CoT)推理模块，让LLM输出决策理由
3. 明确了数据生成的完整pipeline和仿真器设计
4. 设计了多UAV顺序决策+信息传递机制
5. 丰富了实验设计

---

## 改进后的方案

### 改进1: 局部观测窗口 + 压缩编码

**问题**：全局地图信息编码为文本后token数爆炸（100×100地图 → 数万token）

**解决方案**：引入局部观测窗口 + 统计摘要

```
[SYSTEM] You are UAV-{id} in a {L}x{W} grid. Step {t}/{T_max}.

[SELF] pos=({x},{y}) mode={search/attack} ammo={a}

[LOCAL_VIEW] (7x7 window centered at current position)
Row+3: [U U U S S T U]
Row+2: [U S S S S S U]
Row+1: [S S S S S S S]
Row+0: [S S * S S S S]    (* = current position)
Row-1: [S S S S U U U]
Row-2: [U U U U U U U]
Row-3: [U U U U U U T]
Legend: U=unsearched, S=searched, T=threat, *=self

[GLOBAL_SUMMARY]
Total_coverage: 45.2%
Unsearched_density_N: 0.8, NE: 0.6, E: 0.3, SE: 0.2, S: 0.1, SW: 0.4, W: 0.7, NW: 0.9
Nearest_unsearched_dir: NW, dist: 3

[TEAMMATES] (sorted by distance)
UAV-3: pos=(12,8) dir=N mode=search dist=5
UAV-1: pos=(20,15) dir=E mode=attack dist=12

[TARGETS]
T1: pos=(30,25) need=2 assigned=1 dist=18 status=alive
T2: pos=(45,10) need=1 assigned=0 dist=8 status=alive

[MY_HISTORY] move_N, move_N, move_NE, move_E, search
```

**Token数估算**：
- SYSTEM: ~30 tokens
- SELF: ~20 tokens
- LOCAL_VIEW (7×7): ~120 tokens
- GLOBAL_SUMMARY: ~60 tokens
- TEAMMATES (最多5个): ~80 tokens
- TARGETS (最多8个): ~100 tokens
- HISTORY (5步): ~30 tokens
- **总计: ~440 tokens** ✓ 在512以内

### 改进2: Chain-of-Thought 决策推理

**输出格式升级**：
```
[REASONING]
1. Current coverage is 45.2%, need to prioritize search.
2. NW direction has highest unsearched density (0.9).
3. T2 is nearby (dist=8) and unassigned, but I have search mode priority.
4. No teammate is heading NW, so I should go there to avoid overlap.
[DECISION] action=move_NW mode=search target=none
```

**训练时**：CoT部分也参与loss计算，但权重降低（0.3×），决策部分权重为1.0×

**推理时**：CoT提供可解释性，同时引导模型做出更合理的决策

### 改进3: 数据生成Pipeline

```
数据生成流程:
┌─────────────────────────────────────────┐
│ Step 1: 场景生成器                        │
│   - 随机生成地图配置(大小/目标/威胁)        │
│   - 50种不同配置 × 20次随机种子 = 1000场景  │
├─────────────────────────────────────────┤
│ Step 2: ACO仿真器运行                     │
│   - 对每个场景运行ACO算法                   │
│   - 记录每步每个UAV的(state, action)       │
│   - 每场景约100步 × 6 UAV = 600条数据      │
│   - 总计: 1000 × 600 = 600k条原始数据      │
├─────────────────────────────────────────┤
│ Step 3: 数据筛选与增强                     │
│   - 筛选ACO表现前50%的轨迹(质量过滤)        │
│   - 旋转增强(4倍): 0°, 90°, 180°, 270°    │
│   - 最终: ~120k条高质量数据                 │
├─────────────────────────────────────────┤
│ Step 4: CoT标注                           │
│   - 用规则模板为每条数据生成reasoning        │
│   - 基于当前状态的关键特征自动生成           │
│   - 模板覆盖: 搜索优先/攻击优先/协同避让     │
└─────────────────────────────────────────┘
```

**CoT自动标注规则示例**：
```python
def generate_cot(state, action):
    reasons = []
    # 覆盖率分析
    if state.coverage < 0.5:
        reasons.append(f"Coverage is low ({state.coverage:.1%}), prioritize search.")
    # 方向分析
    best_dir = max(state.unsearched_density, key=state.unsearched_density.get)
    reasons.append(f"{best_dir} has highest unsearched density ({state.unsearched_density[best_dir]:.1f}).")
    # 目标分析
    if state.nearby_targets:
        t = state.nearest_target
        reasons.append(f"Target {t.id} at dist={t.dist}, need={t.ammo_needed}, assigned={t.assigned}.")
    # 协同分析
    if state.nearby_teammates:
        for tm in state.nearby_teammates[:2]:
            reasons.append(f"UAV-{tm.id} heading {tm.direction}, avoid overlap.")
    return reasons
```

### 改进4: 多UAV顺序决策机制

**方案**：Round-Robin顺序决策 + 信息传递

```
每个时间步:
  1. 按UAV编号顺序决策: UAV-1 → UAV-2 → ... → UAV-N
  2. UAV-i决策时，可以看到UAV-1~UAV-(i-1)在本步的已决策动作
  3. 将已决策UAV的动作加入prompt:
     [DECIDED_THIS_STEP]
     UAV-1: move_N (search)
     UAV-2: move_E (search)
     (UAV-3 is deciding...)
```

**优势**：
- 后决策的UAV可以避免与先决策UAV的动作冲突
- 模拟了分布式系统中的信息传播

**训练数据中的处理**：
- ACO是并行决策的，需要人工构造顺序决策的数据
- 方法：随机打乱UAV决策顺序，将其他UAV的动作作为已知信息加入prompt

### 改进5: 丰富实验设计

| 实验 | 目的 | 配置 |
|------|------|------|
| E1: LLM vs ACO | 核心对比 | 标准场景(80×80, 6UAV, 5目标) |
| E2: 泛化测试 | 场景泛化 | 训练外场景(120×120, 10UAV, 8目标) |
| E3: CoT消融 | CoT价值 | 有CoT vs 无CoT |
| E4: 观测窗口消融 | 窗口大小影响 | 5×5 vs 7×7 vs 9×9 |
| E5: 数据量消融 | 数据效率 | 10k vs 50k vs 120k |
| E6: 推理延迟 | 实时性 | 测量每步决策时间 |
| E7: 顺序决策消融 | 协同机制 | 顺序 vs 并行决策 |

---

## 更新后的Compute Budget

| 阶段 | 时间 | 说明 |
|------|------|------|
| 场景生成 | 5min | 1000个场景配置 |
| ACO仿真 | 20min | 并行运行，16核CPU |
| 数据处理+CoT标注 | 10min | 筛选+增强+模板标注 |
| LoRA微调 | 2h | A100, ~120k数据, 3 epochs |
| 评估(7个实验) | 40min | A100, 4-bit推理 |
| **总计** | **~3.25h** | 在4h预算内 ✓ |
