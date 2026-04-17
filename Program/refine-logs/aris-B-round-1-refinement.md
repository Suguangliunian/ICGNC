# ARIS-B Round 1 — 诊断 + 创新 + 精炼

> 日期: 2026-04-08
> 基础: 已有Qwen方案(8.9/10) + 阶段A TAGA创新点
> 本轮重点: 融合TAGA思想到LLM prompt，建立基线

---

## Diagnose: 已有Qwen方案的薄弱点

已有方案（qwen-round-5-refinement.md）评分8.9/10，主要薄弱点：

| 问题 | 严重程度 |
|------|----------|
| 1. 邻居信息编码未考虑通信拓扑（全部邻居平等对待） | CRITICAL |
| 2. 缺乏与阶段A TAGA创新点的融合 | CRITICAL |
| 3. 推理延迟可能影响实时性（未验证） | IMPORTANT |
| 4. 反事实数据生成规则可能引入偏差 | MINOR |

**核心问题**: 已有Qwen方案的prompt中，邻居信息是简单列举的，没有体现TAGA的核心思想（距离衰减、任务感知、信息新鲜度）。

---

## Innovation: TAGA-aware Prompt Design

### 改进1: 拓扑感知邻居信息编码

**原方案prompt中的邻居信息**:
```
[NEIGHBORS]
UAV-3: pos=(45,67) mode=search dist=8
UAV-7: pos=(52,71) mode=attack dist=12
UAV-1: pos=(60,80) mode=search dist=18
```

**TAGA-aware改进**:
```
[NEIGHBORS] (sorted by communication quality, closest first)
UAV-3: pos=(45,67) mode=search dist=8km signal=strong freshness=2steps_ago
  → Same mode as you. Recent info. High reliability.
UAV-7: pos=(52,71) mode=attack dist=12km signal=good freshness=1step_ago
  → Different mode (attacking Target-2). Very recent info.
UAV-1: pos=(60,80) mode=search dist=18km signal=weak freshness=5steps_ago
  → Same mode. Stale info. Near comm range limit.
[COMM_SUMMARY] 3 neighbors in range. Avg signal: good. 2 searching, 1 attacking.
```

**关键改进**:
1. 按距离排序（近邻优先，对应TAGA距离衰减）
2. 标注信号质量（strong/good/weak，对应通信信道质量）
3. 标注信息新鲜度（对应TAGA新鲜度因子）
4. 标注模式匹配关系（对应TAGA任务感知因子）
5. 添加通信摘要（帮助LLM快速理解全局态势）

### 改进2: CoT推理中融入TAGA逻辑

**原方案CoT**:
```
[思考] 当前搜索覆盖率低，应继续搜索。东北方向有未探索区域。
[动作] move_NE [模式] search [目标] none
```

**TAGA-aware CoT**:
```
[思考] 
1. 通信态势：3个邻居在通信范围内，UAV-3(近/同模式/新鲜)信息最可靠。
2. 协同分析：UAV-3在西南方搜索，我应向东北避免重复。UAV-7正在攻击T2，无需协助。
3. 决策：搜索模式，向东北移动，与UAV-3形成互补搜索。
[动作] move_NE [模式] search [目标] none
```

---

## 精炼方案: TAGA-LLM

### 完整Prompt模板（Qwen3.5-4B chat template）

```
<|im_start|>system
You are UAV-{id} in a cooperative search-attack mission. Your goal is to maximize team search coverage and minimize target existence time. Consider communication topology when making decisions.
<|im_end|>
<|im_start|>user
[MISSION] {instruction}

[STATE] Position: ({x},{y}), Mode: {mode}, Ammo: {ammo}, Step: {t}/{T_max}

[LOCAL_MAP]
Searched: {searched_cells_summary}
Unsearched_Nearby: {unsearched_cells}
Threats: {threat_info}

[NEIGHBORS] (sorted by comm quality)
{neighbor_1_with_taga_info}
{neighbor_2_with_taga_info}
{neighbor_3_with_taga_info}
[COMM_SUMMARY] {n_neighbors} in range. {mode_distribution}. Avg signal: {avg_signal}.

[TARGETS]
{target_info_with_assignment}

[HISTORY] Last 5 actions: {action_history}

Decide your next action. Think step by step about communication topology and coordination.
<|im_end|>
<|im_start|>assistant
[思考] {chain_of_thought}
[动作] {direction} [模式] {mode} [目标] {target}
<|im_end|>
```

### 训练数据生成（融合TAGA思想）

1. 运行TAGA-MAPPO（阶段A方法）生成专家轨迹
2. 对每个时间步，构建TAGA-aware prompt
3. 用TAGA-MAPPO的决策作为标签
4. 在CoT中注入TAGA的注意力权重解释

**数据量**: ~45K条（含反事实）
**Token/条**: ~350 tokens

### LoRA配置（与已有方案一致）

```python
base_model = "Qwen/Qwen3.5-4B"
lora_config = LoraConfig(r=16, lora_alpha=32, 
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_dropout=0.05, task_type="CAUSAL_LM")
training_args = TrainingArguments(
    per_device_train_batch_size=8, gradient_accumulation_steps=4,
    num_train_epochs=2, learning_rate=2e-4, bf16=True,
    gradient_checkpointing=True, max_seq_length=384)
```

### 评分

| 维度 | Qwen原方案 | B-R1 | 变化 |
|------|-----------|------|------|
| Problem Fidelity | 9 | 9 | → |
| Method Specificity | 9 | 9 | → |
| Contribution Quality | 9 | 9 | → (TAGA融合增强了创新性) |
| Frontier Leverage | 9 | 9 | → |
| Feasibility | 9 | 9 | → |
| Validation Focus | 9 | 8 | ↓1 (需要重新设计对比实验) |
| Venue Readiness | 8 | 8 | → |
| **综合** | **8.9** | **8.8** | **↓0.1** (融合初期略有波动) |
