# Research Proposal: LLM-based Online Adaptive Mission Planning for UAV Swarm

> Round 0 — 初始方案
> 日期: 2026-04-08

---

## 1. Problem Anchor

- **Bottom-line problem**: UAV集群在未知环境中的分布式在线自适应协同搜索-攻击任务规划
- **Must-solve bottleneck**: 传统优化算法（如ACO）依赖手工设计的启发式规则，泛化能力差；需要一种能从数据中学习决策策略的方法
- **Non-goals**: 不涉及真实飞行、不涉及3D环境、不涉及通信延迟建模
- **Constraints**: 单张A100 GPU（80GB），Qwen3.5-4B模型，LoRA微调，训练时间<4小时
- **Success condition**: LLM-based方法在搜索覆盖率和目标存在时间上接近或优于ACO基线（差距<5%）

---

## 2. Technical Gap

### 传统ACO方法的局限
1. 启发式规则需要人工设计，难以适应复杂动态环境
2. 信息素更新机制在大规模场景中收敛慢
3. 缺乏对历史经验的学习和泛化能力
4. 状态转移概率公式固定，无法根据场景自适应调整权重

### LLM的潜在优势
1. 强大的序列建模能力，天然适合时序决策问题
2. 预训练知识提供通用推理先验
3. LoRA微调使得在有限资源下可行
4. 上下文窗口可以编码丰富的环境信息

### 关键技术挑战
1. 如何将连续的空间状态有效编码为离散token
2. 如何保证多UAV之间的协同一致性
3. 如何在推理延迟约束下实现实时决策
4. 如何用有限的专家数据训练出泛化策略

---

## 3. Method Thesis

**One-sentence thesis**: 将UAV集群任务规划建模为条件序列生成问题，用LoRA微调的Qwen3.5-4B作为分布式决策引擎，每个UAV基于局部观测的文本化表示生成动作序列。

**核心创新**: State-Action Tokenization + LoRA-tuned LLM Decision Making

---

## 4. Proposed Method

### 4.1 环境状态文本化编码 (State Tokenization)

将环境状态转化为结构化文本prompt，每个UAV独立构建自己的观测prompt：

```
[SYSTEM] You are UAV-{id} in a {L}x{W} grid mission area. Your mission is to cooperatively search the area and attack discovered targets.

[STATE] Position: ({x},{y}), Mode: {search/attack}, Remaining_Ammo: {a}, Time_Step: {t}

[LOCAL_MAP] 
Searched_Cells: [(x1,y1), (x2,y2), ...]
Unsearched_Nearby: [(x3,y3), (x4,y4), ...]
Threat_Zones: [(x5,y5,threat_level), ...]

[NEIGHBORS]
UAV-{n1}: pos=({x1},{y1}) mode={search} dist={d1}
UAV-{n2}: pos=({x2},{y2}) mode={attack} dist={d2}

[TARGETS]
Target-{t1}: pos=({tx},{ty}) ammo_needed={A1} status={alive} assigned_uavs={k1}
Target-{t2}: pos=({tx},{ty}) ammo_needed={A2} status={destroyed}

[HISTORY] Last 5 actions: [move_N, move_E, search, move_S, attack_T1]

[TASK] Based on the current state, decide your next action to maximize team search coverage and minimize target existence time. Output your decision in the format: [ACTION] {action} [MODE] {mode} [TARGET] {target}
```

### 4.2 动作空间设计 (Action Tokenization)

输出格式严格定义：
```
[ACTION] move_{direction} [MODE] {search/attack} [TARGET] {target_id/none}
```

- `direction` ∈ {N, NE, E, SE, S, SW, W, NW, STAY}（9个移动方向）
- `mode` ∈ {search, attack}（2种工作模式）
- `target` ∈ {T1, T2, ..., Tn, none}（目标分配）

有效动作组合约束：
- mode=search 时，target 必须为 none
- mode=attack 时，target 必须为有效目标ID
- 移动不能超出地图边界

### 4.3 训练数据生成

**数据来源**：
1. **ACO专家轨迹**（主要）：运行原始ACO算法在多种场景配置下生成最优/近优轨迹
2. **规则增强随机策略**（辅助）：基于简单启发式规则的策略，提供多样性

**场景配置多样化**：
- 地图大小：50×50, 80×80, 100×100
- UAV数量：4, 6, 8
- 目标数量：3, 5, 8
- 目标位置：随机生成
- 威胁区域：0, 2, 4个

**数据格式**：每条数据为 (state_prompt, action_label) pair
- 预计生成 ~100k 条 state-action pairs
- 数据生成时间：~30分钟（CPU并行）

### 4.4 LoRA微调配置

```python
# 模型配置
base_model = "Qwen/Qwen3.5-4B"
quantization = None  # bf16 全精度训练
dtype = torch.bfloat16

# LoRA配置
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 训练配置
training_args = TrainingArguments(
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    logging_steps=50,
    save_strategy="epoch",
    max_grad_norm=1.0,
)
```

**显存估算**：
- 模型参数（bf16）：~8GB
- LoRA参数：~50MB
- 优化器状态：~2GB
- 激活值（batch=8, seq_len=1024）：~12GB
- 梯度：~2GB
- **总计：~24GB**（A100 80GB 完全够用）

### 4.5 推理流程

每个时间步的决策流程：
```
for each time_step t:
    for each UAV i:
        1. 收集UAV-i的局部观测
        2. 构建state_prompt（按4.1模板）
        3. 输入Qwen3.5-4B（4-bit量化推理）
        4. 生成action tokens
        5. 解析为(direction, mode, target)
        6. 验证动作合法性（边界检查、目标有效性）
        7. 若非法则回退到默认动作
    执行所有UAV动作
    更新环境状态
```

**推理优化**：
- 使用4-bit量化减少推理显存（~3GB）
- KV-cache复用减少重复计算
- 批量推理：同时处理多个UAV的prompt

### 4.6 损失函数设计

**主损失**：标准因果语言模型损失（Cross-entropy on action tokens）
```
L_main = -Σ log P(a_t | s_t, a_{<t})
```
其中只对 [ACTION], [MODE], [TARGET] 部分的token计算损失，prompt部分mask掉。

**辅助正则化**（可选）：
- 动作多样性正则：防止模型坍缩到单一动作
- 空间覆盖奖励：鼓励探索未搜索区域

---

## 5. Claim-Driven Validation

### Claim 1: LLM能学会UAV搜索-攻击任务规划
- **实验设计**：在标准场景（80×80地图，6架UAV，5个目标）下对比LLM vs ACO
- **Metrics**: 
  - 搜索覆盖率 (Search Coverage Rate, SCR)
  - 目标平均存在时间 (Average Target Existence Time, ATET)
  - 综合指标 J = w1·SCR - w2·ATET
- **预期结果**：LLM方法达到ACO性能的95%以上

### Claim 2: LLM具有场景泛化能力
- **实验设计**：在训练时未见过的场景配置下测试（不同地图大小、UAV数量、目标分布）
- **Metrics**: 性能下降幅度 ΔJ = J_seen - J_unseen
- **预期结果**：性能下降<10%

### Claim 3: LLM推理速度满足在线决策需求
- **实验设计**：测量单步决策延迟
- **Metrics**: 每个UAV每步决策时间
- **预期结果**：<100ms/UAV/step（4-bit量化推理）

---

## 6. Compute Budget

| 阶段 | 时间 | 硬件 | 显存 |
|------|------|------|------|
| 数据生成 | 30min | CPU (16核) | - |
| LoRA微调 | 2h | A100 80GB | ~24GB |
| 评估推理 | 30min | A100 80GB | ~5GB |
| **总计** | **~3h** | | |

---

## 7. 风险与缓解

| 风险 | 概率 | 缓解措施 |
|------|------|----------|
| LLM生成非法动作 | 高 | 动作验证+回退机制 |
| 训练数据不足 | 中 | 数据增强（旋转、翻转对称性） |
| 推理延迟过高 | 中 | 4-bit量化 + 批量推理 |
| 多UAV协同差 | 中 | 在prompt中加入邻居信息 |
| 过拟合训练场景 | 中 | 场景多样化 + dropout |
