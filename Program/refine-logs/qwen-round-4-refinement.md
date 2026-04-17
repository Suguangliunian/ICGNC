# Round 4 Refinement — 训练细节完善 + 理论动机 + 实验补全

> 日期: 2026-04-08
> 改进重点: LoRA超参数、反事实动作规则、LoRA rank消融、理论动机
> 上轮综合分: 8.3/10

---

## 改进摘要

1. **明确LoRA超参数和完整训练配置**
2. **补充反事实动作的具体计算规则**
3. **加入LoRA rank消融实验**
4. **补充通信模型和目标生成方式**
5. **构建Related Work结构**
6. **加入理论动机段落**

---

## 核心改进

### 改进1: 完整训练配置

**LoRA配置**：

```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                          # LoRA rank
    lora_alpha=32,                 # scaling factor = alpha/r = 2
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    # 不对MLP层做LoRA，减少参数量
)
# 可训练参数: ~6.8M (占总参数4B的0.17%)
```

**训练超参数**：

```python
training_args = {
    "num_train_epochs": 2,
    "per_device_train_batch_size": 16,
    "gradient_accumulation_steps": 2,     # 有效batch size = 32
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,                 # ~56步warmup (45k/32*2*0.05)
    "weight_decay": 0.01,
    "bf16": True,                          # A100原生支持bf16
    "max_seq_length": 384,                 # 留一些buffer
    "gradient_checkpointing": True,        # 节省显存
    "logging_steps": 50,
    "save_strategy": "epoch",
    "eval_strategy": "steps",
    "eval_steps": 200,
    "seed": 42,
}
```

**显存估算**：
- 模型权重(bf16): 4B × 2 bytes = 8GB
- LoRA参数: ~6.8M × 2 bytes ≈ 14MB
- 优化器状态(AdamW): LoRA参数 × 8 bytes ≈ 54MB
- 激活值(gradient checkpointing): ~8GB (batch=16, seq=384)
- 总计: ~16GB，A100 80GB完全够用

**[思考]部分的训练策略**：
- loss权重设为0.2（降低，因为推理路径不唯一）
- 只要求[动作]部分严格匹配
- [思考]部分允许多样性，但通过loss引导模型学会"提及关键因素"

### 改进2: 反事实动作的具体计算规则

```python
def compute_best_search_action(state):
    """搜索优先：朝未搜索密度最高的方向移动"""
    best_dir, best_score = None, -1
    for direction in EIGHT_DIRECTIONS:
        # 计算该方向3格范围内的未搜索格子数
        nx, ny = state.pos + direction_to_delta(direction)
        score = count_unsearched_in_cone(state.grid, nx, ny, direction, depth=3)
        if score > best_score:
            best_dir, best_score = direction, score
    return Action(direction=best_dir, mode="search", target=None)

def compute_best_attack_action(state):
    """攻击优先：朝最近可攻击目标移动"""
    if not state.known_targets:
        return compute_best_search_action(state)  # 无目标时退化为搜索
    
    # 找最近的未被充分分配的目标
    best_target = min(
        state.known_targets,
        key=lambda t: manhattan_distance(state.pos, t.pos) 
                      + 10 * max(0, t.assigned_uavs - t.required_uavs)
    )
    
    # A*路径的第一步方向
    path = astar(state.pos, best_target.pos, state.grid)
    if len(path) > 1:
        direction = pos_to_direction(path[0], path[1])
    else:
        direction = pos_to_direction(state.pos, best_target.pos)
    
    # 如果在攻击范围内（相邻格），mode=attack
    dist = manhattan_distance(state.pos, best_target.pos)
    mode = "attack" if dist <= 1 and state.ammo > 0 else "search"
    
    return Action(direction=direction, mode=mode, target=best_target.id if mode == "attack" else None)

def compute_safest_action(state):
    """安全优先：远离威胁区域"""
    threat_center = compute_threat_centroid(state)
    if threat_center is None:
        return compute_best_search_action(state)
    
    # 选择远离威胁中心的方向
    best_dir = max(
        EIGHT_DIRECTIONS,
        key=lambda d: euclidean_distance(
            state.pos + direction_to_delta(d), threat_center
        )
    )
    return Action(direction=best_dir, mode="search", target=None)
```

**数据质量控制**：
- 只保留反事实动作与原始动作不同的样本（避免重复）
- 对每条反事实数据，验证动作的合法性（不越界、不碰撞）
- 反事实数据占比控制在30-40%，避免喧宾夺主

### 改进3: LoRA Rank消融实验

新增实验 **E9: LoRA Rank消融**：

| Rank | 可训练参数 | 预期训练时间 | 评估 |
|------|-----------|-------------|------|
| 4 | ~1.7M | ~2.4h | 可能欠拟合 |
| 8 | ~3.4M | ~2.5h | 基础配置 |
| 16 | ~6.8M | ~2.6h | 默认配置 |
| 32 | ~13.6M | ~2.8h | 可能过拟合 |

每个rank配置运行主实验(E1)的20次评估，报告覆盖率和任务完成率。

### 改进4: 通信模型与目标生成

**通信模型**：
- 采用**有限距离通信**：通信半径 R_comm = 20格
- 超出通信距离的队友信息延迟1步更新（使用上一步的位置）
- 这比全局通信更真实，也为鲁棒性测试(E8)提供了自然的测试场景

**目标生成方式**：
- 目标位置：在地图中随机均匀采样，但保证任意两目标间距≥10格
- 目标难度：随机分配 required_uavs ∈ {1, 2, 3}，required_ammo ∈ {1, 2}
- 目标初始状态：未被发现（需要UAV搜索到视野内才可见）

### 改进5: Related Work结构

```
2. Related Work
  2.1 UAV协同任务规划
    - 传统优化方法：ACO, GA, PSO在UAV路径规划中的应用
    - 分布式方法：拍卖算法、共识协议
    
  2.2 深度强化学习在UAV中的应用
    - 单UAV: DQN/PPO用于路径规划
    - 多UAV: MAPPO, QMIX等多智能体方法
    - 局限：样本效率低、不可解释、泛化差
    
  2.3 LLM在决策与规划中的应用
    - LLM作为规划器：SayCan, Inner Monologue, ProgPrompt
    - LLM作为策略：直接输出动作的方法
    - LLM + 微调：领域适配的方法
    
  2.4 指令跟随与可控生成
    - Instruction tuning的发展
    - 可控文本生成到可控决策的迁移
```

### 改进6: 理论动机

**为什么序列决策可以转化为序列生成？**

核心论点：UAV协同任务规划本质上是一个**条件序列生成问题**。

1. **状态-动作序列的马尔可夫性**：在给定当前态势描述的条件下，最优动作只依赖当前状态（和指令），这与自回归语言模型的条件生成机制一致。

2. **LLM的序列建模能力**：预训练LLM已经学会了对复杂序列模式的建模。UAV决策中的空间推理（"NW方向未搜索密度高"）、时序推理（"目标正在被队友接近"）、多约束推理（"弹药不足应优先搜索"）都可以映射到自然语言的推理模式。

3. **指令条件化的理论基础**：条件语言模型 P(action | state, instruction) 天然支持通过改变条件（instruction）来改变输出分布，这为指令调控提供了理论基础。

4. **CoT作为隐式规划**：Chain-of-Thought推理可以看作是在动作空间中的隐式搜索——模型通过逐步推理来缩小候选动作集，类似于传统规划中的启发式搜索。

---

## 更新后的完整实验列表

| 实验 | 目的 | 配置 | 运行次数 | 预计时间 |
|------|------|------|----------|----------|
| E1: 主实验 | 全方法对比 | 80×80, 6UAV, 5目标 | 20次 | 10min |
| E2: 规模泛化 | 地图/UAV数泛化 | 120×120, 10UAV, 8目标 | 10次 | 5min |
| E3: 指令调控 | LLM独特优势 | 同一场景，5种指令 | 10次×5指令 | 8min |
| E4: CoT消融 | CoT价值 | 有CoT vs 无CoT | 20次 | 5min |
| E5: 数据量消融 | 数据效率 | 5k/10k/20k/30k/45k | 各10次 | 需额外训练 |
| E6: 推理延迟 | 实时性 | 测量决策时间 | 100次 | 2min |
| E7: 可解释性案例 | 定性分析 | 展示CoT推理过程 | 5个案例 | 手动 |
| E8: 鲁棒性 | 异常场景 | 4种异常×3方法 | 各10次 | 10min |
| E9: LoRA Rank | 超参数敏感性 | rank=4/8/16/32 | 各20次 | 需额外训练 |

**注**：E5和E9需要额外训练时间。建议在主实验完成后，选择性地运行（优先E9，因为只需改rank重新训练）。

---

## 更新后的Compute Budget

| 阶段 | 设备 | 时间 | 说明 |
|------|------|------|------|
| 数据生成+处理 | CPU | 25min | ACO仿真+反事实扩增+CoT标注 |
| LoRA微调(rank=16) | A100 | 2.6h | 主配置 |
| PPO baseline | CPU | 1h | 与LoRA并行 |
| BC-MLP baseline | A100 | 10min | LoRA完成后 |
| 评估(E1-E4,E6-E8) | A100 | 30min | |
| **主流程总计** | | **~3.3h** | ✓ |
| LoRA rank消融(E9) | A100 | +3×0.5h | 可选，额外1.5h |
| 数据量消融(E5) | A100 | +4×0.8h | 可选，额外3.2h |
