# Round 3 Refinement — Chat Template适配 + 反事实数据 + 定量评估

> 日期: 2026-04-08
> 改进重点: 使用Qwen原生chat template、完善反事实数据生成、加入指令调控定量指标
> 上轮综合分: 7.4/10

---

## 改进摘要

1. **【紧急】使用Qwen原生chat template替代自定义special tokens**
2. **【重要】完善反事实数据生成方法**
3. **【重要】加入指令调控的定量评估指标**
4. **【建议】加入鲁棒性测试和failure case分析**
5. **【建议】优化PPO训练的资源分配**

---

## 核心改进

### 改进1: 使用Qwen原生Chat Template

**问题**：Round 2中使用了自定义special tokens（`<|system|>`, `<|view|>`等），可能导致：
- tokenizer切分不一致
- 与预训练权重不兼容
- 需要resize embedding层，破坏预训练表示

**修正方案**：完全使用Qwen3.5的原生chat template格式。

**新Prompt格式**：

```
<|im_start|>system
你是UAV集群协同决策引擎。根据战场态势和任务指令，为指定UAV生成决策。
任务指令: {instruction}
<|im_end|>
<|im_start|>user
[态势]
UAV-{id} 位于({x},{y}), 模式={m}, 弹药={a}, 步数={t}
地图: {L}x{W}网格

[视野] (7x7, U=未搜索 S=已搜索 T=目标 *=自身)
+3:UUUSSTU
+2:USSSSU.
+1:SSSSSSS
 0:SS*SSSS
-1:SSSUUUU
-2:UUUUUUU
-3:UUUUUUT

[摘要] 覆盖率=45%, 最优方向=NW(0.9), 最近未搜索=3格

[队友]
U3:(12,8)朝N,搜索中,距5 | U1:(20,15)朝E,攻击中,距12

[目标]
T1:(30,25)需2机1弹,距18 | T2:(45,10)需1机0弹,距8

[已决策] U1:NW,搜索 | U2:E,搜索

[历史动作] N,N,NE,E,搜索
<|im_end|>
<|im_start|>assistant
[思考] 覆盖率偏低，应优先搜索。NW方向未搜索密度最高(0.9)。T2距离较近但已有U1前往。无队友冲突。
[动作] NW,搜索,无目标
<|im_end|>
```

**关键设计决策**：
- 所有标记使用Qwen原生的 `<|im_start|>` / `<|im_end|>` 格式
- 用中文方括号标签 `[态势]` `[视野]` `[思考]` `[动作]` 替代自定义special tokens
- 视野矩阵保持紧凑的ASCII格式不变
- `[思考]` 替代 `<|think|>`，使用自然语言短句而非压缩关键词链

**Token数估算**：
- system部分：~40 tokens
- user部分：~220 tokens
- assistant部分：~50 tokens
- 总计：~310 tokens（与Round 2基本持平）

**训练时间影响**：无显著变化，仍为~1.7h。

### 改进2: 反事实数据生成方法

**问题**：Round 2中"反事实数据"概念模糊——同一状态下如何生成不同指令对应的不同动作？

**具体方法**：

**Step 1: 从ACO轨迹提取状态-动作对**
```python
# 原始数据: (state, action) from ACO
# action = (direction, mode, target_id)
```

**Step 2: 对每个状态，用规则生成多指令-动作对**

```python
def generate_counterfactual(state, original_action):
    """为同一状态生成不同指令下的合理动作"""
    pairs = []
    
    # 原始动作 → 匹配最合适的指令
    pairs.append((match_instruction(state, original_action), original_action))
    
    # 反事实1: 搜索优先指令
    if state.has_nearby_targets():
        search_action = compute_best_search_action(state)
        pairs.append(("I1_search_priority", search_action))
    
    # 反事实2: 攻击优先指令
    if state.has_known_targets():
        attack_action = compute_best_attack_action(state)
        pairs.append(("I2_attack_priority", attack_action))
    
    # 反事实3: 安全优先指令
    if state.has_threat_nearby():
        safe_action = compute_safest_action(state)
        pairs.append(("I4_safety_priority", safe_action))
    
    return pairs
```

**Step 3: 反事实动作的生成规则**

| 指令 | 动作生成规则 | 适用条件 |
|------|-------------|----------|
| I1 搜索优先 | 选择未搜索密度最高方向 + mode=搜索 | 视野内有未搜索区域 |
| I2 攻击优先 | 朝最近已知目标移动 + mode=攻击 | 存在已知目标 |
| I3 平衡策略 | 保持ACO原始动作 | 默认 |
| I4 安全优先 | 远离威胁区域方向 + mode=搜索 | 附近有威胁 |
| I5 区域聚焦 | 朝指定区域中心移动 | 指定区域与当前位置不同 |

**数据量控制**：
- 原始ACO数据：30k条
- 反事实扩增后：~50k条（平均每条原始数据生成1.7条反事实）
- 但只取有意义的反事实（原始动作与反事实动作不同的情况）
- 最终训练数据：~45k条

**训练时间修正**：
- 45k × 2 epochs × 310 tokens = 27.9M tokens
- 27.9M / 3000 tokens/s ≈ 9300s ≈ **2.6小时**
- 仍在预算内 ✓

### 改进3: 指令调控定量评估指标

**E3实验的定量指标体系**：

| 指标 | 定义 | 计算方法 |
|------|------|----------|
| 搜索-攻击比 (SAR) | 搜索动作占总动作的比例 | `n_search / (n_search + n_attack)` |
| 方向一致性 (DC) | 动作方向与指令意图的一致程度 | 余弦相似度(移动方向, 指令目标方向) |
| 策略熵 (PE) | 动作分布的多样性 | `-Σ p(a) log p(a)` |
| 指令响应度 (IR) | 切换指令后策略变化的幅度 | `KL(π_new || π_old)` |
| 目标响应时间 (TRT) | 发现目标到开始攻击的步数 | `t_attack_start - t_target_found` |

**预期结果模式**：

| 指令 | SAR↑ | DC↑ | PE | TRT |
|------|------|-----|-----|-----|
| I1 搜索优先 | >0.9 | 高(朝未搜索区) | 中 | 长 |
| I2 攻击优先 | <0.3 | 高(朝目标) | 低 | 短 |
| I3 平衡 | 0.5-0.7 | 中 | 高 | 中 |
| I4 安全优先 | >0.8 | 高(远离威胁) | 低 | 长 |

**统计检验**：对每对指令组合做Wilcoxon秩和检验，验证SAR和TRT的差异显著性（p<0.05）。

### 改进4: 鲁棒性测试

新增实验 **E8: 鲁棒性测试**：

| 异常场景 | 具体设置 | 评估指标 |
|----------|----------|----------|
| UAV损失 | 任务中随机移除1-2架UAV | 任务完成率下降幅度 |
| 通信延迟 | 队友信息延迟2-5步 | 覆盖率和攻击成功率 |
| 目标移动 | 目标以1格/步速度随机移动 | 任务完成时间 |
| 传感器噪声 | 视野中5%格子信息错误 | 决策准确率 |

**Failure Case分析框架**：
- 收集所有任务失败的episode
- 分类失败原因：(a)协调失败 (b)目标遗漏 (c)弹药耗尽 (d)CoT推理错误
- 对每类失败，展示典型的CoT推理过程，分析LLM在哪一步出错

### 改进5: 资源分配优化

**修正后的Compute Budget**：

| 阶段 | 设备 | 时间 | 说明 |
|------|------|------|------|
| ACO数据生成 | CPU | 15min | 并行仿真 |
| 反事实数据扩增 | CPU | 5min | 规则生成 |
| CoT标注 | CPU | 5min | 模板填充 |
| LoRA微调 | A100 GPU | 2.6h | 45k数据, 2 epochs |
| PPO baseline | CPU (多核) | 1h | 环境仿真为主，不需GPU |
| BC-MLP baseline | A100 GPU | 10min | 简单MLP训练 |
| 全部评估 | A100 GPU | 30min | 推理为主 |
| **总计** | | **~3.2h** | GPU占用2.6h+0.7h=3.3h ✓ |

**关键优化**：PPO训练移至CPU（环境仿真是瓶颈，不需要GPU），GPU专用于LoRA微调和评估。

---

## 更新后的仿真环境参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 地图尺寸 | 80×80 网格 | 主实验 |
| 时间步长 | 离散步，每步1个决策周期 | |
| UAV速度 | 1格/步 | 8方向移动 |
| 传感器范围 | 7×7视野（以自身为中心） | |
| 弹药模型 | 每架UAV携带2枚，攻击消耗1枚 | |
| 目标消灭条件 | 需要≥n架UAV同时在目标相邻格 | n由目标难度决定(1-3) |
| 最大步数 | 200步 | 超时视为失败 |
| UAV数量 | 6架（主实验） | |
| 目标数量 | 5个（主实验） | |

---

## 更新后的论文标题

**精简版**：
> LLM-Steered: Instruction-Tuned Language Model for Cooperative UAV Mission Planning

**副标题**（可选）：
> with Chain-of-Thought Reasoning and Zero-Shot Generalization
