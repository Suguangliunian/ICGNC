# Round 4 Review — 工程完备性与实验严谨性评审

> 评审日期: 2026-04-08
> 评审对象: qwen-round-3-refinement.md

---

## 评分总览

| 维度 | R0 | R1 | R2 | R3 | 变化 | 评价 |
|------|----|----|-----|-----|------|------|
| 1. Problem Fidelity | 8 | 8 | 8 | 9 | ↑1 | 仿真参数完整 |
| 2. Method Specificity | 5 | 7 | 8 | 9 | ↑1 | chat template + 反事实方法具体 |
| 3. Contribution Quality | 6 | 6 | 7 | 8 | ↑1 | 定量指标体系完善 |
| 4. Frontier Leverage | 5 | 7 | 8 | 8 | → | 稳定 |
| 5. Feasibility | 7 | 7 | 8 | 9 | ↑1 | CPU/GPU分离方案合理 |
| 6. Validation Focus | 5 | 7 | 7 | 8 | ↑1 | 鲁棒性+failure case加入 |
| 7. Venue Readiness | 4 | 5 | 6 | 7 | ↑1 | 标题精简，故事完整 |
| **综合** | **5.7** | **6.7** | **7.4** | **8.3** | **↑0.9** | |

---

## 详细评审

### 1. Problem Fidelity (9/10)
**显著改善**：
- 仿真环境参数表完整（地图、速度、传感器、弹药、目标消灭条件）
- 时间步长、最大步数等关键参数明确

**微调建议**：
- 补充UAV间通信模型：是否全局通信？还是有通信距离限制？
- 明确目标初始位置的生成方式：随机均匀分布？还是有聚类？

### 2. Method Specificity (9/10)
**显著改善**：
- 使用Qwen原生chat template，完全避免了自定义special tokens的风险
- 反事实数据生成方法具体可执行，规则清晰
- token数估算合理

**微调建议**：
- `[思考]` 部分的训练目标需要更明确：是让模型学会生成"正确的推理过程"，还是"任何合理的推理过程"？
  - 建议：只要最终动作正确，推理过程的loss权重可以更低（0.2而非0.3）
- 反事实数据中，`compute_best_search_action` 和 `compute_best_attack_action` 的具体实现需要明确
  - 建议：搜索动作=朝未搜索密度最高方向；攻击动作=朝最近可攻击目标的A*路径第一步

### 3. Contribution Quality (8/10)
**显著改善**：
- 指令调控的定量指标体系（SAR, DC, PE, IR, TRT）完整且可计算
- 统计检验方法明确

**微调建议**：
- 指令响应度(IR)用KL散度衡量，但需要明确"策略分布"的定义——是动作的经验分布？还是模型输出的logits分布？
  - 建议：使用动作的经验分布（在10次运行中统计动作频率），更直观
- 需要加入一个"指令忠实度"指标：模型是否真的按指令行事，而非忽略指令
  - 定义：在I1(搜索优先)下，附近有目标时仍选择搜索的比例

### 4. Frontier Leverage (8/10)
- 保持稳定
- chat template的使用更好地利用了Qwen的预训练对话能力
- 建议在论文中讨论：为什么选择Qwen3.5-4B而非其他模型（Llama, Mistral等）？
  - 理由：中文支持好、4B参数量适合单卡LoRA、chat template成熟

### 5. Feasibility (9/10)
**显著改善**：
- CPU/GPU分离方案合理，总时间3.2h在预算内
- PPO移至CPU是正确决策

**微调建议**：
- 反事实数据扩增后45k条，训练时间2.6h——建议预留0.5h的buffer用于调试
- LoRA超参数需要明确：rank, alpha, target modules
  - 建议：rank=16, alpha=32, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
- 学习率调度：建议cosine schedule with warmup（warmup_ratio=0.05）

### 6. Validation Focus (8/10)
**显著改善**：
- 鲁棒性测试覆盖了4种异常场景
- Failure case分析框架系统化

**微调建议**：
- E8鲁棒性测试的运行次数需要明确（建议每种场景10次）
- 需要明确：鲁棒性测试是否对所有方法都做？还是只对LLM方法？
  - 建议：至少对ACO、PPO、Ours三种方法都做，以展示LLM的鲁棒性优势
- 缺少一个关键实验：**LoRA rank消融**（rank=4/8/16/32），验证rank=16是否最优

### 7. Venue Readiness (7/10)
**改善**：标题精简，故事线完整

**仍需改进**：
- 需要明确目标venue：建议ICRA/IROS（机器人）或AAAI/IJCAI（AI）
- 需要Related Work的结构：
  - (a) UAV任务规划传统方法
  - (b) DRL在UAV中的应用
  - (c) LLM在决策/规划中的应用
  - (d) 指令跟随与可控生成
- 需要一个清晰的Method图（architecture diagram）的描述
- 缺少理论动机：为什么序列决策问题可以转化为序列生成问题？

---

## 本轮重点改进方向

1. **【重要】明确LoRA超参数和训练细节**
2. **【重要】补充反事实动作的具体计算规则**
3. **【重要】加入LoRA rank消融实验**
4. **【建议】补充通信模型和目标生成方式**
5. **【建议】构建Related Work结构**
6. **【建议】加入理论动机段落**
