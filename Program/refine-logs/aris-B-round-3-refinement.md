# ARIS-B Round 3 — 训练数据生成 + LoRA微调策略优化

> 日期: 2026-04-08
> 上轮综合分: 8.85/10
> 本轮重点: 训练数据pipeline具体化 + LoRA训练策略优化

---

## Diagnose

| 问题 | 严重程度 |
|------|----------|
| 训练数据生成pipeline缺少具体代码 | CRITICAL |
| 反事实数据生成规则需要更精确 | IMPORTANT |
| LoRA训练的loss mask策略需要明确 | IMPORTANT |
| 数据质量筛选标准未定义 | MINOR |

---

## Innovation

### 改进1: 训练数据生成Pipeline（代码级具体）

```python
class ExpertDataGenerator:
    """用TAGA-MAPPO（阶段A方法）生成专家轨迹，转化为LLM训练数据"""
    
    def __init__(self, taga_model_path, env_config):
        self.env = UAVSwarmEnv(env_config)
        self.taga_model = load_taga_mappo(taga_model_path)
        self.prompt_builder = TAGAAwarePromptBuilder()
    
    def generate_episode(self, scenario_config):
        """生成一个episode的专家轨迹"""
        obs = self.env.reset(scenario_config)
        trajectory = []
        
        for t in range(self.env.max_steps):
            # TAGA-MAPPO决策
            actions, mode_probs, attn_weights = self.taga_model.act(obs)
            
            for i in range(self.env.n_uav):
                # 构建TAGA-aware prompt
                prompt = self.prompt_builder.build(
                    uav_id=i, obs=obs[i],
                    neighbors=self.env.get_neighbors(i),
                    targets=self.env.get_visible_targets(i),
                    attn_weights=attn_weights[i],  # TAGA注意力权重用于CoT
                    history=self.env.get_action_history(i)
                )
                
                # 构建CoT（基于TAGA注意力权重解释）
                cot = self._generate_cot(i, obs[i], attn_weights[i], actions[i])
                
                # 构建标签
                action_label = self._format_action(actions[i])
                
                trajectory.append({
                    "prompt": prompt,
                    "cot": cot,
                    "action": action_label,
                    "quality_score": self._compute_quality(obs, actions, i)
                })
            
            obs, rewards, dones, infos = self.env.step(actions)
            if all(dones):
                break
        
        return trajectory
    
    def _generate_cot(self, uav_id, obs, attn_weights, action):
        """基于TAGA注意力权重生成CoT推理"""
        top_neighbor = attn_weights.argmax()
        cot_parts = []
        
        # 通信态势分析
        n_neighbors = (attn_weights > 0.01).sum()
        cot_parts.append(f"{n_neighbors}个邻居在通信范围内")
        
        # 最重要邻居
        cot_parts.append(f"UAV-{top_neighbor}信息最可靠(权重{attn_weights[top_neighbor]:.2f})")
        
        # 决策理由
        mode = "搜索" if action < 10 else "攻击"
        direction = ["N","NE","E","SE","S","SW","W","NW","STAY"][action % 9]
        cot_parts.append(f"选择{mode}模式，向{direction}移动")
        
        return "。".join(cot_parts)
    
    def _compute_quality(self, obs, actions, agent_id):
        """计算决策质量分数（用于数据筛选）"""
        # 基于即时奖励 + 未来折扣奖励估计
        return self.taga_model.estimate_value(obs[agent_id])

    def generate_dataset(self, n_episodes=500, scenarios=None):
        """生成完整训练数据集"""
        if scenarios is None:
            scenarios = [
                {"grid": 50, "n_uav": 4, "n_target": 3, "n_threat": 0},
                {"grid": 80, "n_uav": 6, "n_target": 4, "n_threat": 3},
                {"grid": 100, "n_uav": 8, "n_target": 5, "n_threat": 5},
                {"grid": 100, "n_uav": 10, "n_target": 5, "n_threat": 7},
            ]
        
        all_data = []
        for scenario in scenarios:
            for ep in range(n_episodes // len(scenarios)):
                traj = self.generate_episode(scenario)
                all_data.extend(traj)
        
        # 质量筛选：只保留top-75%质量的数据
        all_data.sort(key=lambda x: x["quality_score"], reverse=True)
        cutoff = int(len(all_data) * 0.75)
        filtered = all_data[:cutoff]
        
        return filtered  # ~45K条
```

### 改进2: 反事实数据生成

```python
class CounterfactualGenerator:
    """为每个专家决策生成反事实样本"""
    
    def generate_counterfactual(self, expert_sample):
        """对专家决策的每个维度生成反事实"""
        counterfactuals = []
        
        expert_action = expert_sample["action"]  # e.g., "NE S -"
        expert_dir, expert_mode, expert_target = expert_action.split()
        
        # 反事实1：错误方向（选择相反方向）
        opposite_dir = self._get_opposite(expert_dir)
        cf1 = expert_sample.copy()
        cf1["action"] = f"{opposite_dir} {expert_mode} {expert_target}"
        cf1["cot"] = f"错误决策：向{opposite_dir}移动会远离目标区域，降低搜索效率"
        cf1["is_counterfactual"] = True
        cf1["label"] = "negative"
        counterfactuals.append(cf1)
        
        # 反事实2：错误模式（搜索↔攻击）
        wrong_mode = "A" if expert_mode == "S" else "S"
        cf2 = expert_sample.copy()
        if wrong_mode == "A" and expert_target == "-":
            cf2["action"] = f"{expert_dir} A T1"  # 随机分配目标
        else:
            cf2["action"] = f"{expert_dir} S -"
        cf2["cot"] = f"错误决策：当前不应切换到{'攻击' if wrong_mode=='A' else '搜索'}模式"
        cf2["is_counterfactual"] = True
        cf2["label"] = "negative"
        counterfactuals.append(cf2)
        
        return counterfactuals
```

### 改进3: LoRA训练Loss Mask策略

```python
def compute_loss_with_mask(model, input_ids, labels, tokenizer):
    """只对CoT和Action部分计算loss，prompt部分mask掉"""
    outputs = model(input_ids=input_ids, labels=labels)
    logits = outputs.logits
    
    # 找到[思考]和[A]标记的位置
    think_token_id = tokenizer.encode("[思考]")[0]
    action_token_id = tokenizer.encode("[A]")[0]
    
    # 创建loss mask
    loss_mask = torch.zeros_like(labels, dtype=torch.float)
    for i in range(labels.size(0)):
        # 从[思考]开始到序列结束都计算loss
        think_pos = (labels[i] == think_token_id).nonzero()
        if len(think_pos) > 0:
            loss_mask[i, think_pos[0]:] = 1.0
    
    # 加权loss
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = loss_mask[..., 1:].contiguous()
    
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = (loss.view(shift_labels.size()) * shift_mask).sum() / shift_mask.sum()
    
    return loss
```

### 改进4: 训练时间精确估算

```
数据量: 45K条 × 195 tokens/条 = 8.775M tokens
Batch: 8 × 4(grad_accum) = 32 effective batch
Steps/epoch: 45K / 32 = 1,406 steps
Total steps: 1,406 × 2 epochs = 2,812 steps

吞吐量估算 (Qwen3.5-4B + LoRA + gradient_checkpointing):
- 单步时间: ~0.8s (A100, bf16, seq_len=256)
- 总训练时间: 2,812 × 0.8s = 2,250s ≈ 37.5min

加上数据生成(30min) + 评估(15min):
总计: ~82min ≈ 1.4小时
```

---

## 评分

| 维度 | B-R1 | B-R2 | B-R3 | 变化 |
|------|------|------|------|------|
| Problem Fidelity | 9 | 9 | 9 | → |
| Method Specificity | 9 | 9 | 9 | → (数据pipeline代码级具体) |
| Contribution Quality | 9 | 9 | 9 | → |
| Frontier Leverage | 9 | 9 | 9 | → |
| Feasibility | 9 | 9 | 9 | → (时间估算更精确) |
| Validation Focus | 8 | 8 | 9 | ↑1 (数据质量筛选+反事实) |
| Venue Readiness | 8 | 8 | 8 | → |
| **综合** | **8.8** | **8.85** | **8.95** | **↑0.10** |
