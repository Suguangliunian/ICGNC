# ARIS-A Round 2 — 创新提案

> 日期: 2026-04-08
> 针对薄弱点: Method Specificity (7), Contribution Quality (7)

---

## 创新改进 1: TAGA理论动机强化

**问题**: TAGA三因子组合缺乏理论支撑。

**改进**: 从信息论角度给出TAGA的理论动机。

**论证**：在分布式MARL中，UAV i 从邻居 j 获取的有效信息量 I_ij 受三个因素约束：

1. **通信信道质量** ∝ f(距离): 无线通信中，信号强度随距离平方衰减（自由空间路径损耗），因此远距离邻居的信息可靠性低
2. **信息相关性** ∝ f(任务模式): 同模式UAV面临相似的决策问题，其观测信息对当前UAV的决策更相关（互信息更高）
3. **信息时效性** ∝ f(通信延迟): 在动态环境中，过时的信息可能导致错误决策（信息熵随时间增加）

因此 TAGA 的注意力权重可以理解为对"有效信息量"的近似估计：
```
α_ij^TAGA ≈ P(信息有效 | 距离, 模式匹配, 时效性)
```

这不是简单的拼凑，而是对分布式通信约束的自然建模。

## 创新改进 2: 观测编码器升级

**问题**: CNN感受野不足。

**改进**: 
- 增加一层卷积（3层CNN），感受野扩大到7×7，覆盖搜索半径
- 或使用5×5卷积核替代3×3

选择方案：使用2层5×5卷积
```
Conv2d(5, 32, 5, padding=2) + ReLU + MaxPool(2)  # 感受野5×5
Conv2d(32, 64, 5, padding=2) + ReLU + MaxPool(2)  # 感受野13×13（覆盖全patch）
Flatten → Linear(64*2*2, 128)
```

## 创新改进 3: 动作合法性mask

**问题**: 未定义动作合法性处理。

**改进**: 在策略输出前应用动作mask：
```python
def get_action_mask(state):
    mask = torch.ones(20)  # 20个联合动作
    # 边界检查：禁止移出地图
    if at_north_border: mask[north_actions] = 0
    # 弹药检查：无弹药时禁止攻击模式
    if ammo == 0: mask[attack_actions] = 0
    # 威胁规避：已知威胁区方向降低概率（soft mask）
    for threat_dir in known_threats:
        mask[threat_dir_actions] *= 0.1
    return mask
```
