# ARIS-B Round 2 — State/Action Tokenization优化

> 日期: 2026-04-08
> 上轮综合分: 8.8/10
> 本轮重点: 优化token效率，减少prompt长度，提高信息密度

---

## Diagnose

| 问题 | 严重程度 |
|------|----------|
| Prompt过长（~350 tokens），推理延迟高 | IMPORTANT |
| 邻居信息的自然语言描述冗余（"Same mode as you"等） | IMPORTANT |
| 视野信息编码效率低（逐格列举） | MINOR |
| 动作输出格式可以更紧凑 | MINOR |

---

## Innovation: 紧凑Token编码

### 改进1: 压缩邻居信息编码

**Round 1的冗余格式** (~80 tokens for 3 neighbors):
```
[NEIGHBORS] (sorted by comm quality)
UAV-3: pos=(45,67) mode=search dist=8km signal=strong freshness=2steps_ago
  → Same mode as you. Recent info. High reliability.
...
```

**压缩格式** (~40 tokens for 3 neighbors):
```
[NEIGHBORS]
#3 (45,67) S d=8 sig=H fresh=2 | #7 (52,71) A d=12 sig=M fresh=1 | #1 (60,80) S d=18 sig=L fresh=5
[COMM] 3nb 2S/1A avg=M
```

编码规则：
- S=search, A=attack
- sig: H=strong(d<10), M=good(10≤d<15), L=weak(15≤d<20)
- fresh: 步数差
- 用 `|` 分隔邻居，一行搞定

### 改进2: 视野信息压缩

**原格式** (~60 tokens):
```
[LOCAL_MAP]
Searched: [(43,65),(44,65),(45,65),(43,66),(44,66),(45,66),...]
Unsearched_Nearby: [(46,68),(47,68),(48,68),...]
```

**压缩格式** (~25 tokens):
```
[MAP] 7x7 view: searched=28/49(57%) unsearched_dir=NE,E threats=SW(d=3)
```

只保留统计摘要和方向性信息，不逐格列举。

### 改进3: 动作输出压缩

**原格式**:
```
[动作] move_NE [模式] search [目标] none
```

**压缩格式**:
```
[A] NE S -
```

编码：方向(N/NE/E/SE/S/SW/W/NW/STAY) + 模式(S/A) + 目标(T1/T2/-表示none)

### 改进4: 完整压缩Prompt模板

```
<|im_start|>system
UAV mission planner. Output: [思考]...[A] dir mode target
<|im_end|>
<|im_start|>user
[CMD] {instruction_short}
[ME] ({x},{y}) {mode} ammo={a} t={t}
[MAP] {grid_size} searched={pct}% unsearched={dirs} threats={threat_dirs}
[NB] {neighbor_compact_list}
[COMM] {n}nb {mode_dist} avg={sig}
[TGT] {target_compact_list}
[HIST] {last_5_actions_compact}
<|im_end|>
<|im_start|>assistant
[思考] {cot}
[A] {dir} {mode} {target}
<|im_end|>
```

**Token估算**:
- system: ~15 tokens
- user prompt: ~120 tokens (从~250压缩)
- CoT + action: ~60 tokens
- **总计: ~195 tokens** (从~350压缩，节省44%)

### 改进5: 截断优先级（当>256 tokens时）

| 优先级 | 截断对象 | 策略 |
|--------|----------|------|
| 1 | [HIST] | 只保留最近3步 |
| 2 | [NB] | 只保留最近2个邻居 |
| 3 | [TGT] | 只保留最近2个目标 |
| 4 | [思考] | 限制为1句话 |

---

## 评分

| 维度 | B-R1 | B-R2 | 变化 |
|------|------|------|------|
| Problem Fidelity | 9 | 9 | → |
| Method Specificity | 9 | 9 | → (token编码更具体) |
| Contribution Quality | 9 | 9 | → |
| Frontier Leverage | 9 | 9 | → |
| Feasibility | 9 | 9 | → (token减少→训练更快) |
| Validation Focus | 8 | 8 | → |
| Venue Readiness | 8 | 8 | → |
| **综合** | **8.8** | **8.85** | **↑0.05** |

Token压缩本身不直接提升评分，但为后续的推理延迟优化和训练效率提升打下基础。
