# Round 4 — 公式规范 + 符号一致性

> 日期: 2026-04-08

## 审阅发现与修改

### 1. 注意力公式 head 上标不一致

- **问题**: Eq. (1) 定义 `e_{ij}^{(k)}`，但 Eq. (2) 中使用 `e_{ij}` 无上标
- **修改**: 在 Eq. (2) 前添加说明 "Dropping the head superscript (k) for clarity"，明确省略约定

### 2. GAE λ 与 TAGA λ 符号冲突

- **问题**: PPO 的 GAE 参数用 `λ = 0.95`，与 TAGA 的 `λ_d, λ_m, λ_t` 使用相同希腊字母
- **修改**: 将 GAE 参数改为 `λ_{GAE} = 0.95`，避免混淆

### 3. 残差连接维度不匹配

- **问题**: 第一层 TAGA 从 128→64 维度，直接残差连接 `h_i + W_O[...]` 维度不匹配
- **修改**: 引入 `W_R` 投影矩阵用于维度不匹配时的残差连接，并说明第二层（64→64）退化为恒等映射

### 4. 符号一致性检查通过

- `N` = UAV 数量：sec_problem (N=10) ↔ sec_method (N agents) ✅
- `G` = 网格大小：sec_problem (G=100) ✅
- `R_comm` = 通信半径：sec_problem (R_comm=20) ↔ sec_method (R_comm=20) ✅
- `d_{ij}` = 距离：sec_problem ↔ sec_method ✅
- `m_i` = 模式：sec_problem (search/attack) ↔ sec_method ✅
- `o_i` = 观测：sec_problem ↔ sec_method ✅
- `γ` = 折扣因子：sec_problem (0.99) ↔ sec_method (0.99) ✅

### 5. 公式编号检查

- Eq. (1): GAT attention logit — 有 label ✅
- Eq. (2): TAGA attention — 有 label ✅
- Eq. (3): CNN encoder — 有 label ✅
- Eq. (4): MLP encoder — 有 label ✅
- Eq. (5): TAGA aggregation — 有 label ✅
- Eq. (6): Reward function — 有 label ✅
- Eq. (7): PPO objective — 有 label ✅
- Eq. (8): Total loss — 有 label ✅
- Eq. (9): SFT loss — 有 label ✅

## 修改文件

- `sec_method.tex`: 3处修正（head 上标说明、GAE λ 重命名、残差投影）