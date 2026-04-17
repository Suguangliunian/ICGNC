# ARIS 迭代优化日志 (5轮)

> 日期: 2026-04-09

## 第1轮：公式推导严谨性 + 字体格式规范

### 公式修复

1. **Eq.(1) TAGA 注意力**：将 `\operatorname{softmax}_j` 展开为完整的 softmax 分式，明确求和范围 $j' \in \mathcal{N}_i$
2. **Eq.(2) 聚合公式**：修复符号不一致——$\alpha_{ij}^{(k)}$ → $\alpha_{ij}^{\text{TAGA},(k)}$，明确拼接符号
3. **Eq.(3) PPO 目标**：定义 importance sampling ratio $r_t(\theta)$，添加 GAE 引用
4. **Eq.(4) SFT 损失**：定义 mask 集合 $\mathcal{M}$，使用 `\mid` 替代 `|`

### 字体格式修复

- 条件概率统一使用 `\mid` 而非 `|`
- 向量/矩阵统一使用 `\mathbf`
- 算子统一使用 `\operatorname`

## 第2轮：跨节一致性 + 结构完善

1. Problem Formulation 中添加 $\Delta t_{ij}$ 定义
2. Reward 部分添加正式公式 Eq.(reward)
3. Dec-POMDP 定义添加下标范围
4. Introduction 添加论文组织段落
5. Conclusion 中指令格式统一为 `\textsc`

## 第3轮：深度公式推导检查

1. 维度一致性：$\mathbf{h}_i \in \mathbb{R}^d$ → $\mathbb{R}^{128}$
2. $\Delta t_{ij}$ 去重：method 中改为引用 problem 定义
3. 指令缩写格式统一为 `\textsc`
4. $d_k$ 定义明确化

## 第4轮：Related Work 增强 + 引用完整性

1. 新增 "Knowledge distillation for policy transfer" 段落
2. 添加 Rusu 2016 policy distillation 引用
3. 添加 Schulman 2016 GAE 引用
4. Abstract 句间距修复

## 第5轮：最终全面审查

1. $\mathbf{W}_Q^{(k)}$ 维度标注明确化
2. 拼接符号 `\Vert` → `\concat` 自定义命令
3. 全文交叉检查通过

## 修改文件清单


| 文件                     | 修改内容                                      |
| ---------------------- | ----------------------------------------- |
| `main.tex`             | 添加 `\concat` 命令，Abstract 格式修复             |
| `sec_introduction.tex` | 添加论文组织段落                                  |
| `sec_related.tex`      | 新增知识蒸馏段落                                  |
| `sec_problem.tex`      | Dec-POMDP 下标、$\Delta t_{ij}$ 定义、Reward 公式 |
| `sec_method.tex`       | 4个公式全面修复、维度一致性、符号统一                       |
| `sec_conclusion.tex`   | 指令格式 `\textsc`、符号引用                       |
| `references.bib`       | 新增 schulman2016gae、rusu2016policy         |
