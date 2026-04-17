# OmniSkill 迭代优化日志（10 轮）

> 日期: 2026-04-10  
> 模板参照: `D:\北航工作日常\ICGNC\文献\` 下 ICGNC 终稿 PDF（`0156_final.pdf`、`0158_final.pdf`、`0159_final(1).pdf`、`0165_final.pdf`、`0166_final.pdf`）之版式与行文密度；工程内 `模板/ICGNC_TeX`（Springer `svproc`）为 TeX 结构参照。  
> 目标稿: `Paper/main.tex` 及各 `sec_*.tex`。

---

## 第 1 轮（公式：TAGA 推导链）

- 将原「在 $\exp$ 内连乘 $e_{ij}$ 与三因子」改写为两步：先定义模态亲和度 $s_{ij}^{(k)}$（式 `\eqref{eq:taga_s}`），再对 $\mathcal{N}_i$ 做标准 softmax 得 $\alpha_{ij}^{\text{TAGA},(k)}$（式 `\eqref{eq:taga}`）。
- 与标准 GAT 的「logit → softmax」顺序对齐，便于审稿人核对归一化与邻域约束。

## 第 2 轮（公式：邻域与符号一致）

- 在式 `\eqref{eq:taga}` 后明确：仅 $j\in\mathcal{N}_i$ 参与分母求和，与 `\mathcal{E}_t`、距离门限叙述一致。
- 多头上统一使用 $\alpha_{ij}^{\text{TAGA},(k)}$，与式 `\eqref{eq:taga_agg}` 中按头加权求和一致。

## 第 3 轮（公式：PPO 目标符号）

- 在式 `\eqref{eq:ppo}` 处补充说明：首项负号将「裁剪代理目标最大化」化为与梯度下降一致的「损失最小化」，避免与常见 PPO 叙述符号相反时的误读。

## 第 4 轮（公式/叙述：提示编码与 TAGA 一致）

- 将邻居表「H/M/L 仅由 $f_d$ 映射」改为「由 $(f_d, f_t)$ 的联合分位阈值」生成离散档位，与文中同时展示信号质量与新鲜度一致，避免逻辑缺口。

## 第 5 轮（公式：共享奖励语义）

- 在式 `\eqref{eq:reward}` 后说明各 $r_t^{\text{cov}},\ldots,r_t^{\text{pen}}$ 为逐步增量且通常稀疏，与 Dec-POMDP 中共享标量奖励的常见写法对齐。

## 第 6 轮（格式：节引用）

- 将正文中的 `Sec.~\ref` 统一为 `Sect.~\ref`；多节范围使用 `Sects.~\ref{sec:taga}--\ref{sec:mappo}`，与引言中 `Sect.~\ref{...}` 体例一致。

## 第 7 轮（格式：实验节占位）

- `sec_experiments.tex` 中各 `\emph{To be completed:...}` 子句统一为分号分隔的并列结构，列举项使用 `(i)~...` 与分号，减少逗号嵌套歧义。

## 第 8 轮（格式：算法伪代码）

- 算法步骤标题由 `\textbf{Stage ...}` 改为 `\textit{Stage ...}`，满足「正文不使用加粗」之排版要求（与 `svproc` 常见算法体例兼容）。

## 第 9 轮（格式：摘要与引言）

- 去除 `main.tex` 摘要、`sec_introduction.tex` 中 `\textbf{...}`（如 TAGA-LLM、贡献条目标记），改为普通正文字体，避免与会议稿「正文少强调」习惯冲突。

## 第 10 轮（格式：交叉引用与公式编号）

- 算法中公式引用更新为 `Eq.~\ref{eq:taga_s}--\ref{eq:taga_agg}`，覆盖新增式 `\eqref{eq:taga_s}`。
- 当前主链编号顺序：`eq:taga_s` → `eq:taga` → `eq:taga_agg` → `eq:ppo` → `eq:sft` → `eq:reward`（后四者与原稿一致）。

---

## 本轮修改文件清单

| 文件 | 说明 |
|------|------|
| `main.tex` | 摘要去掉 `\textbf` |
| `sec_introduction.tex` | 引言去掉 `\textbf` |
| `sec_problem.tex` | 奖励式后补充稀疏增量语义 |
| `sec_method.tex` | TAGA 两式推导、PPO 说明、提示 H/M/L、节引用、算法体 |
| `sec_experiments.tex` | 占位段落标点与列举格式统一 |

## 编译检查（建议本地执行）

在 `Paper/` 目录：`pdflatex main.tex` → `bibtex main` → `pdflatex` ×2，确认无 undefined references。
