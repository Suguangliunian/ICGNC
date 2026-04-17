# OmniSkill 迭代日志：实验复现 + `main.tex`（10 轮）

> 日期: 2026-04-10  
> 产出: `Paper/data/quick_experiment_results.json`、`Paper/figures/paper_exp/*`、`sec_experiments.tex`、`main.tex`（摘要 + `\graphicspath`）、`Program/scripts/paper_quick_experiment.py`

## 第 1 轮（可复现入口）

- 新增 `Program/scripts/paper_quick_experiment.py`：一键随机基线、短训、绘图、贪心评估、JSON 落盘，避免手工拼命令。

## 第 2 轮（训练 API）

- `train_and_plot.train_with_logging(..., return_trainer=True)` 可选返回 `MAPPOTrainer`，避免为保存权重重复整段训练循环。

## 第 3 轮（评估与训练一致）

- `MAPPOEvaluator.evaluate_episode` 补齐 TAGA 前向（原先用零向量占位 GAT 特征），与 `MAPPOTrainer._get_features` 语义对齐；并用 `env.obs_patch_size` 替代硬编码 `11×11`。

## 第 4 轮（图表资产）

- `plot_losses` 为 `stage_timeline` 同时导出 PDF，便于 LaTeX 矢量插图。

## 第 5 轮（论文章节）

- `sec_experiments.tex` 由占位改为完整英文叙述：仿真设置、课程学习参数、快速验证协议、与 JSON 对齐的数值表、双图（`fig:train_curves`、`fig:stage_timeline`）。

## 第 6 轮（诚实表述）

- 正文明确 $90$ 次更新为 **sanity check**，摧毁率等指标未饱和时说明需延长训练与 LLM 阶段，避免过度声称。

## 第 7 轮（主文件）

- `main.tex` 增加 `\graphicspath{{figures/}}`；摘要补一句与快速 CPU 验证一致的陈述。

## 第 8 轮（模板修复）

- `\institute` / `\email` 括号错误（`Too many }'s`）修正：将 `\email` 移入 `\institute` 内，保证 `latexmk` 可稳定通过。

## 第 9 轮（数据侧车）

- `quick_experiment_log.json` 保存训练曲线原始序列，支持仅重绘图表而无需重训。

## 第 10 轮（后续工作节）

- `sec_experiments` 中指令可控性、消融、可扩展性保留为「计划实验」短段，与全文叙事闭合。
