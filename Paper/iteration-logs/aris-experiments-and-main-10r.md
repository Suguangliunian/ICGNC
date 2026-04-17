# ARIS 迭代日志：实验复现 + `main.tex`（10 轮）

> 日期: 2026-04-10  
> 维度权重沿用 `Program/refine-logs/aris-A-setup.md` 思路，聚焦 **Validation Focus**、**Method Specificity**、**Feasibility**。

## 第 1 轮

- **Diagnose**: 实验节长期为 `\emph{To be completed}`，与摘要中「已评估」口吻脱节（Validation 薄弱）。  
- **Innovate**: 用可脚本化、可重复的数字与图替换占位。  
- **Refine**: 落地 `paper_quick_experiment.py` + `sec_experiments.tex`。

## 第 2 轮

- **Diagnose**: `MAPPOEvaluator` 与训练特征不一致（Specificity 低），贪心指标不可解释。  
- **Innovate**: 在评估路径中执行完整 TAGA 栈。  
- **Test**: 短训后贪心 rollout 数值与训练日志同向（覆盖率高于随机）。

## 第 3 轮

- **Diagnose**: `RolloutBuffer`/Trainer 已用 `obs_patch_size`，评估仍写死 11（接口漂移风险）。  
- **Refine**: `evaluate.py` reshape 与 `BattlefieldEnv` 对齐。

## 第 4 轮

- **Diagnose**: 无 `trainer` 返回时保存检查点需复制训练循环（Feasibility 差）。  
- **Refine**: `return_trainer` 分支。

## 第 5 轮

- **Diagnose**: 仅 PNG 时间线不利于印刷（Venue Readiness）。  
- **Refine**: 导出 `stage_timeline.pdf`。

## 第 6 轮

- **Diagnose**: 短训下 `target_destroy_rate=0` 若隐瞒将损害 **Problem Fidelity**。  
- **Refine**: 表中如实报告并在讨论段解释需更长 budget。

## 第 7 轮

- **Diagnose**: 图片路径散落导致编译脆弱。  
- **Refine**: `\graphicspath{{figures/}}` + 相对路径 `paper_exp/...`。

## 第 8 轮

- **Diagnose**: `main.tex` 机构邮箱处 TeX 语法错误阻断 PDF（Feasibility 归零）。  
- **Refine**: 修正 `\institute`/`\email` 嵌套。

## 第 9 轮

- **Diagnose**: 摘要未提及快速验证则读者对「Figure 从哪来」缺锚点。  
- **Refine**: 摘要增一句与 Fig.~\ref{fig:train_curves} 呼应（见 `main.tex`）。

## 第 10 轮

- **Diagnose**: 全文引用链需可编译。  
- **Evaluate**: `latexmk -pdf -f main.tex` 成功生成 `main.pdf`；剩余 overfull hbox 为版式细调项，留待后续排版轮次。
