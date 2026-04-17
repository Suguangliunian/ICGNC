# OmniSkill 迭代优化日志 (5轮)

> 日期: 2026-04-09

## 第1轮：全面质量审查

1. Abstract 句间距统一为双空格
2. Figure caption 扩展为完整描述（含 Stage I/II 说明）
3. Experiments placeholder 中指令格式统一为 `\textsc`
4. Algorithm 伪代码大幅增强：添加 `\ENSURE`、curriculum 逻辑、LoRA 参数、公式引用

## 第2轮：写作流畅度 + 论证逻辑

1. Related Work "Knowledge distillation" 段落引用修正
2. Introduction "key insight" 表述增强——明确 complementary gap
3. Method 开头段落改为结构化导航（含 Section 引用）

## 第3轮：深度优化

1. Distillation "Data" → "Data collection"，扩展为完整数据收集流程描述
2. Prompt encoding 段落重写——明确四个 block 结构 (a)-(d)
3. 添加 TAGA-derived signal quality 与 $f_d$, $f_t$ 的映射说明

## 第4轮：Conclusion 增强

1. 新增 Limitations 段落（同质 UAV 假设、数据集覆盖、prompt 截断、模板 CoT）
2. Future work 独立为段落，扩展三个方向
3. 全文 `\textsc` 一致性验证通过
4. 全文 `\mid` vs `|` 一致性验证通过

## 第5轮：最终打磨

1. 源代码行宽优化（避免 overfull hbox）
2. 全文最终交叉检查
3. 公式编号连续性验证：eq:taga → eq:taga_agg → eq:ppo → eq:sft → eq:reward

## 最终文件状态


| 文件                     | 行数   | 状态                     |
| ---------------------- | ---- | ---------------------- |
| `main.tex`             | ~88  | ✅ 完整                   |
| `sec_introduction.tex` | ~53  | ✅ 完整                   |
| `sec_related.tex`      | ~45  | ✅ 完整（含知识蒸馏段落）          |
| `sec_problem.tex`      | ~43  | ✅ 完整（含 reward 公式）      |
| `sec_method.tex`       | ~195 | ✅ 完整（4个公式 + Algorithm） |
| `sec_experiments.tex`  | ~43  | ⏳ Placeholder          |
| `sec_conclusion.tex`   | ~30  | ✅ 完整（含 Limitations）    |
| `references.bib`       | ~195 | ✅ 24 条引用               |
