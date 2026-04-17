# Round 2 — 逻辑连贯性 + 论证深度

> 日期: 2026-04-08

## 审阅发现与修改

### 1. Introduction 过渡论证不足

- **问题**: 从 MARL 不可解释性直接跳到 TAGA-LLM 方案，缺少"为什么用 LLM 来解决可解释性"的论证
- **修改**: 添加 key insight 段落，阐明 MARL 擅长发现策略但锁在黑箱中，LLM 擅长推理但缺乏领域知识，蒸馏是桥梁

### 2. Related Work 缺少 research gap 总结

- **问题**: 三个子节各自独立，没有明确指出现有工作的共同不足
- **修改**: 添加 Summary 段落，明确三个 gap：(i) 静态图假设 (ii) LLM 缺乏协同策略 (iii) 传统方法不适应动态环境

### 3. Dec-POMDP 定义不完整

- **问题**: tuple 中的 γ 和各符号未解释
- **修改**: 展开 tuple 各元素的含义，包括 γ = 0.99

### 4. Centralized critic 描述不精确

- **问题**: 原写"拼接所有 agent 特征"，但代码中 trainer 实际用的是单 agent 特征输入 critic（简化版）
- **修改**: 改为 mean-pooled TAGA features + 两层 MLP 描述，更准确反映实现

### 5. Stage I → Stage II 缺少过渡

- **问题**: 4.6 节直接开始蒸馏描述，没有解释为什么需要 Stage II
- **修改**: 添加过渡段落，阐明 Stage I 的局限性（不可解释、不可指令控制）和 Stage II 的动机

### 6. Conclusion 与 Introduction 重复

- **问题**: Conclusion 几乎重述了 Introduction 的内容
- **修改**: 重写 Conclusion，聚焦于技术贡献的意义、实际应用价值、以及与现有工作的区别

## 修改文件

- `sec_introduction.tex`: 添加 key insight 过渡段落
- `sec_related.tex`: 添加 research gap Summary
- `sec_problem.tex`: 展开 Dec-POMDP tuple 定义
- `sec_method.tex`: 修正 critic 描述 + 添加 Stage II 过渡
- `sec_conclusion.tex`: 重写，避免重复，增加实际意义讨论