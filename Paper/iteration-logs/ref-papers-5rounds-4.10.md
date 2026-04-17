# 对照 0158_final / 0166_final 五轮结构与格式迭代（2026-04-10）

参考 PDF 中体现的 ICGNC/Springer proceedings 常见写法（作者单位上标、Abstract/Keywords、
“The rest of this paper is organized as follows”、Problem Statement、Simulation、
Fig./Algorithm 说明句式、(a)(b)(c) 贡献编号等）对本稿做对齐。

## Round 1
- `main.tex`：作者统一 `\inst{1}`；单位改为学院全称 + 北京邮编；修正邮箱为逐个地址（避免 `\{\}` 形式的编组邮箱在非 tt 字体下的问题）；摘要句间空格与参考稿一致。

## Round 2
- `sec_introduction.tex`：贡献改为 (a)(b)(c) 枚举（对齐 0166）；组织结构段改为 “The rest of this paper is organized as follows.” + “In Sect.~...” 串行叙述（对齐 0158）。

## Round 3
- `sec_problem.tex`：章节标题 **Problem Formulation → Problem Statement**（对齐 0158 第 2 节命名）。
- `sec_experiments.tex`：总标题 **Experiments and Simulation Results**；子节 **Simulation Setup**、**Simulation Results**（对齐 0158 “Simulation Result” 类命名）。

## Round 4
- `sec_method.tex`：首段改为 “As shown in Fig.~\ref{...}”；图题改为 “The … framework.” + Stage I/II 简要分句（接近 proceedings 图注风格）；算法标题改为句号结尾的说明句。

## Round 5
- `sec_related.tex`：增加段首总起句，使 Related Work 结构更完整（引言链 + 分主题段落）。
- `sec_conclusion.tex`：首句改为 “This paper presented …”（常见收束句式）。

## 编译产物
- 使用 `latexmk` 与 `-jobname=main_4.10` 生成 `main_4.10.pdf`（同目录）。
