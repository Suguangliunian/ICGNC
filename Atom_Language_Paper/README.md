# ICGNC 2026 — Atom Language Paper

## 论文信息
- 标题: Atom Language: A Rule-Based Atomic Token Representation for Converting STEP B-Rep Models to Structured CAD Sequences
- 作者: Xinran Guo, Haibin Duan (Beihang University)
- 会议: ICGNC 2026 (Springer LNCS Proceedings)

## 文件结构
```
Atom_Language_Paper/
├── atom_language_paper.tex    # 主论文 LaTeX 源文件（定稿）
├── author.tex                 # Springer 模板参考
├── styles/                    # 编译所需样式文件
│   ├── svproc.cls             # Springer Proceedings 文档类
│   ├── aliascnt.sty
│   ├── remreset.sty
│   └── bibtex/                # BibTeX 样式
├── figures/                   # 图片（待补充 pipeline 流程图）
├── review_logs/               # 10 轮 ARIS+OMNI 迭代审查日志
│   ├── round_1.md ~ round_10.md
├── authinst.pdf               # 作者指南
├── authsamp.pdf               # 作者示例
├── quickstart.pdf             # 快速入门
├── refguide.pdf               # 参考指南
└── readme.txt                 # Springer 原始说明
```

## 编译方法
```bash
cd Atom_Language_Paper
pdflatex atom_language_paper.tex
pdflatex atom_language_paper.tex   # 运行两次以解析交叉引用
```

## 迭代记录
经过 10 轮 ARIS（审查）+ OMNI（写作）迭代优化：
- R1-R4: 结构搭建 + 内容补全（比较表、消融实验、算法伪代码）
- R5-R7: 写作质量提升（表格压缩、段落过渡、术语统一）
- R8-R10: 数据验证 + 最终定稿（所有数值与项目实际数据交叉验证）

最终评分: 9/10，建议提交前补充 pipeline 流程图。
