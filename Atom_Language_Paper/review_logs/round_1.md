# Round 1 — ARIS Review

## Review Date: 2026-04-19
## Reviewer: ARIS (Internal)

---

## Major Issues

1. **Missing Figure**: The paper references Fig.1 (pipeline overview) but no figure is defined. A pipeline diagram is essential for a methods paper.

2. **Small evaluation scale**: Only 29 models (18 single-feature + 11 complex) is a very small test set. The paper should explicitly acknowledge this limitation and discuss generalizability.

3. **No comparison with baselines**: The paper lacks any comparison with existing methods (BRepNet, UV-Net, DeepCAD, etc.). Even if direct comparison is difficult due to different output formats, some form of comparative analysis is needed.

4. **OCC verification too limited**: Only 3 models verified end-to-end is insufficient. The paper should explain why only 3 were selected and provide more verification results.

5. **No ablation study**: The paper claims 12 innovations but provides no ablation study showing the contribution of each innovation to the final metrics.

## Minor Issues

1. Abstract slightly exceeds 150 words (Springer recommends 70-150).
2. The degree symbol in "θ > 180°" should use LaTeX \textdegree or ^\circ.
3. Table 2 (Exp1) confidence values for operations like Fillet/Chamfer/Rib/Shell appear to be estimated rather than from actual data.
4. Missing pipeline figure reference will cause LaTeX warning.
5. Some references (gong2025cross, su2024image, su2024enhanced) seem unrelated to the paper topic.
6. The paper lacks a formal algorithm pseudocode for the pipeline.

## Strengths

1. Well-defined token vocabulary with clear semantic categories.
2. Complete pipeline from STEP to tokens with closed-loop verification concept.
3. Good coverage of 18 SolidWorks operation types.
4. Multi-evidence fusion confidence is a solid contribution.
5. Clear writing style with good mathematical formalization.

## Specific Suggestions

### Abstract
- Tighten to 120-150 words
- Add one sentence about the significance/application of the work

### Introduction
- Add a pipeline overview figure
- Strengthen the motivation with specific industry use cases

### Related Work
- Add comparison table showing what each method can/cannot do
- Discuss graph neural network approaches for B-Rep more thoroughly

### Method
- Add Algorithm 1: Overall pipeline pseudocode
- Add a concrete atom sequence example for one workstep
- Clarify how the 6-level quantization boundaries were determined

### Experiments
- Add ablation study (at minimum: with/without dihedral angles, with/without residual face recycling, with/without multi-evidence fusion)
- Add timing/performance analysis
- Expand OCC verification to more models or explain the selection criteria
- Add comparison table with related methods

### Conclusion
- Discuss limitations more explicitly
- Connect future work to broader CAD/manufacturing community needs
