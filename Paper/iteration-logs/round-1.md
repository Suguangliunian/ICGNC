# Round 1 — 结构完整性 + 技术准确性

> 日期: 2026-04-08

## 审阅发现

### 技术准确性问题（已修正）

1. **TAGA 公式与代码不一致**: 论文原写加法形式 `e_ij + λ_d f_d + λ_m f_m + λ_t f_t`，但代码实现为乘法形式 `e_ij * f_d * f_m * f_t`。已修正为乘法形式。
2. **注意力机制描述错误**: 论文原写标准 GAT（concat + LeakyReLU），但代码实现为 scaled dot-product attention（QKV 投影）。已修正为 QKV 形式。
3. **CNN branch 描述错误**: 论文原写 MaxPool，代码用 AdaptiveAvgPool2d(1)。已修正。
4. **MLP branch 输入描述错误**: 论文原写 "position, mode, ammunition, timestep"，代码中实际为 "normalized position, velocity, energy, mode"。已修正。
5. **Policy head 隐藏维度错误**: 论文原写 Linear(192, 64)，代码中为 Linear(192, 128)。已修正。
6. **TAGA 架构细节不准确**: 论文原写 "ELU activation and dropout 0.1"，代码中无 ELU 和 dropout，使用 LayerNorm + 残差连接。已修正。
7. **距离衰减因子公式**: 论文原写 `-d/R_comm`（线性），代码实现为 `exp(-λ_d * d / R_comm)`（指数）。已修正为指数形式。
8. **任务感知因子公式**: 论文原写 `1(m_i = m_j)`（0/1），代码实现为 `1 + λ_m * 1(m_i = m_j)`（1/1+λ_m）。已修正。

### 结构完整性检查


| 章节                  | 状态  | 备注                    |
| ------------------- | --- | --------------------- |
| Abstract            | ✅   | 70-150词范围内，含 keywords |
| Introduction        | ✅   | 背景+动机+贡献完整            |
| Related Work        | ✅   | 三个子节覆盖 MARL/LLM/UAV   |
| Problem Formulation | ✅   | Dec-POMDP 定义完整        |
| Method 4.1-4.4      | ✅   | 框架+TAGA+编码器+策略        |
| Method 4.5          | ✅   | MAPPO+课程学习            |
| Method 4.6          | ✅   | LLM蒸馏完整               |
| Experiments         | ✅   | Placeholder           |
| Conclusion          | ✅   | 总结+Future Work        |
| References          | ✅   | 22篇引用                 |


### 新增内容

- 添加了 TAGA 聚合公式 (Eq. taga_agg)，明确 V 投影和多头拼接
- 添加了残差连接和 LayerNorm 的数学描述
- 补充了 λ_d, λ_m, λ_t 的初始化值

## 修改文件

- `sec_method.tex`: 8处技术修正