# ARIS 代码迭代优化日志（Program，5 轮）

> 日期: 2026-04-10  
> 框架: 7 维度中优先 **Method Specificity**、**Feasibility**、**Validation Focus**

## 第 1 轮

- **Diagnose（薄弱维度）**: Method Specificity — 观测空间 spatial 尺寸在环境、Buffer、Trainer 三处隐含假设为 11×11，与配置 `obs_radius` 不一致时不可实现且难查。
- **Innovate**: 以环境为单一真源暴露 `obs_patch_size`。
- **Refine / Implement**: `RolloutBuffer(..., obs_patch_size)`；`MAPPOTrainer` reshape 使用该字段。

## 第 2 轮

- **Diagnose**: Feasibility — CI/本地常从非 `Program` 目录调用 `python train/smoke_test.py`，相对配置路径失败。
- **Innovate**: 脚本级 `_PROG_ROOT` + 绝对配置路径 + `chdir`（与 `train_madrl.main` 一致策略）。
- **Implement**: 更新 `smoke_test.py`。

## 第 3 轮

- **Diagnose**: Problem Fidelity — `UAVSearchAttackEnv` 对外导出 gym API 但未实现，易误导为可用环境。
- **Innovate**: 快速失败 + 文档化「当前训练路径为 `BattlefieldEnv`」。
- **Implement**: `NotImplementedError` 桩实现。

## 第 4 轮

- **Diagnose**: Contribution Quality（工程侧）— 重复计算威胁局部坐标网格，`_get_obs` 在 rollout 内高频调用。
- **Innovate**: 构造期缓存 `mgrid` 偏移。
- **Implement**: `BattlefieldEnv.__init__` 预计算 `_threat_local_offsets`。

## 第 5 轮

- **Diagnose**: Validation Focus — 需可重复的极小算力回归验证。
- **Innovate**: 保留并强化 `smoke_test.py` 为单一命令回归；与本轮代码契约变更一并运行。
- **Evaluate**: 本地执行 `python train/smoke_test.py`（工作目录任意）应输出 `SMOKE TEST PASSED`。

## 评分对比（定性）

| 轮次 | Method Spec. | Feasibility | 备注 |
|------|----------------|-------------|------|
| 前 | 中 | 中 | 硬编码 11、cwd 敏感 |
| 后 | 高 | 高 | 配置驱动 + 根目录锚定 |
