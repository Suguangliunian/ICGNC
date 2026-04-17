# OmniSkill 代码迭代优化日志（Program，5 轮）

> 日期: 2026-04-10  
> 范围: `D:\北航工作日常\ICGNC\Program` Python 源码

## 第 1 轮：结构与可维护性

1. 训练入口 `train_madrl.main()` 在解析配置前锚定项目根目录并 `chdir`，保证相对路径 `checkpoints/`、`configs/` 与从任意 cwd 启动行为一致。
2. `smoke_test.py` 使用 `Program` 根目录拼接 `configs/default.yaml`，避免「未在 Program 目录下运行即失败」。

## 第 2 轮：接口与契约一致性

1. `RolloutBuffer.get_batches` 与 `MAPPOTrainer._get_features` 中原硬编码 `(5, 11, 11)` 与 `BattlefieldEnv.obs_radius` 解耦失败时会导致静默 shape 错误；统一为 `env.obs_patch_size = 2 * obs_radius + 1` 驱动。

## 第 3 轮：文档与注释对齐实现

1. `CurriculumScheduler` 文档串中网格尺寸与 `configs/default.yaml` 三阶段（30 / 50 / 100）对齐，减少读者误解。

## 第 4 轮：性能与热点

1. `_get_obs` 中威胁通道用到的 `local_offsets` 网格仅依赖 `obs_radius`，在 `__init__` 中预计算并缓存，避免每步重复分配 `mgrid`。

## 第 5 轮：失败模式与可调试性

1. `env/uav_env.py` 中占位 `reset`/`step` 等改为显式 `NotImplementedError` 与说明，避免误用 `...` 导致难以理解的 `TypeError`。

## 落地改动文件

| 文件 | 变更摘要 |
|------|-----------|
| `train/train_madrl.py` | `obs_patch_size`、RolloutBuffer 维度、`main` 根目录 |
| `train/smoke_test.py` | 根路径、RolloutBuffer 参数 |
| `train/sweep_and_validate.py` | RolloutBuffer `obs_patch_size` |
| `train/train_and_plot.py` | 同上 |
| `env/uav_env.py` | 占位方法显式未实现 |
