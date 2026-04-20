# Atom Language Paper 实验执行 Pipeline

> 服务器配置：8× NVIDIA A100-SXM4-80GB (CUDA 13.0)
> 当前可用：GPU 0 (59GB free), GPU 2-7 (81GB free), GPU 1 (13GB free, 占用中)
> 更新日期：2026-04-20

---

## 一、实验资源分配总览

| 实验 | 计算类型 | GPU 需求 | 推荐分配 | 预估时间 |
|------|----------|----------|----------|----------|
| Exp1: 单特征覆盖率 | CPU-only (OCC) | 0 卡 | CPU 并行 | 5 min |
| Exp2: 复杂模型分解 | CPU-only (OCC) | 0 卡 | CPU 并行 | 5 min |
| Exp3: OCC 闭环验证 | CPU-only (OCC + 点云) | 0 卡 | CPU 并行 | 10 min |
| Exp4: 运行时分析 | CPU-only | 0 卡 | 单线程（避免干扰计时） | 2 min |
| Exp5: Diffusion 训练 (DSE) | GPU (单卡) | 1 卡/配置 | 6 卡并行跑 6 配置 | 30 min |
| Exp6: Diffusion 评估 | GPU (单卡) | 1 卡 | 单卡 | 5 min |
| Exp7: Diffusion 推理 | GPU (单卡) | 1 卡 | 单卡 | 5 min |
| Exp8: 大规模泛化验证 | CPU + GPU | 1~2 卡 | 可选 | 1~2h |

---

## 二、单卡 vs 多卡分析

### 适合 CPU（无需 GPU）的实验

| 实验 | 原因 |
|------|------|
| Exp1-4 (Atom Language Pipeline) | 纯规则引擎 + PythonOCC 几何运算，无神经网络，总耗时 < 0.4s |
| OCC 闭环验证 | BRep 布尔运算 + 点云采样，CPU 密集型但单模型 < 1s |
| 点云 CD/HD 计算 | NumPy 矩阵运算，8192 点 < 0.5s |

### 适合单卡的实验

| 实验 | 原因 | 显存需求 |
|------|------|----------|
| Diffusion 单配置训练 | 模型 67.7M params, bs=16, res=32, 显存 ~4.3GB | ~5 GB |
| Diffusion 评估 | 推理模式，无梯度，显存 ~2GB | ~2 GB |
| Diffusion 推理 | 单样本逐步预测，显存 < 1GB | ~1 GB |

### 适合多卡并行的实验

| 实验 | 并行策略 | 卡数 | 原因 |
|------|----------|------|------|
| DSE 超参搜索 | 数据并行（每卡跑不同配置） | 6 卡 | 12 个配置，每个独立，单卡 5GB，互不干扰 |
| 大规模泛化验证 | 数据并行（每卡跑不同模型子集） | 2~4 卡 | 100+ 模型的 inference 可分片 |

**注意：** 当前 Diffusion 模型较小（67.7M），单卡 A100 80GB 绰绰有余，**不需要模型并行（DDP/FSDP）**。多卡的价值在于同时跑多个独立实验配置。

---

## 三、实验执行 Pipeline（按顺序）

### Phase 1: Atom Language 实验（CPU-only，~20 min）

无需 GPU，可在任意终端直接运行。

```bash
# 环境激活
source /home/dataset-local/batchcom/conda-env/pointer-cad/bin/activate
cd /home/dataset-assist-0/gxr/Atom-to-CAD/Atom-to-CAD/Atom_Language

# Step 1: Exp1 + Exp2（管线覆盖率 + 复杂分解）
python experiments/run_experiments.py --experiment 1
# 输出: experiments/local_Assemble/exp1_results.json

python experiments/run_experiments.py --experiment 2  
# 输出: experiments/local_Assemble/exp2_results.json

# Step 2: Exp3（OCC 闭环验证 + CD 计算）
python experiments/run_experiments.py --experiment 2
# 输出: exp2_*_rebuilt.step + exp2_results.json (含 CD/HD)

# Step 3: Exp4（运行时分析 — 单线程避免干扰）
python -c "
import json
with open('experiments/local_Assemble/full_report.json') as f:
    d = json.load(f)
for exp in d['experiments']:
    timings = [r['timings'] for r in exp['results'] if 'timings' in r]
    if timings:
        avg = {k: sum(t.get(k,0) for t in timings)/len(timings)*1000 
               for k in ['face_graph','partition','features','worksteps','atoms']}
        print(f'{exp[\"experiment\"]}: {avg}')
"
```

**预估时间：5~10 分钟**（已有结果可直接用，重跑也很快）

---

### Phase 2: Diffusion DSE 超参搜索（6 卡并行，~30 min）

每个配置独立占用 1 张 GPU (~5GB)，6 张卡同时跑 6 个配置。

```bash
source /home/dataset-local/batchcom/conda-env/pointer-cad/bin/activate
cd /home/dataset-assist-0/gxr/Atom-to-CAD/Atom-to-CAD/Diffusion_Feature/scheme1_code

# 并行启动 6 个训练任务（GPU 2-7）
for i in $(seq 0 5); do
    GPU_ID=$((i + 2))  # 使用 GPU 2-7
    CONFIG_IDX=$((i))
    
    # 从 12 个配置中取参数
    case $CONFIG_IDX in
        0) ARGS="--base-ch 32 --lr 1e-4 --dropout 0.1" ;;
        1) ARGS="--base-ch 32 --lr 1e-4 --dropout 0.2" ;;
        2) ARGS="--base-ch 32 --lr 3e-4 --dropout 0.1" ;;
        3) ARGS="--base-ch 48 --lr 1e-4 --dropout 0.1" ;;
        4) ARGS="--base-ch 48 --lr 1e-4 --dropout 0.2" ;;
        5) ARGS="--base-ch 48 --lr 3e-4 --dropout 0.1" ;;
    esac
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
        $ARGS \
        --res 32 --train-samples 1024 --val-samples 128 \
        --epochs 20 --batch-size 16 --patience 6 \
        --label-smoothing 0.1 --no-alignment \
        --save-dir dse_results/config_${CONFIG_IDX} \
        > dse_results/config_${CONFIG_IDX}/train.log 2>&1 &
    
    echo "Started config_${CONFIG_IDX} on GPU ${GPU_ID} (PID: $!)"
done

echo "All 6 configs launched. Monitor with: tail -f dse_results/config_*/train.log"
wait
echo "All training complete."
```

**预估时间：**
- 单配置：1024 samples × 20 epochs ÷ 16 batch = 1280 iterations，A100 上 ~5 min
- 6 配置并行：~5 min（受最慢配置限制）
- 剩余 6 配置第二轮：再 ~5 min
- **总计：~10 min**

---

### Phase 3: Diffusion 评估（单卡，~5 min）

```bash
# 选取 DSE 最佳配置评估
BEST_CKPT="dse_results/config_A/best_model.pt"  # 根据 DSE 结果选择

CUDA_VISIBLE_DEVICES=2 python evaluate.py \
    --checkpoint $BEST_CKPT \
    --base-ch 48 --res 32 \
    --test-samples 256 \
    --output evaluation_results.json

# 输出: evaluation_results.json (per-class accuracy, confusion matrix, param errors)
```

---

### Phase 4: Diffusion 推理演示（单卡，~5 min）

```bash
CUDA_VISIBLE_DEVICES=2 python inference.py \
    --checkpoint $BEST_CKPT \
    --base-ch 48 --res 32 \
    --max-steps 8 \
    --num-samples 10 \
    --output inference_results.json

# 输出: inference_results.json (predicted operation sequences)
```

---

### Phase 5: 可视化 + 图表生成（CPU，~10 min）

```bash
# 重建 STEP 可视化
python visualize.py \
    --results inference_results.json \
    --output-dir figures/

# 汇总所有实验结果为论文表格
python -c "
import json
# ... 汇总脚本（生成 LaTeX 表格）
"
```

---

## 四、并行执行时间线

```
时间轴 (分钟)
0         5         10        15        20        25        30
|---------|---------|---------|---------|---------|---------|
[=== Phase 1: Atom Language (CPU) ===]
          [======= Phase 2: DSE Round 1 (GPU 2-7) =======]
                    [=== Phase 2: DSE Round 2 (GPU 2-7) ==]
                              [= Phase 3: Eval (GPU 2) =]
                                        [= Phase 4: Infer =]
                                                  [= Phase 5: Viz =]
```

**Phase 1 和 Phase 2 可以同时启动**（CPU 和 GPU 互不干扰），总时间可压缩到 ~20 min。

---

## 五、GPU 分配方案

```
GPU 0: 空闲（59GB free）— 备用 / 大模型实验
GPU 1: 占用中（13GB free）— 不动
GPU 2: DSE config_0 → Eval → Inference
GPU 3: DSE config_1
GPU 4: DSE config_2
GPU 5: DSE config_3
GPU 6: DSE config_4
GPU 7: DSE config_5
```

---

## 六、一键执行脚本

```bash
#!/bin/bash
# run_all_experiments.sh — 全部实验一键执行
set -e

CONDA_ENV="/home/dataset-local/batchcom/conda-env/pointer-cad/bin/activate"
PROJECT="/home/dataset-assist-0/gxr/Atom-to-CAD/Atom-to-CAD"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT}/experiment_logs/${TIMESTAMP}"
mkdir -p $LOG_DIR

source $CONDA_ENV

echo "========== Phase 1: Atom Language Pipeline (CPU) =========="
cd ${PROJECT}/Atom_Language
python experiments/run_experiments.py --experiment all > ${LOG_DIR}/atom_pipeline.log 2>&1 &
PID_ATOM=$!

echo "========== Phase 2: Diffusion DSE (GPU 2-7) =========="
cd ${PROJECT}/Diffusion_Feature/scheme1_code

CONFIGS=("--base-ch 32 --lr 1e-4 --dropout 0.1"
         "--base-ch 32 --lr 1e-4 --dropout 0.2"
         "--base-ch 32 --lr 3e-4 --dropout 0.1"
         "--base-ch 48 --lr 1e-4 --dropout 0.1"
         "--base-ch 48 --lr 1e-4 --dropout 0.2"
         "--base-ch 48 --lr 3e-4 --dropout 0.1")

for i in $(seq 0 5); do
    GPU_ID=$((i + 2))
    SAVE_DIR="dse_results/config_${i}"
    mkdir -p $SAVE_DIR
    CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
        ${CONFIGS[$i]} \
        --res 32 --train-samples 1024 --val-samples 128 \
        --epochs 20 --batch-size 16 --patience 6 \
        --label-smoothing 0.1 --no-alignment \
        --save-dir $SAVE_DIR \
        > ${LOG_DIR}/dse_config_${i}.log 2>&1 &
done

# 等待 Phase 1 完成
wait $PID_ATOM
echo "Phase 1 complete."

# 等待所有 DSE 完成
wait
echo "Phase 2 complete."

echo "========== Phase 3: Evaluation =========="
# 找最佳配置
BEST=$(python -c "
import json, glob
best_loss = float('inf')
best_dir = ''
for d in glob.glob('dse_results/config_*'):
    try:
        ckpt = torch.load(f'{d}/best_model.pt', map_location='cpu', weights_only=False)
        if ckpt.get('val_loss', float('inf')) < best_loss:
            best_loss = ckpt['val_loss']
            best_dir = d
    except: pass
print(best_dir)
")

CUDA_VISIBLE_DEVICES=2 python evaluate.py \
    --checkpoint ${BEST}/best_model.pt \
    --base-ch 48 --res 32 --test-samples 256 \
    --output ${LOG_DIR}/evaluation_results.json

echo "========== Phase 4: Inference =========="
CUDA_VISIBLE_DEVICES=2 python inference.py \
    --checkpoint ${BEST}/best_model.pt \
    --base-ch 48 --res 32 --max-steps 8 --num-samples 10 \
    --output ${LOG_DIR}/inference_results.json

echo "========== All Experiments Complete =========="
echo "Results in: ${LOG_DIR}/"
ls -la ${LOG_DIR}/
```

---

## 七、时间预估总结

| 阶段 | 并行方式 | 时间 | 备注 |
|------|----------|------|------|
| Phase 1: Atom Pipeline | CPU (与 Phase 2 并行) | 5 min | 已有结果可跳过 |
| Phase 2: DSE 超参搜索 | 6 卡并行 | 10 min | 12 配置分 2 轮 |
| Phase 3: 评估 | 单卡 | 5 min | |
| Phase 4: 推理 | 单卡 | 5 min | |
| Phase 5: 可视化 | CPU | 10 min | |
| **总计（串行）** | | **35 min** | |
| **总计（最大并行）** | | **~20 min** | Phase 1+2 并行 |

如果只需要复现论文已有数据（不重新训练），仅需 Phase 1 + Phase 3-4，**总计 ~15 min**。
