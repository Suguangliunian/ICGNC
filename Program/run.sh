#!/bin/bash
# ============================================================
# TAGA 完整 Pipeline 一键运行脚本
# ============================================================
# 用法：
#   bash run.sh              # 运行完整 pipeline
#   bash run.sh --stage A    # 只运行阶段A（MAPPO训练）
#   bash run.sh --stage B    # 只运行阶段B（数据生成 + LoRA微调）
#   bash run.sh --stage eval # 只运行评估
# ============================================================

set -e  # 遇到错误立即退出

# --- 默认参数 ---
CONFIG="configs/default.yaml"
DEVICE="auto"
SEED=42
STAGE="all"
MAPPO_UPDATES=5000
MAPPO_CKPT="checkpoints/final.pt"
QWEN_CKPT="checkpoints/qwen_lora/final"
DATA_DIR="data/generated"
EVAL_EPISODES=20

# --- 解析命令行参数 ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"; shift 2 ;;
        --device)
            DEVICE="$2"; shift 2 ;;
        --seed)
            SEED="$2"; shift 2 ;;
        --stage)
            STAGE="$2"; shift 2 ;;
        --mappo_updates)
            MAPPO_UPDATES="$2"; shift 2 ;;
        --eval_episodes)
            EVAL_EPISODES="$2"; shift 2 ;;
        --help|-h)
            echo "用法: bash run.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --config PATH        配置文件路径 (默认: configs/default.yaml)"
            echo "  --device DEVICE       设备 cpu/cuda/auto (默认: auto)"
            echo "  --seed SEED           随机种子 (默认: 42)"
            echo "  --stage STAGE         运行阶段 all/A/B/eval (默认: all)"
            echo "  --mappo_updates N     MAPPO 训练更新次数 (默认: 5000)"
            echo "  --eval_episodes N     评估回合数 (默认: 20)"
            exit 0 ;;
        *)
            echo "未知参数: $1"; exit 1 ;;
    esac
done

# --- 创建必要目录 ---
mkdir -p checkpoints logs data/generated eval/results

echo "============================================================"
echo "TAGA Pipeline"
echo "============================================================"
echo "  配置文件: $CONFIG"
echo "  设备: $DEVICE"
echo "  随机种子: $SEED"
echo "  运行阶段: $STAGE"
echo "============================================================"
echo ""

START_TIME=$(date +%s)

# ============================================================
# 阶段 A: 训练 TAGA-MAPPO
# ============================================================
if [[ "$STAGE" == "all" || "$STAGE" == "A" ]]; then
    echo "============================================================"
    echo "[阶段 A] 训练 TAGA-MAPPO"
    echo "  预计耗时: ~1.5 小时"
    echo "============================================================"

    python train/train_madrl.py \
        --config "$CONFIG" \
        --device "$DEVICE" \
        --seed "$SEED" \
        --total_updates "$MAPPO_UPDATES"

    echo ""
    echo "[阶段 A] 训练完成！"
    echo ""
fi

# ============================================================
# 阶段 B-1: 生成专家数据
# ============================================================
if [[ "$STAGE" == "all" || "$STAGE" == "B" ]]; then
    echo "============================================================"
    echo "[阶段 B-1] 生成专家数据 + 反事实数据"
    echo "  目标: ~45K 条数据"
    echo "============================================================"

    python data/generate_data.py \
        --config "$CONFIG" \
        --checkpoint "$MAPPO_CKPT" \
        --device "$DEVICE" \
        --seed "$SEED"

    echo ""
    echo "[阶段 B-1] 数据生成完成！"
    echo ""

    # ============================================================
    # 阶段 B-2: LoRA 微调 Qwen
    # ============================================================
    echo "============================================================"
    echo "[阶段 B-2] LoRA 微调 Qwen"
    echo "  预计耗时: ~37 分钟"
    echo "============================================================"

    python train/train_qwen.py \
        --config "$CONFIG" \
        --data_path "${DATA_DIR}/train.json" \
        --device "$DEVICE" \
        --seed "$SEED"

    echo ""
    echo "[阶段 B-2] 微调完成！"
    echo ""
fi

# ============================================================
# 阶段 C: 评估
# ============================================================
if [[ "$STAGE" == "all" || "$STAGE" == "eval" ]]; then
    echo "============================================================"
    echo "[阶段 C] 评估"
    echo "  评估回合数: $EVAL_EPISODES"
    echo "============================================================"

    # 评估 TAGA-MAPPO
    echo ""
    echo "--- 评估 TAGA-MAPPO ---"
    python eval/evaluate.py \
        --config "$CONFIG" \
        --method mappo \
        --checkpoint "$MAPPO_CKPT" \
        --n_episodes "$EVAL_EPISODES" \
        --device "$DEVICE" \
        --seed "$SEED" \
        --visualize

    # 评估 TAGA-LLM
    echo ""
    echo "--- 评估 TAGA-LLM ---"
    python eval/evaluate.py \
        --config "$CONFIG" \
        --method llm \
        --checkpoint "$QWEN_CKPT" \
        --n_episodes "$EVAL_EPISODES" \
        --device "$DEVICE" \
        --seed "$SEED" \
        --visualize

    # 对比评估
    echo ""
    echo "--- 对比评估 ---"
    python eval/evaluate.py \
        --config "$CONFIG" \
        --compare "$MAPPO_CKPT" "$QWEN_CKPT" \
        --n_episodes "$EVAL_EPISODES" \
        --device "$DEVICE" \
        --seed "$SEED" \
        --visualize

    echo ""
    echo "[阶段 C] 评估完成！"
    echo ""
fi

# ============================================================
# 完成
# ============================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo "============================================================"
echo "Pipeline 完成！"
echo "  总耗时: ${MINUTES}分${SECONDS}秒"
echo "  检查点: checkpoints/"
echo "  训练数据: data/generated/"
echo "  评估结果: eval/results/"
echo "============================================================"
