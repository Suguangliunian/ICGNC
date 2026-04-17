#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
论文配套快速实验（本地、低算力）
--------------------------------
1) 随机策略基线：若干回合统计覆盖率等指标
2) TAGA-MAPPO 短训 + 训练曲线图（复用 train_and_plot）
3) 保存检查点并用 MAPPOEvaluator 做少量贪心评估

输出：
  Paper/figures/paper_exp/training_losses.{pdf,png}
  Paper/figures/paper_exp/stage_timeline.png
  Paper/data/quick_experiment_results.json
  Paper/data/quick_mappo.pt
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time

_PROG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PAPER = os.path.abspath(os.path.join(_PROG, "..", "Paper"))
sys.path.insert(0, _PROG)
os.chdir(_PROG)

import numpy as np
import torch
import yaml

from eval.evaluate import MAPPOEvaluator, MetricsCalculator
from train.train_and_plot import plot_losses, train_with_logging
from train.train_madrl import BattlefieldEnv


def _summarize_metrics(rows: list[dict]) -> dict:
    out = {}
    for k in rows[0]:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        out[k] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        }
    return out


def run_random_baseline(cfg: dict, n_episodes: int = 24, seed: int = 0, step_cap: int = 180) -> dict:
    """在课程最难阶段规模上运行随机策略。"""
    rng = np.random.default_rng(seed)
    st = cfg["curriculum"]["stages"][2]
    env_cfg = {
        **cfg["env"],
        "n_uavs": st["n_uavs"],
        "n_targets": st["n_targets"],
        "n_threats": st["n_threats"],
        "grid_size": st["grid_size"],
    }
    env = BattlefieldEnv(env_cfg, cfg["reward"], n_envs=1)
    rows = []
    for _ in range(n_episodes):
        env.reset()
        traj = {"total_attacks": 0, "cooperative_attacks": 0}
        for _ in range(min(env.max_steps, step_cap)):
            n = env.n_uavs
            mode_act = rng.integers(0, 2, (1, n), dtype=np.int32)
            move_act = rng.integers(0, 10, (1, n), dtype=np.int32)
            _, _, dones, _ = env.step(mode_act, move_act)
            if dones[0]:
                break
        rows.append(MetricsCalculator.compute(env, traj))
    return {"n_episodes": n_episodes, "summary": _summarize_metrics(rows)}


def main():
    fig_dir = os.path.join(_PAPER, "figures", "paper_exp")
    data_dir = os.path.join(_PAPER, "data")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    cfg_path = os.path.join(_PROG, "configs", "default.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg = copy.deepcopy(cfg)
    cfg["mappo"]["lr"] = 3e-4
    cfg["mappo"]["entropy_coef"] = 0.05

    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(cfg.get("seed", 42))
    torch.manual_seed(cfg.get("seed", 42))

    t0 = time.time()
    print("[1/4] 随机基线...", flush=True)
    random_stats = run_random_baseline(cfg, n_episodes=24, seed=1)

    n_updates = 90
    print(f"[2/4] TAGA-MAPPO 短训 (updates={n_updates})...", flush=True)
    log, trainer = train_with_logging(cfg, device, max_updates=n_updates, return_trainer=True)

    print("[3/4] 绘图...", flush=True)
    plot_losses(log, save_dir=fig_dir)

    ckpt_path = os.path.join(data_dir, "quick_mappo.pt")
    trainer.save_checkpoint(ckpt_path)

    print("[4/4] 贪心评估 (10 episodes)...", flush=True)
    evaluator = MAPPOEvaluator(cfg, ckpt_path, device=device)
    ecfg = trainer.scheduler.make_env_config(cfg["env"])
    eval_env = BattlefieldEnv(ecfg, cfg["reward"], n_envs=1)
    greedy_rows = []
    for _ in range(10):
        m, _ = evaluator.evaluate_episode(eval_env)
        greedy_rows.append(m)
    greedy_summary = _summarize_metrics(greedy_rows)

    # 训练末段窗口（与日志一致）
    tail = log["coverage"][-15:] if len(log["coverage"]) >= 15 else log["coverage"]
    train_cov_tail_mean = float(np.mean(tail))

    results = {
        "generated_at_s": time.time() - t0,
        "device": device,
        "n_updates": n_updates,
        "random_baseline": random_stats,
        "mappo_greedy_eval": {"n_episodes": 10, "summary": greedy_summary},
        "training_coverage_tail_mean": train_cov_tail_mean,
        "figures": {
            "training_curves_pdf": os.path.relpath(
                os.path.join(fig_dir, "training_losses.pdf"), _PAPER
            ),
            "training_curves_png": os.path.relpath(
                os.path.join(fig_dir, "training_losses.png"), _PAPER
            ),
        },
    }
    json_path = os.path.join(data_dir, "quick_experiment_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log_dump = {
        "update": log["update"],
        "policy_loss": log["policy_loss"],
        "value_loss": log["value_loss"],
        "entropy": log["entropy"],
        "reward": log["reward"],
        "coverage": log["coverage"],
        "stage": log["stage"],
        "stage_transitions": log["stage_transitions"],
    }
    with open(os.path.join(data_dir, "quick_experiment_log.json"), "w", encoding="utf-8") as f:
        json.dump(log_dump, f, indent=2)

    print(f"完成，耗时 {time.time()-t0:.1f}s")
    print(f"  JSON: {json_path}")
    print(f"  权重: {ckpt_path}")


if __name__ == "__main__":
    main()
