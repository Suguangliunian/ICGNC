"""
参数扫描 + 快速验证训练脚本
==============================
功能：
  1. 对关键超参数做小规模 grid search
  2. 验证 stage_1 → stage_2 → stage_3 全流程可达
  3. 输出最优参数组合和收敛报告

用法：
  python train/sweep_and_validate.py --config configs/default.yaml --mode sweep
  python train/sweep_and_validate.py --config configs/default.yaml --mode validate
"""

import os
import sys
import time
import copy
import argparse
import yaml
import numpy as np
import torch
import itertools
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train.train_madrl import MAPPOTrainer, BattlefieldEnv, CurriculumScheduler


def quick_train(cfg, device, max_updates=300, verbose=True):
    """快速训练并返回指标摘要"""
    cfg = copy.deepcopy(cfg)
    cfg["mappo"]["total_updates"] = max_updates
    cfg["mappo"]["log_interval"] = 20
    cfg["mappo"]["save_interval"] = max_updates + 1
    # 减小并行环境数和 rollout 步数以加速
    cfg["mappo"]["n_parallel_envs"] = 2
    cfg["mappo"]["rollout_steps"] = 32
    cfg["mappo"]["ppo_epochs"] = 3
    cfg["mappo"]["mini_batch_size"] = 32
    # 缩短 episode 长度
    cfg["env"]["max_steps"] = 100

    print(f"    [quick_train] 创建 Trainer...", flush=True)
    trainer = MAPPOTrainer(cfg, device=device)
    print(f"    [quick_train] Trainer 就绪, n_envs={trainer.n_envs}, n_agents={trainer.n_agents}", flush=True)

    start_time = time.time()
    obs = trainer.env.reset()
    episode_coverages = deque(maxlen=200)
    stage_transitions = []
    all_rewards = []
    all_coverages = []
    all_losses = []

    mappo_cfg = trainer.mappo_cfg
    rollout_steps = mappo_cfg["rollout_steps"]

    for update in range(1, max_updates + 1):
        trainer.total_updates = update

        from train.train_madrl import RolloutBuffer
        buffer = RolloutBuffer(
            trainer.n_envs,
            trainer.n_agents,
            rollout_steps,
            trainer.device,
            obs_patch_size=trainer.env.obs_patch_size,
        )
        ep_rewards = np.zeros(trainer.n_envs)

        for step in range(rollout_steps):
            mode_act, move_act, mode_lp, move_lp, values, feats = trainer.select_actions(obs, trainer.n_agents)
            next_obs, rewards, dones, infos = trainer.env.step(mode_act, move_act)

            flat_rewards = rewards.reshape(-1)
            norm_rewards = trainer.reward_normalizer.normalize(flat_rewards).reshape(rewards.shape)
            buffer.add(obs, mode_act, move_act, mode_lp, move_lp, norm_rewards, dones, values, feats)
            ep_rewards += rewards.mean(axis=1)

            for e in range(trainer.n_envs):
                if dones[e]:
                    trainer.episode_rewards.append(ep_rewards[e])
                    episode_coverages.append(infos[e]["coverage"])
                    ep_rewards[e] = 0
                    trainer.env.reset_single(e)

            obs = trainer.env._get_obs()

        with torch.no_grad():
            feats_last = trainer._get_features(obs, trainer.n_agents)
            last_values = trainer.critic(feats_last).squeeze(-1)
            last_values = last_values.cpu().numpy().reshape(trainer.n_envs, trainer.n_agents)
        buffer.compute_gae(last_values, mappo_cfg["gamma"], mappo_cfg["lambda_gae"])

        losses = trainer.ppo_update(buffer)

        if len(episode_coverages) > 0:
            avg_coverage = np.mean(list(episode_coverages)[-20:])
        else:
            metrics = trainer.env.get_metrics()
            avg_coverage = np.mean([m["search_coverage"] for m in metrics])

        old_stage = trainer.scheduler.current_stage
        stage_changed = trainer.scheduler.report_performance(avg_coverage)
        if stage_changed:
            stage_transitions.append({
                "from_stage": old_stage,
                "to_stage": trainer.scheduler.current_stage,
                "update": update,
                "coverage": avg_coverage,
            })
            trainer._rebuild_env()
            obs = trainer.env.reset()
            episode_coverages.clear()

        avg_reward = np.mean(trainer.episode_rewards) if trainer.episode_rewards else 0
        all_rewards.append(avg_reward)
        all_coverages.append(avg_coverage)
        all_losses.append(losses["policy_loss"])

        if verbose and update % 50 == 0:
            print(f"  [update {update:4d}] stage={trainer.scheduler.stage_name} "
                  f"reward={avg_reward:.2f} cov={avg_coverage:.3f} "
                  f"ploss={losses['policy_loss']:.4f}")

    elapsed = time.time() - start_time
    final_stage = trainer.scheduler.current_stage

    # 收敛判断：最后 50 步的 reward 趋势
    recent_rewards = all_rewards[-50:] if len(all_rewards) >= 50 else all_rewards
    reward_trend = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0] if len(recent_rewards) > 5 else 0

    return {
        "final_stage": final_stage,
        "stage_transitions": stage_transitions,
        "n_transitions": len(stage_transitions),
        "final_reward": np.mean(all_rewards[-20:]) if all_rewards else 0,
        "final_coverage": np.mean(all_coverages[-20:]) if all_coverages else 0,
        "reward_trend": reward_trend,
        "final_loss": np.mean(all_losses[-20:]) if all_losses else 0,
        "elapsed_s": elapsed,
        "converging": reward_trend > 0,
    }


def run_sweep(cfg, device):
    """对关键参数做 grid search"""
    param_grid = {
        "lr": [3e-4, 7e-4],
        "entropy_coef": [0.02, 0.05],
    }

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))

    print(f"参数扫描：{len(combos)} 种组合")
    print("=" * 70)

    results = []

    for i, combo in enumerate(combos):
        trial_cfg = copy.deepcopy(cfg)
        params = dict(zip(keys, combo))

        trial_cfg["mappo"]["lr"] = params["lr"]
        trial_cfg["mappo"]["entropy_coef"] = params["entropy_coef"]

        print(f"\n--- 试验 {i+1}/{len(combos)} ---")
        print(f"  参数: {params}")

        try:
            result = quick_train(trial_cfg, device, max_updates=100, verbose=False)
            result["params"] = params
            results.append(result)

            print(f"  最终阶段: stage_{result['final_stage']+1} "
                  f"| 晋级次数: {result['n_transitions']} "
                  f"| 奖励: {result['final_reward']:.2f} "
                  f"| 覆盖率: {result['final_coverage']:.3f} "
                  f"| 收敛: {'是' if result['converging'] else '否'} "
                  f"| 耗时: {result['elapsed_s']:.1f}s")
        except Exception as e:
            print(f"  失败: {e}")
            results.append({"params": params, "error": str(e), "n_transitions": -1})

    print("\n" + "=" * 70)
    print("参数扫描结果排名")
    print("=" * 70)

    valid = [r for r in results if "error" not in r]
    valid.sort(key=lambda r: (r["n_transitions"], r["final_reward"]), reverse=True)

    for rank, r in enumerate(valid[:5], 1):
        print(f"  #{rank}: {r['params']}")
        print(f"       晋级={r['n_transitions']} 阶段={r['final_stage']+1} "
              f"奖励={r['final_reward']:.2f} 覆盖={r['final_coverage']:.3f} "
              f"收敛={'是' if r['converging'] else '否'}")

    if valid:
        best = valid[0]
        print(f"\n最优参数: {best['params']}")
        return best["params"]
    return None


def run_validate(cfg, device, best_params=None):
    """使用最优参数做完整验证训练"""
    cfg = copy.deepcopy(cfg)

    if best_params:
        for k, v in best_params.items():
            if k in cfg["mappo"]:
                cfg["mappo"][k] = v

    print("\n" + "=" * 70)
    print("验证训练（目标：走完 stage_1 → stage_2 → stage_3）")
    print(f"  参数: lr={cfg['mappo']['lr']}, "
          f"entropy={cfg['mappo']['entropy_coef']}, "
          f"rollout={cfg['mappo']['rollout_steps']}")
    print("=" * 70)

    result = quick_train(cfg, device, max_updates=200, verbose=True)

    print("\n" + "=" * 70)
    print("验证结果")
    print("=" * 70)
    print(f"  最终到达阶段: stage_{result['final_stage']+1}")
    print(f"  阶段切换记录:")
    for t in result["stage_transitions"]:
        print(f"    stage_{t['from_stage']+1} → stage_{t['to_stage']+1} "
              f"@ update {t['update']} (coverage={t['coverage']:.3f})")
    print(f"  最终平均奖励: {result['final_reward']:.2f}")
    print(f"  最终平均覆盖率: {result['final_coverage']:.3f}")
    print(f"  奖励趋势(斜率): {result['reward_trend']:.4f}")
    print(f"  收敛判断: {'收敛中' if result['converging'] else '未收敛'}")
    print(f"  训练耗时: {result['elapsed_s']:.1f}s")

    if result["n_transitions"] >= 2:
        print("\n  [成功] 成功进入所有阶段！")
    elif result["n_transitions"] == 1:
        print("\n  [部分成功] 进入了 stage_2，但未到 stage_3（可增加 max_updates）")
    else:
        print("\n  [需关注] 未能进入 stage_2，建议检查阈值或增加训练步数")

    return result


def main():
    parser = argparse.ArgumentParser(description="参数扫描 + 验证训练")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--mode", type=str, choices=["sweep", "validate", "both"], default="both")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"设备: {device} | 种子: {args.seed}")

    best_params = None

    if args.mode in ("sweep", "both"):
        best_params = run_sweep(cfg, device)

    if args.mode in ("validate", "both"):
        run_validate(cfg, device, best_params)


if __name__ == "__main__":
    main()
