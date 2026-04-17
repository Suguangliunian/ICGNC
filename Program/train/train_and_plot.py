"""
训练并绘制损失曲线
===================
运行一次完整的快速训练，记录所有损失指标，最后生成出版级质量的损失曲线图。

用法：
  python train/train_and_plot.py --config configs/default.yaml
"""

import os, sys, time, copy, argparse, yaml
import numpy as np
import torch
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train.train_madrl import (
    BattlefieldEnv, CurriculumScheduler, MAPPOTrainer, RolloutBuffer
)


def train_with_logging(cfg, device, max_updates=200, return_trainer=False):
    """训练并返回完整的逐 update 日志；若 return_trainer=True 同时返回 MAPPOTrainer（用于保存检查点）。"""
    cfg = copy.deepcopy(cfg)
    cfg["mappo"]["total_updates"] = max_updates
    cfg["mappo"]["n_parallel_envs"] = 2
    cfg["mappo"]["rollout_steps"] = 32
    cfg["mappo"]["ppo_epochs"] = 3
    cfg["mappo"]["mini_batch_size"] = 32
    cfg["env"]["max_steps"] = 100

    trainer = MAPPOTrainer(cfg, device=device)
    obs = trainer.env.reset()
    episode_coverages = deque(maxlen=200)
    mappo_cfg = trainer.mappo_cfg
    rollout_steps = mappo_cfg["rollout_steps"]

    # 日志容器
    log = {
        "update": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "reward": [],
        "coverage": [],
        "stage": [],
        "stage_transitions": [],  # (update, from, to)
    }

    t0 = time.time()
    for update in range(1, max_updates + 1):
        trainer.total_updates = update
        buffer = RolloutBuffer(
            trainer.n_envs,
            trainer.n_agents,
            rollout_steps,
            trainer.device,
            obs_patch_size=trainer.env.obs_patch_size,
        )
        ep_rewards = np.zeros(trainer.n_envs)

        for step in range(rollout_steps):
            mode_act, move_act, mode_lp, move_lp, values, feats = \
                trainer.select_actions(obs, trainer.n_agents)
            next_obs, rewards, dones, infos = trainer.env.step(mode_act, move_act)
            flat_r = rewards.reshape(-1)
            norm_r = trainer.reward_normalizer.normalize(flat_r).reshape(rewards.shape)
            buffer.add(obs, mode_act, move_act, mode_lp, move_lp, norm_r, dones, values, feats)
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
            avg_cov = np.mean(list(episode_coverages)[-20:])
        else:
            metrics = trainer.env.get_metrics()
            avg_cov = np.mean([m["search_coverage"] for m in metrics])

        avg_reward = np.mean(trainer.episode_rewards) if trainer.episode_rewards else 0

        old_stage = trainer.scheduler.current_stage
        stage_changed = trainer.scheduler.report_performance(avg_cov)
        if stage_changed:
            log["stage_transitions"].append(
                (update, old_stage + 1, trainer.scheduler.current_stage + 1)
            )
            trainer._rebuild_env()
            obs = trainer.env.reset()
            episode_coverages.clear()

        # 记录
        log["update"].append(update)
        log["policy_loss"].append(losses["policy_loss"])
        log["value_loss"].append(losses["value_loss"])
        log["entropy"].append(losses["entropy"])
        log["reward"].append(avg_reward)
        log["coverage"].append(avg_cov)
        log["stage"].append(trainer.scheduler.current_stage + 1)

        if update % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [update {update:4d}/{max_updates}] "
                  f"stage={trainer.scheduler.stage_name} "
                  f"ploss={losses['policy_loss']:.4f} "
                  f"vloss={losses['value_loss']:.4f} "
                  f"ent={losses['entropy']:.4f} "
                  f"rew={avg_reward:.2f} "
                  f"cov={avg_cov:.3f} "
                  f"({elapsed:.0f}s)", flush=True)

    print(f"\n训练完成，共 {max_updates} updates，耗时 {time.time()-t0:.0f}s")
    if return_trainer:
        return log, trainer
    return log


def plot_losses(log, save_dir="figures"):
    """绘制出版级损失曲线"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 出版级样式
    SCIENCE_STYLE = {
        "font.size": 12,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.linewidth": 1.2,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.fontsize": 11,
        "legend.framealpha": 0.8,
        "legend.edgecolor": "0.8",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
    plt.rcParams.update(SCIENCE_STYLE)

    COLORS = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6"]

    os.makedirs(save_dir, exist_ok=True)
    updates = np.array(log["update"])
    transitions = log["stage_transitions"]

    def smooth(y, window=7):
        """简单滑动平均平滑"""
        y = np.array(y, dtype=float)
        if len(y) < window:
            return y
        kernel = np.ones(window) / window
        padded = np.pad(y, (window // 2, window // 2), mode="edge")
        return np.convolve(padded, kernel, mode="valid")[:len(y)]

    def add_stage_lines(ax, transitions):
        """在图上标注阶段切换竖线"""
        for (u, s_from, s_to) in transitions:
            ax.axvline(x=u, color="#832db6", linestyle="--", linewidth=1.2, alpha=0.7)
            ymin, ymax = ax.get_ylim()
            ax.text(u + 1, ymax - (ymax - ymin) * 0.08,
                    f"S{s_from}→S{s_to}", fontsize=9, color="#832db6",
                    fontweight="bold", va="top")

    # ========== 图1: 四合一面板 ==========
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    # (a) Policy Loss
    ax = axes[0, 0]
    ax.plot(updates, smooth(log["policy_loss"]), color=COLORS[0], linewidth=1.8, label="Policy Loss")
    ax.fill_between(updates, log["policy_loss"], alpha=0.12, color=COLORS[0])
    ax.set_xlabel("Update")
    ax.set_ylabel("Policy Loss")
    ax.set_title("(a) Policy Loss")
    add_stage_lines(ax, transitions)
    ax.legend(loc="best")

    # (b) Value Loss
    ax = axes[0, 1]
    ax.plot(updates, smooth(log["value_loss"]), color=COLORS[1], linewidth=1.8, label="Value Loss")
    ax.fill_between(updates, log["value_loss"], alpha=0.12, color=COLORS[1])
    ax.set_xlabel("Update")
    ax.set_ylabel("Value Loss")
    ax.set_title("(b) Value Loss")
    add_stage_lines(ax, transitions)
    ax.legend(loc="best")

    # (c) Entropy
    ax = axes[1, 0]
    ax.plot(updates, smooth(log["entropy"]), color=COLORS[2], linewidth=1.8, label="Entropy")
    ax.fill_between(updates, log["entropy"], alpha=0.12, color=COLORS[2])
    ax.set_xlabel("Update")
    ax.set_ylabel("Entropy")
    ax.set_title("(c) Policy Entropy")
    add_stage_lines(ax, transitions)
    ax.legend(loc="best")

    # (d) Reward & Coverage
    ax = axes[1, 1]
    ax.plot(updates, smooth(log["reward"]), color=COLORS[3], linewidth=1.8, label="Avg Reward")
    ax2 = ax.twinx()
    ax2.plot(updates, smooth(log["coverage"]), color=COLORS[4], linewidth=1.8,
             linestyle="-.", label="Coverage")
    ax.set_xlabel("Update")
    ax.set_ylabel("Average Reward", color=COLORS[3])
    ax2.set_ylabel("Coverage", color=COLORS[4])
    ax.set_title("(d) Reward & Coverage")
    add_stage_lines(ax, transitions)
    # 合并图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best")

    fig.suptitle("TAGA-MAPPO Training Curves (Curriculum Learning)", fontsize=15, fontweight="bold")

    path_png = os.path.join(save_dir, "training_losses.png")
    path_pdf = os.path.join(save_dir, "training_losses.pdf")
    fig.savefig(path_png)
    fig.savefig(path_pdf)
    plt.close(fig)
    print(f"[保存] {path_png}")
    print(f"[保存] {path_pdf}")

    # ========== 图2: 阶段进度时间线 ==========
    fig2, ax = plt.subplots(figsize=(10, 3), constrained_layout=True)
    stages = np.array(log["stage"])
    stage_colors = {1: COLORS[0], 2: COLORS[1], 3: COLORS[2]}
    for s in [1, 2, 3]:
        mask = stages == s
        if mask.any():
            ax.fill_between(updates, 0, 1, where=mask, alpha=0.5,
                            color=stage_colors[s], label=f"Stage {s}")
    ax.set_xlabel("Update")
    ax.set_yticks([])
    ax.set_title("Curriculum Stage Timeline")
    ax.legend(loc="upper right", ncol=3)
    path2 = os.path.join(save_dir, "stage_timeline.png")
    path2pdf = os.path.join(save_dir, "stage_timeline.pdf")
    fig2.savefig(path2)
    fig2.savefig(path2pdf)
    plt.close(fig2)
    print(f"[保存] {path2}")
    print(f"[保存] {path2pdf}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--output", default="figures")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 使用最优参数
    cfg["mappo"]["lr"] = 3e-4
    cfg["mappo"]["entropy_coef"] = 0.05

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    np.random.seed(42)
    torch.manual_seed(42)

    print(f"设备: {device} | 更新次数: {args.updates}")
    print("=" * 60)

    log = train_with_logging(cfg, device, max_updates=args.updates)
    plot_losses(log, save_dir=args.output)

    print("\n完成！图片保存在", args.output)


if __name__ == "__main__":
    main()
