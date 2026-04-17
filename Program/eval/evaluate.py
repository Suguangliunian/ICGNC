"""
统一评估框架
=============
功能：
  - 评估指标：搜索覆盖率、目标摧毁率、任务完成时间、UAV存活率、协同攻击比例
  - 支持评估 TAGA-MAPPO 和 TAGA-LLM
  - 多次运行统计（均值±标准差）
  - Welch's t-test 显著性检验
  - 结果保存为 JSON 和表格
  - 轨迹热力图可视化

用法：
  python eval/evaluate.py --config configs/default.yaml --method mappo --checkpoint checkpoints/final.pt
  python eval/evaluate.py --config configs/default.yaml --method llm --checkpoint checkpoints/qwen_lora/final
  python eval/evaluate.py --config configs/default.yaml --compare mappo llm
"""

import os
import sys
import json
import argparse
import yaml
import numpy as np
import torch
from scipy import stats
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train.train_madrl import BattlefieldEnv

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[警告] 未安装 matplotlib，可视化功能不可用")


# ============================================================
# 评估指标计算器
# ============================================================
class MetricsCalculator:
    """评估指标计算器"""

    METRIC_NAMES = {
        "search_coverage": "搜索覆盖率",
        "target_destroy_rate": "目标摧毁率",
        "completion_time": "任务完成时间",
        "uav_survival_rate": "UAV存活率",
        "cooperative_attack_ratio": "协同攻击比例",
    }

    @staticmethod
    def compute(env, trajectory_info):
        """从环境和轨迹信息计算所有指标

        参数:
            env: BattlefieldEnv 实例
            trajectory_info: 轨迹记录字典
        返回:
            metrics: 指标字典
        """
        gs = env.grid_size
        metrics = {}

        # 搜索覆盖率
        metrics["search_coverage"] = float(env.explored[0].sum() / (gs * gs))

        # 目标摧毁率
        n_destroyed = int((~env.target_alive[0]).sum())
        metrics["target_destroy_rate"] = float(n_destroyed / max(env.n_targets, 1))

        # 任务完成时间（归一化到 [0, 1]）
        metrics["completion_time"] = float(env.step_count[0] / env.max_steps)

        # UAV 存活率
        metrics["uav_survival_rate"] = float(env.uav_alive[0].sum() / env.n_uavs)

        # 协同攻击比例
        total_attacks = trajectory_info.get("total_attacks", 0)
        coop_attacks = trajectory_info.get("cooperative_attacks", 0)
        metrics["cooperative_attack_ratio"] = float(
            coop_attacks / max(total_attacks, 1)
        )

        return metrics


# ============================================================
# TAGA-MAPPO 评估器
# ============================================================
class MAPPOEvaluator:
    """TAGA-MAPPO 评估器"""

    def __init__(self, cfg, checkpoint_path, device="cpu"):
        self.cfg = cfg
        self.device = device

        # 加载模型（复用 ExpertDataGenerator 的加载逻辑）
        from model.madrl_gat import ObservationEncoder, TAGALayer
        import torch.nn as nn

        print(f"[评估] 加载 MAPPO 模型: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        mcfg = cfg["model"]
        self.obs_encoder = ObservationEncoder(
            grid_channels=mcfg["obs_encoder"]["grid_channels"],
            state_dim=mcfg["obs_encoder"]["state_dim"],
        ).to(device)

        taga_cfg = mcfg["taga"]
        self.taga_layers = nn.ModuleList([
            TAGALayer(
                in_dim=taga_cfg["in_dim"] if i == 0 else taga_cfg["out_dim"],
                out_dim=taga_cfg["out_dim"],
                n_heads=taga_cfg["n_heads"],
                R_comm=taga_cfg["R_comm"],
            ).to(device)
            for i in range(taga_cfg["n_layers"])
        ])

        feat_dim = mcfg["obs_encoder"]["cnn_out_dim"] + taga_cfg["out_dim"]
        hidden = mcfg["policy"]["hidden_dim"]
        n_modes = mcfg["policy"]["n_modes"]
        n_actions = mcfg["policy"]["n_actions_per_mode"]

        self.policy_mode = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_modes),
        ).to(device)

        self.policy_action = nn.Sequential(
            nn.Linear(feat_dim + n_modes, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        ).to(device)

        # 加载权重
        self.obs_encoder.load_state_dict(ckpt["obs_encoder"])
        self.taga_layers.load_state_dict(ckpt["taga_layers"])
        self.policy_mode.load_state_dict(ckpt["policy_mode"])
        self.policy_action.load_state_dict(ckpt["policy_action"])

        self.obs_encoder.eval()
        self.taga_layers.eval()
        self.policy_mode.eval()
        self.policy_action.eval()

    def evaluate_episode(self, env):
        """评估单个回合"""
        obs = env.reset()
        trajectory = {"positions": [], "total_attacks": 0, "cooperative_attacks": 0}

        for step in range(env.max_steps):
            n_agents = env.n_uavs

            with torch.no_grad():
                grids = torch.tensor(obs["grids"], dtype=torch.float32, device=self.device)
                states_t = torch.tensor(obs["states"], dtype=torch.float32, device=self.device)
                B, N = grids.shape[0], grids.shape[1]

                ps = env.obs_patch_size
                grids_flat = grids.reshape(B * N, 5, ps, ps)
                states_flat = states_t.reshape(B * N, 6)
                obs_feat = self.obs_encoder(grids_flat, states_flat)

                positions = torch.tensor(obs["positions"], dtype=torch.float32, device=self.device)
                modes = torch.tensor(obs["modes"], dtype=torch.long, device=self.device)
                last_comm = torch.tensor(obs["last_comm_time"], dtype=torch.float32, device=self.device)
                mcfg = self.cfg["model"]
                taga_in = mcfg["taga"]["in_dim"]
                gat_feats_list = []
                t_step = float(step)
                for e in range(B):
                    h = obs_feat[e * N : (e + 1) * N]
                    if h.shape[-1] != taga_in:
                        h_proj = h[:, :taga_in]
                    else:
                        h_proj = h
                    pos_e = positions[e]
                    mode_e = modes[e]
                    lc_e = last_comm[e]
                    for taga in self.taga_layers:
                        h_proj = taga(h_proj, pos_e, mode_e, lc_e, t_step)
                    gat_feats_list.append(h_proj)
                gat_feat = torch.cat(gat_feats_list, dim=0)
                feats = torch.cat([obs_feat, gat_feat], dim=-1)

                mode_logits = self.policy_mode(feats)
                mode_actions = mode_logits.argmax(dim=-1)  # 贪心选择

                n_modes = self.cfg["model"]["policy"]["n_modes"]
                mode_onehot = torch.zeros(B * N, n_modes, device=self.device)
                mode_onehot.scatter_(1, mode_actions.unsqueeze(1), 1.0)
                action_input = torch.cat([feats, mode_onehot], dim=-1)
                action_logits = self.policy_action(action_input)
                move_actions = action_logits.argmax(dim=-1)

            mode_act = mode_actions.cpu().numpy().reshape(B, N)
            move_act = move_actions.cpu().numpy().reshape(B, N)

            # 记录轨迹
            trajectory["positions"].append(env.uav_pos[0].copy())

            # 统计攻击
            for i in range(n_agents):
                if move_act[0, i] == 9 and mode_act[0, i] == 1:
                    trajectory["total_attacks"] += 1
                    # 检查是否协同
                    for j in range(n_agents):
                        if j != i and move_act[0, j] == 9 and mode_act[0, j] == 1:
                            if env.uav_alive[0, j]:
                                dist = np.sqrt(np.sum((env.uav_pos[0, i] - env.uav_pos[0, j])**2))
                                if dist <= env.R_comm:
                                    trajectory["cooperative_attacks"] += 1
                                    break

            obs, rewards, dones, infos = env.step(mode_act, move_act)
            if dones[0]:
                break

        metrics = MetricsCalculator.compute(env, trajectory)
        return metrics, trajectory


# ============================================================
# TAGA-LLM 评估器
# ============================================================
class LLMEvaluator:
    """TAGA-LLM 评估器（Qwen + LoRA）"""

    def __init__(self, cfg, checkpoint_path, device="auto"):
        self.cfg = cfg
        self.device = device

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel
        except ImportError:
            raise RuntimeError("需要安装 transformers 和 peft 库")

        print(f"[评估] 加载 LLM 模型: {checkpoint_path}")

        # 加载 tokenizer 和模型
        base_model_name = cfg["qwen"]["model_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto" if device == "cuda" else None,
        )
        self.model = PeftModel.from_pretrained(base_model, checkpoint_path)
        self.model.eval()

        from data.generate_data import SceneDescriber
        self.describer = SceneDescriber()

    def _parse_llm_action(self, text):
        """从 LLM 输出解析动作"""
        mode = 0  # 默认搜索
        action = 8  # 默认停留

        if "[A]" in text:
            action_text = text.split("[A]")[-1].strip()
            if "攻击" in action_text:
                mode = 1
                action = 9
            elif "向上" in action_text:
                action = 0
            elif "向下" in action_text:
                action = 1
            elif "向左上" in action_text:
                action = 4
            elif "向右上" in action_text:
                action = 5
            elif "向左下" in action_text:
                action = 6
            elif "向右下" in action_text:
                action = 7
            elif "向左" in action_text:
                action = 2
            elif "向右" in action_text:
                action = 3
            elif "停留" in action_text:
                action = 8

            if "攻击模式" in action_text:
                mode = 1
            elif "搜索模式" in action_text:
                mode = 0

        return mode, action

    def evaluate_episode(self, env):
        """评估单个回合"""
        obs = env.reset()
        trajectory = {"positions": [], "total_attacks": 0, "cooperative_attacks": 0}

        for step in range(env.max_steps):
            n_agents = env.n_uavs
            mode_actions = np.zeros((1, n_agents), dtype=np.int32)
            move_actions = np.zeros((1, n_agents), dtype=np.int32)

            for i in range(n_agents):
                if not env.uav_alive[0, i]:
                    continue

                # 生成场景描述
                prompt = self.describer.describe_scene(env, 0, i)
                prompt += "\n请分析当前态势并给出决策。"

                # LLM 推理
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.3,
                        do_sample=True,
                        top_p=0.9,
                    )
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                # 解析动作
                mode, action = self._parse_llm_action(response)
                mode_actions[0, i] = mode
                move_actions[0, i] = action

            # 记录轨迹
            trajectory["positions"].append(env.uav_pos[0].copy())

            # 统计攻击
            for i in range(n_agents):
                if move_actions[0, i] == 9 and mode_actions[0, i] == 1:
                    trajectory["total_attacks"] += 1
                    for j in range(n_agents):
                        if j != i and move_actions[0, j] == 9 and mode_actions[0, j] == 1:
                            if env.uav_alive[0, j]:
                                dist = np.sqrt(np.sum((env.uav_pos[0, i] - env.uav_pos[0, j])**2))
                                if dist <= env.R_comm:
                                    trajectory["cooperative_attacks"] += 1
                                    break

            obs, rewards, dones, infos = env.step(mode_actions, move_actions)
            if dones[0]:
                break

        metrics = MetricsCalculator.compute(env, trajectory)
        return metrics, trajectory


# ============================================================
# 统计分析
# ============================================================
class StatisticalAnalyzer:
    """统计分析工具"""

    @staticmethod
    def summarize(all_metrics):
        """计算均值±标准差

        参数:
            all_metrics: 列表，每个元素是一个指标字典
        返回:
            summary: {metric_name: {"mean": ..., "std": ..., "values": [...]}}
        """
        summary = {}
        if not all_metrics:
            return summary

        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "values": values,
            }
        return summary

    @staticmethod
    def welch_t_test(summary_a, summary_b, alpha=0.05):
        """Welch's t-test 显著性检验

        参数:
            summary_a, summary_b: summarize() 的输出
            alpha: 显著性水平
        返回:
            results: {metric: {"t_stat": ..., "p_value": ..., "significant": bool, "better": str}}
        """
        results = {}
        for key in summary_a:
            if key not in summary_b:
                continue
            vals_a = summary_a[key]["values"]
            vals_b = summary_b[key]["values"]

            t_stat, p_value = stats.ttest_ind(vals_a, vals_b, equal_var=False)

            # 判断哪个更好（对于 completion_time，越小越好）
            mean_a = np.mean(vals_a)
            mean_b = np.mean(vals_b)
            if key == "completion_time":
                better = "A" if mean_a < mean_b else "B"
            else:
                better = "A" if mean_a > mean_b else "B"

            results[key] = {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": bool(p_value < alpha),
                "better": better,
                "mean_a": float(mean_a),
                "mean_b": float(mean_b),
            }
        return results

    @staticmethod
    def print_comparison_table(summary_a, summary_b, test_results, name_a="MAPPO", name_b="LLM"):
        """打印对比表格"""
        print(f"\n{'='*80}")
        print(f"{'指标':<20} {'':>2} {name_a:>18} {'':>2} {name_b:>18} {'':>2} {'p值':>8} {'显著':>4}")
        print(f"{'-'*80}")

        for key in summary_a:
            if key not in summary_b:
                continue
            cn_name = MetricsCalculator.METRIC_NAMES.get(key, key)
            ma = summary_a[key]
            mb = summary_b[key]
            tr = test_results.get(key, {})

            sig = "✓" if tr.get("significant", False) else ""
            better = tr.get("better", "")
            marker_a = "★" if better == "A" else ""
            marker_b = "★" if better == "B" else ""

            print(f"{cn_name:<18} {marker_a:>2} "
                  f"{ma['mean']:.4f}±{ma['std']:.4f} {marker_b:>2} "
                  f"{mb['mean']:.4f}±{mb['std']:.4f}   "
                  f"{tr.get('p_value', 0):.4f}  {sig:>4}")

        print(f"{'='*80}")
        print(f"★ = 该指标上更优的方法, ✓ = 差异在 α=0.05 水平下显著")


# ============================================================
# 可视化
# ============================================================
class Visualizer:
    """评估结果可视化"""

    @staticmethod
    def plot_trajectory_heatmap(trajectories, grid_size, save_path, title="UAV 轨迹热力图"):
        """绘制轨迹热力图

        参数:
            trajectories: 轨迹列表，每个元素是 [steps, n_uavs, 2] 的位置数组
            grid_size: 网格大小
            save_path: 保存路径
            title: 图标题
        """
        if not HAS_MPL:
            print("[警告] matplotlib 不可用，跳过可视化")
            return

        heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)

        for traj in trajectories:
            positions = traj["positions"]
            for step_pos in positions:
                for uav_pos in step_pos:
                    x, y = int(np.clip(uav_pos[0], 0, grid_size - 1)), \
                           int(np.clip(uav_pos[1], 0, grid_size - 1))
                    heatmap[x, y] += 1

        # 归一化
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        im = ax.imshow(heatmap, cmap="hot", interpolation="bilinear", origin="lower")
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.colorbar(im, ax=ax, label="访问频率（归一化）")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[可视化] 热力图已保存: {save_path}")

    @staticmethod
    def plot_metrics_comparison(summary_a, summary_b, save_path, name_a="MAPPO", name_b="LLM"):
        """绘制指标对比柱状图"""
        if not HAS_MPL:
            print("[警告] matplotlib 不可用，跳过可视化")
            return

        metrics = list(summary_a.keys())
        cn_names = [MetricsCalculator.METRIC_NAMES.get(m, m) for m in metrics]
        means_a = [summary_a[m]["mean"] for m in metrics]
        stds_a = [summary_a[m]["std"] for m in metrics]
        means_b = [summary_b[m]["mean"] for m in metrics]
        stds_b = [summary_b[m]["std"] for m in metrics]

        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, means_a, width, yerr=stds_a,
                       label=name_a, color="#4C72B0", capsize=3)
        bars2 = ax.bar(x + width/2, means_b, width, yerr=stds_b,
                       label=name_b, color="#DD8452", capsize=3)

        ax.set_ylabel("指标值")
        ax.set_title("TAGA-MAPPO vs TAGA-LLM 评估对比")
        ax.set_xticks(x)
        ax.set_xticklabels(cn_names, rotation=15, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[可视化] 对比图已保存: {save_path}")


# ============================================================
# 主函数
# ============================================================
def run_evaluation(evaluator, cfg, n_episodes, method_name):
    """运行评估"""
    env = BattlefieldEnv(cfg["env"], cfg["reward"], n_envs=1)
    all_metrics = []
    all_trajectories = []

    print(f"\n[评估] 开始评估 {method_name}，共 {n_episodes} 回合")
    for ep in range(n_episodes):
        metrics, trajectory = evaluator.evaluate_episode(env)
        all_metrics.append(metrics)
        all_trajectories.append(trajectory)

        if (ep + 1) % 5 == 0:
            avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
            print(f"  回合 {ep + 1}/{n_episodes} | "
                  f"覆盖率={avg['search_coverage']:.3f} | "
                  f"摧毁率={avg['target_destroy_rate']:.3f} | "
                  f"存活率={avg['uav_survival_rate']:.3f}")

    summary = StatisticalAnalyzer.summarize(all_metrics)
    return summary, all_trajectories


def main():
    parser = argparse.ArgumentParser(description="统一评估框架")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--method", type=str, choices=["mappo", "llm"], default=None, help="评估方法")
    parser.add_argument("--checkpoint", type=str, default=None, help="模型检查点路径")
    parser.add_argument("--n_episodes", type=int, default=None, help="评估回合数")
    parser.add_argument("--device", type=str, default="cpu", help="设备")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--compare", nargs=2, metavar=("CKPT_A", "CKPT_B"),
                        help="对比两个模型: --compare mappo_ckpt llm_ckpt")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--visualize", action="store_true", help="生成可视化")
    args = parser.parse_args()

    # 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 设置参数
    seed = args.seed or cfg.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_episodes = args.n_episodes or cfg["eval"]["n_episodes"]
    output_dir = args.output_dir or cfg["eval"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    if args.compare:
        # --- 对比模式 ---
        ckpt_a, ckpt_b = args.compare
        print("=" * 60)
        print("对比评估模式")
        print("=" * 60)

        # 评估 MAPPO
        eval_a = MAPPOEvaluator(cfg, ckpt_a, device=args.device)
        summary_a, traj_a = run_evaluation(eval_a, cfg, n_episodes, "TAGA-MAPPO")

        # 评估 LLM
        eval_b = LLMEvaluator(cfg, ckpt_b, device=args.device)
        summary_b, traj_b = run_evaluation(eval_b, cfg, n_episodes, "TAGA-LLM")

        # 统计检验
        test_results = StatisticalAnalyzer.welch_t_test(
            summary_a, summary_b, alpha=cfg["eval"]["significance_level"]
        )

        # 打印对比表格
        StatisticalAnalyzer.print_comparison_table(summary_a, summary_b, test_results)

        # 保存结果
        results = {
            "mappo": {k: {"mean": v["mean"], "std": v["std"]} for k, v in summary_a.items()},
            "llm": {k: {"mean": v["mean"], "std": v["std"]} for k, v in summary_b.items()},
            "t_test": test_results,
            "n_episodes": n_episodes,
            "seed": seed,
        }
        result_path = os.path.join(output_dir, "comparison_results.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[保存] 对比结果: {result_path}")

        # 可视化
        if args.visualize:
            gs = cfg["env"]["grid_size"]
            Visualizer.plot_trajectory_heatmap(
                traj_a, gs, os.path.join(output_dir, "heatmap_mappo.png"), "TAGA-MAPPO 轨迹热力图"
            )
            Visualizer.plot_trajectory_heatmap(
                traj_b, gs, os.path.join(output_dir, "heatmap_llm.png"), "TAGA-LLM 轨迹热力图"
            )
            Visualizer.plot_metrics_comparison(
                summary_a, summary_b, os.path.join(output_dir, "comparison.png")
            )

    elif args.method:
        # --- 单方法评估 ---
        print("=" * 60)
        print(f"评估方法: {args.method.upper()}")
        print("=" * 60)

        checkpoint = args.checkpoint
        if checkpoint is None:
            if args.method == "mappo":
                checkpoint = os.path.join(cfg["paths"]["checkpoint_dir"], "final.pt")
            else:
                checkpoint = os.path.join(cfg["paths"]["checkpoint_dir"], "qwen_lora", "final")

        if args.method == "mappo":
            evaluator = MAPPOEvaluator(cfg, checkpoint, device=args.device)
            method_name = "TAGA-MAPPO"
        else:
            evaluator = LLMEvaluator(cfg, checkpoint, device=args.device)
            method_name = "TAGA-LLM"

        summary, trajectories = run_evaluation(evaluator, cfg, n_episodes, method_name)

        # 打印结果
        print(f"\n{'='*60}")
        print(f"{method_name} 评估结果 ({n_episodes} 回合)")
        print(f"{'='*60}")
        for key, val in summary.items():
            cn_name = MetricsCalculator.METRIC_NAMES.get(key, key)
            print(f"  {cn_name}: {val['mean']:.4f} ± {val['std']:.4f}")

        # 保存结果
        results = {
            "method": args.method,
            "metrics": {k: {"mean": v["mean"], "std": v["std"]} for k, v in summary.items()},
            "n_episodes": n_episodes,
            "seed": seed,
        }
        result_path = os.path.join(output_dir, f"{args.method}_results.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[保存] 结果: {result_path}")

        # 可视化
        if args.visualize:
            gs = cfg["env"]["grid_size"]
            Visualizer.plot_trajectory_heatmap(
                trajectories, gs,
                os.path.join(output_dir, f"heatmap_{args.method}.png"),
                f"{method_name} 轨迹热力图"
            )

    else:
        parser.print_help()
        print("\n请指定 --method 或 --compare 参数")


if __name__ == "__main__":
    main()
