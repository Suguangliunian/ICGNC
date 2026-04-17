"""
训练数据生成脚本
================
功能：
  - ExpertDataGenerator: 用训练好的 TAGA-MAPPO 生成专家轨迹
  - CounterfactualGenerator: 生成反事实数据
  - 数据格式：JSON，每条包含 prompt, cot, action, quality_score
  - 质量筛选：保留 top-75%
  - 目标：~45K 条数据

用法：
  python data/generate_data.py --config configs/default.yaml --checkpoint checkpoints/final.pt
"""

import os
import sys
import json
import argparse
import yaml
import numpy as np
import torch
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train.train_madrl import BattlefieldEnv


# ============================================================
# 场景描述模板
# ============================================================
class SceneDescriber:
    """将环境状态转换为自然语言场景描述"""

    # 模式名称映射
    MODE_NAMES = {0: "搜索模式", 1: "攻击模式"}

    # 动作名称映射
    ACTION_NAMES = {
        0: "向上移动", 1: "向下移动", 2: "向左移动", 3: "向右移动",
        4: "向左上移动", 5: "向右上移动", 6: "向左下移动", 7: "向右下移动",
        8: "原地停留", 9: "执行攻击",
    }

    @staticmethod
    def describe_scene(env, env_idx, uav_idx):
        """生成场景描述 prompt"""
        gs = env.grid_size
        pos = env.uav_pos[env_idx, uav_idx]
        energy = env.uav_energy[env_idx, uav_idx]
        mode = env.uav_mode[env_idx, uav_idx]
        step = env.step_count[env_idx]

        # 基本信息
        lines = [
            f"[场景] 战场大小: {gs}×{gs}, 当前步数: {step}/{env.max_steps}",
            f"[自身] UAV-{uav_idx} 位置: ({pos[0]:.0f}, {pos[1]:.0f}), "
            f"能量: {energy:.2f}, 模式: {SceneDescriber.MODE_NAMES[mode]}",
        ]

        # 友方信息
        allies = []
        for j in range(env.n_uavs):
            if j == uav_idx or not env.uav_alive[env_idx, j]:
                continue
            apos = env.uav_pos[env_idx, j]
            dist = np.sqrt(np.sum((pos - apos)**2))
            if dist <= env.R_comm:
                allies.append(f"UAV-{j}({apos[0]:.0f},{apos[1]:.0f},距离{dist:.0f})")
        if allies:
            lines.append(f"[友方] 通信范围内: {', '.join(allies)}")
        else:
            lines.append("[友方] 通信范围内无友方UAV")

        # 目标信息
        targets = []
        for t in range(env.n_targets):
            if env.target_discovered[env_idx, t] and env.target_alive[env_idx, t]:
                tpos = env.target_pos[env_idx, t]
                dist = np.sqrt(np.sum((pos - tpos)**2))
                targets.append(f"目标-{t}({tpos[0]:.0f},{tpos[1]:.0f},距离{dist:.0f})")
        if targets:
            lines.append(f"[目标] 已发现: {', '.join(targets)}")
        else:
            lines.append("[目标] 暂无已发现目标")

        # 威胁信息
        threats = []
        for t in range(env.n_threats):
            tpos = env.threat_pos[env_idx, t]
            dist = np.sqrt(np.sum((pos - tpos)**2))
            if dist <= env.obs_radius * 3:  # 较大范围内的威胁
                threats.append(f"威胁区-{t}({tpos[0]:.0f},{tpos[1]:.0f},半径{env.threat_radius[env_idx,t]:.0f},距离{dist:.0f})")
        if threats:
            lines.append(f"[威胁] 附近: {', '.join(threats)}")

        # 覆盖率
        coverage = env.explored[env_idx].sum() / (gs * gs)
        lines.append(f"[全局] 搜索覆盖率: {coverage:.1%}, 存活UAV: {env.uav_alive[env_idx].sum()}/{env.n_uavs}")

        return "\n".join(lines)

    @staticmethod
    def describe_reasoning(env, env_idx, uav_idx, mode_action, move_action, reward):
        """生成思维链推理过程"""
        pos = env.uav_pos[env_idx, uav_idx]
        energy = env.uav_energy[env_idx, uav_idx]

        lines = ["[思考]"]

        # 分析当前态势
        if energy < 0.3:
            lines.append("能量较低，需要谨慎行动，避免进入威胁区。")

        # 分析目标
        has_nearby_target = False
        for t in range(env.n_targets):
            if env.target_discovered[env_idx, t] and env.target_alive[env_idx, t]:
                dist = np.sqrt(np.sum((pos - env.target_pos[env_idx, t])**2))
                if dist <= env.obs_radius * 2:
                    has_nearby_target = True
                    lines.append(f"目标-{t}距离较近({dist:.0f})，考虑切换到攻击模式。")

        # 分析威胁
        for t in range(env.n_threats):
            dist = np.sqrt(np.sum((pos - env.threat_pos[env_idx, t])**2))
            if dist <= env.threat_radius[env_idx, t] + 3:
                lines.append(f"威胁区-{t}距离过近({dist:.0f})，需要规避。")

        # 模式决策
        if mode_action == 0:
            lines.append("当前选择搜索模式，扩大探索范围。")
        else:
            lines.append("当前选择攻击模式，准备打击目标。")

        # 动作决策
        lines.append(f"决定{SceneDescriber.ACTION_NAMES[move_action]}。")

        if reward > 0:
            lines.append(f"预期获得正向奖励({reward:.2f})。")

        return "\n".join(lines)

    @staticmethod
    def describe_action(mode_action, move_action):
        """生成动作描述"""
        mode_name = SceneDescriber.MODE_NAMES[mode_action]
        action_name = SceneDescriber.ACTION_NAMES[move_action]
        return f"[A] 模式: {mode_name}, 动作: {action_name}"


# ============================================================
# 专家数据生成器
# ============================================================
class ExpertDataGenerator:
    """用训练好的 TAGA-MAPPO 生成专家轨迹数据"""

    def __init__(self, cfg, checkpoint_path, device="cpu"):
        self.cfg = cfg
        self.device = device
        self.describer = SceneDescriber()

        # 加载模型
        print(f"[专家数据] 加载模型检查点: {checkpoint_path}")
        self.checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # 构建网络（复用训练器的网络结构）
        from model.madrl_gat import ObservationEncoder, TAGALayer
        import torch.nn as nn

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
        self.obs_encoder.load_state_dict(self.checkpoint["obs_encoder"])
        self.taga_layers.load_state_dict(self.checkpoint["taga_layers"])
        self.policy_mode.load_state_dict(self.checkpoint["policy_mode"])
        self.policy_action.load_state_dict(self.checkpoint["policy_action"])

        # 设为评估模式
        self.obs_encoder.eval()
        self.taga_layers.eval()
        self.policy_mode.eval()
        self.policy_action.eval()

    def generate(self, scenarios, n_episodes_per_scenario):
        """生成专家轨迹数据

        参数:
            scenarios: 场景配置列表
            n_episodes_per_scenario: 每个场景的回合数
        返回:
            data: 数据列表
        """
        all_data = []

        for sc_idx, scenario in enumerate(scenarios):
            print(f"\n[专家数据] 场景 {sc_idx + 1}/{len(scenarios)}: {scenario}")
            env_cfg = dict(self.cfg["env"])
            env_cfg.update(scenario)
            env = BattlefieldEnv(env_cfg, self.cfg["reward"], n_envs=1)

            for ep in range(n_episodes_per_scenario):
                obs = env.reset()
                ep_data = []

                for step in range(env.max_steps):
                    n_agents = env.n_uavs

                    # 用专家策略选择动作
                    with torch.no_grad():
                        grids = torch.tensor(obs["grids"], dtype=torch.float32, device=self.device)
                        states = torch.tensor(obs["states"], dtype=torch.float32, device=self.device)
                        B, N = grids.shape[0], grids.shape[1]

                        grids_flat = grids.reshape(B * N, 5, 11, 11)
                        states_flat = states.reshape(B * N, 6)
                        obs_feat = self.obs_encoder(grids_flat, states_flat)

                        # 简化：不做 GAT 聚合
                        gat_dim = self.cfg["model"]["taga"]["out_dim"]
                        gat_feat = torch.zeros(B * N, gat_dim, device=self.device)
                        feats = torch.cat([obs_feat, gat_feat], dim=-1)

                        mode_logits = self.policy_mode(feats)
                        mode_dist = torch.distributions.Categorical(logits=mode_logits)
                        mode_actions = mode_dist.sample()

                        n_modes = self.cfg["model"]["policy"]["n_modes"]
                        mode_onehot = torch.zeros(B * N, n_modes, device=self.device)
                        mode_onehot.scatter_(1, mode_actions.unsqueeze(1), 1.0)
                        action_input = torch.cat([feats, mode_onehot], dim=-1)
                        action_logits = self.policy_action(action_input)
                        action_dist = torch.distributions.Categorical(logits=action_logits)
                        move_actions = action_dist.sample()

                    mode_act = mode_actions.cpu().numpy().reshape(B, N)
                    move_act = move_actions.cpu().numpy().reshape(B, N)

                    # 执行动作
                    next_obs, rewards, dones, infos = env.step(mode_act, move_act)

                    # 为每个存活 UAV 生成数据
                    for i in range(n_agents):
                        if not env.uav_alive[0, i]:
                            continue

                        prompt = self.describer.describe_scene(env, 0, i)
                        cot = self.describer.describe_reasoning(
                            env, 0, i, mode_act[0, i], move_act[0, i], rewards[0, i]
                        )
                        action = self.describer.describe_action(mode_act[0, i], move_act[0, i])

                        # 质量分数：基于奖励归一化
                        quality = float(np.clip((rewards[0, i] + 5) / 10, 0, 1))

                        ep_data.append({
                            "prompt": prompt,
                            "cot": cot,
                            "action": action,
                            "quality_score": round(quality, 4),
                            "scenario": sc_idx,
                            "episode": ep,
                            "step": step,
                            "uav_id": i,
                        })

                    obs = next_obs
                    if dones[0]:
                        break

                all_data.extend(ep_data)

                if (ep + 1) % 100 == 0:
                    print(f"  回合 {ep + 1}/{n_episodes_per_scenario}, 累计数据: {len(all_data)} 条")

        print(f"\n[专家数据] 生成完成，共 {len(all_data)} 条")
        return all_data


# ============================================================
# 反事实数据生成器
# ============================================================
class CounterfactualGenerator:
    """反事实数据生成器

    通过修改专家决策的某些条件，生成"如果做了不同选择会怎样"的数据，
    用于增强模型的推理能力。
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.describer = SceneDescriber()

    def generate(self, expert_data, n_counterfactual):
        """从专家数据生成反事实数据

        策略：
        1. 随机选择一条专家数据
        2. 修改动作（选择次优动作）
        3. 生成对应的推理过程（解释为什么原动作更好）
        """
        print(f"\n[反事实] 开始生成 {n_counterfactual} 条反事实数据...")
        cf_data = []
        rng = np.random.RandomState(self.cfg.get("seed", 42))

        for i in range(n_counterfactual):
            # 随机选择一条专家数据
            idx = rng.randint(len(expert_data))
            original = expert_data[idx]

            # 生成反事实动作
            cf_mode = 1 - int("攻击" in original["action"])  # 翻转模式
            cf_move = rng.randint(0, 10)  # 随机动作

            # 构建反事实推理
            cf_cot_lines = ["[思考]"]
            cf_cot_lines.append(f"考虑替代方案：{SceneDescriber.MODE_NAMES[cf_mode]}，"
                               f"{SceneDescriber.ACTION_NAMES[cf_move]}。")

            # 分析为什么原方案更好
            if original["quality_score"] > 0.5:
                cf_cot_lines.append("但原方案的预期收益更高，因为：")
                if "搜索" in original["action"]:
                    cf_cot_lines.append("- 当前区域未充分探索，搜索模式能提高覆盖率。")
                else:
                    cf_cot_lines.append("- 目标在攻击范围内，应优先打击。")
                cf_cot_lines.append(f"因此维持原决策。")
                # 反事实数据的质量分数较低
                quality = max(0, original["quality_score"] - rng.uniform(0.2, 0.4))
            else:
                cf_cot_lines.append("该替代方案可能带来更好的结果。")
                quality = min(1, original["quality_score"] + rng.uniform(0.1, 0.3))

            cf_action = SceneDescriber.describe_action(cf_mode, cf_move)

            cf_data.append({
                "prompt": original["prompt"],
                "cot": "\n".join(cf_cot_lines),
                "action": cf_action,
                "quality_score": round(float(quality), 4),
                "is_counterfactual": True,
                "original_action": original["action"],
            })

            if (i + 1) % 5000 == 0:
                print(f"  已生成 {i + 1}/{n_counterfactual} 条")

        print(f"[反事实] 生成完成，共 {len(cf_data)} 条")
        return cf_data


# ============================================================
# 数据质量筛选
# ============================================================
class QualityFilter:
    """数据质量筛选器

    保留 top-75% 的高质量数据。
    """

    @staticmethod
    def filter(data, threshold_percentile=75):
        """按 quality_score 筛选数据

        参数:
            data: 数据列表
            threshold_percentile: 保留的百分位数（默认 top-75%）
        返回:
            filtered: 筛选后的数据
        """
        if not data:
            return data

        scores = [d["quality_score"] for d in data]
        # 计算阈值：保留 top-75% 意味着去掉最低的 25%
        cutoff = np.percentile(scores, 100 - threshold_percentile)

        filtered = [d for d in data if d["quality_score"] >= cutoff]

        print(f"[质量筛选] 原始: {len(data)} 条, "
              f"阈值: {cutoff:.4f}, "
              f"保留: {len(filtered)} 条 ({len(filtered)/len(data)*100:.1f}%)")

        # 统计信息
        filtered_scores = [d["quality_score"] for d in filtered]
        print(f"  质量分数: 均值={np.mean(filtered_scores):.4f}, "
              f"中位数={np.median(filtered_scores):.4f}, "
              f"最小={np.min(filtered_scores):.4f}")

        return filtered


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="训练数据生成")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/final.pt", help="MAPPO 模型检查点")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录（覆盖配置文件）")
    parser.add_argument("--device", type=str, default="cpu", help="设备")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--expert_only", action="store_true", help="只生成专家数据")
    parser.add_argument("--counterfactual_only", action="store_true", help="只生成反事实数据")
    args = parser.parse_args()

    # 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 覆盖参数
    if args.seed is not None:
        cfg["seed"] = args.seed
    seed = cfg.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_dir = args.output_dir or cfg["data_gen"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    data_cfg = cfg["data_gen"]
    scenarios = data_cfg["scenarios"]
    n_episodes = data_cfg["n_expert_episodes"] // len(scenarios)

    all_data = []

    # --- 1. 生成专家数据 ---
    if not args.counterfactual_only:
        print("=" * 60)
        print("阶段 1: 生成专家轨迹数据")
        print("=" * 60)

        expert_gen = ExpertDataGenerator(cfg, args.checkpoint, device=args.device)
        expert_data = expert_gen.generate(scenarios, n_episodes)
        all_data.extend(expert_data)

        # 保存专家数据
        expert_path = os.path.join(output_dir, "expert_data.json")
        with open(expert_path, "w", encoding="utf-8") as f:
            json.dump(expert_data, f, ensure_ascii=False, indent=2)
        print(f"[保存] 专家数据: {expert_path} ({len(expert_data)} 条)")
    else:
        # 加载已有专家数据
        expert_path = os.path.join(output_dir, "expert_data.json")
        print(f"[加载] 已有专家数据: {expert_path}")
        with open(expert_path, "r", encoding="utf-8") as f:
            expert_data = json.load(f)
        all_data.extend(expert_data)

    # --- 2. 生成反事实数据 ---
    if not args.expert_only:
        print("\n" + "=" * 60)
        print("阶段 2: 生成反事实数据")
        print("=" * 60)

        cf_gen = CounterfactualGenerator(cfg)
        cf_data = cf_gen.generate(expert_data, data_cfg["n_counterfactual"])
        all_data.extend(cf_data)

        # 保存反事实数据
        cf_path = os.path.join(output_dir, "counterfactual_data.json")
        with open(cf_path, "w", encoding="utf-8") as f:
            json.dump(cf_data, f, ensure_ascii=False, indent=2)
        print(f"[保存] 反事实数据: {cf_path} ({len(cf_data)} 条)")

    # --- 3. 质量筛选 ---
    print("\n" + "=" * 60)
    print("阶段 3: 质量筛选")
    print("=" * 60)

    threshold = int(data_cfg["quality_threshold"] * 100)
    filtered_data = QualityFilter.filter(all_data, threshold_percentile=threshold)

    # 如果数据量超过目标，随机采样
    target_total = data_cfg["target_total"]
    if len(filtered_data) > target_total:
        np.random.shuffle(filtered_data)
        filtered_data = filtered_data[:target_total]
        print(f"[采样] 截断至目标数量: {target_total} 条")

    # --- 4. 保存最终训练数据 ---
    train_path = os.path.join(output_dir, "train.json")
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"数据生成完成！")
    print(f"  总数据量: {len(filtered_data)} 条")
    print(f"  保存路径: {train_path}")
    print(f"  专家数据: {sum(1 for d in filtered_data if not d.get('is_counterfactual', False))} 条")
    print(f"  反事实数据: {sum(1 for d in filtered_data if d.get('is_counterfactual', False))} 条")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
