"""
TAGA-MAPPO 训练脚本
====================
功能：
  - MAPPO 训练循环（GAE + PPO clip）
  - 课程学习调度器（3阶段，基于性能自适应切换）
  - 奖励归一化（running mean/var）
  - 训练日志 & 模型保存

用法：
  python train/train_madrl.py --config configs/default.yaml
"""

import os
import sys
import time
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

# 将项目根目录加入 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# 简易战场环境（向量化版本）
# ============================================================
class BattlefieldEnv:
    """简易战场环境（支持向量化并行）

    每个环境实例模拟一个 grid_size×grid_size 的网格战场，
    包含 n_uavs 架 UAV、n_targets 个目标、n_threats 个威胁区。
    """

    def __init__(self, cfg_env, cfg_reward, n_envs=1, device="cpu"):
        self.n_envs = n_envs
        self.device = device
        self.grid_size = cfg_env["grid_size"]
        self.n_uavs = cfg_env["n_uavs"]
        self.n_targets = cfg_env["n_targets"]
        self.n_threats = cfg_env["n_threats"]
        self.obs_radius = cfg_env.get("obs_radius", 5)
        # 局部网格边长（与 CNN 输入一致，全项目唯一真源）
        self.obs_patch_size = 2 * self.obs_radius + 1
        if self.n_threats > 0:
            r = self.obs_radius
            self._threat_local_offsets = np.stack(
                np.mgrid[-r : r + 1, -r : r + 1], axis=-1
            ).astype(np.float32)
        else:
            self._threat_local_offsets = None
        self.max_steps = cfg_env.get("max_steps", 200)
        self.threat_damage = cfg_env.get("threat_damage", 0.3)
        self.energy_per_step = cfg_env.get("energy_per_step", 0.005)
        self.R_comm = cfg_env.get("R_comm", 20.0)
        self.rw = cfg_reward

        # 动作映射：10个动作 = 8方向移动 + 停留 + 攻击
        # 方向：上、下、左、右、左上、右上、左下、右下、停留、攻击
        self._dx = [0, 0, -1, 1, -1, 1, -1, 1, 0, 0]
        self._dy = [1, -1, 0, 0, 1, 1, -1, -1, 0, 0]

    def reset(self):
        """重置所有并行环境，返回观测字典"""
        gs = self.grid_size
        self.step_count = np.zeros(self.n_envs, dtype=np.int32)

        # UAV 位置、速度、能量、模式、存活状态
        self.uav_pos = np.random.randint(0, gs, (self.n_envs, self.n_uavs, 2)).astype(np.float32)
        self.uav_vel = np.zeros((self.n_envs, self.n_uavs, 2), dtype=np.float32)
        self.uav_energy = np.ones((self.n_envs, self.n_uavs), dtype=np.float32)
        self.uav_mode = np.zeros((self.n_envs, self.n_uavs), dtype=np.int32)  # 0=搜索
        self.uav_alive = np.ones((self.n_envs, self.n_uavs), dtype=bool)

        # 目标位置和存活状态
        self.target_pos = np.random.randint(10, gs - 10, (self.n_envs, self.n_targets, 2)).astype(np.float32)
        self.target_alive = np.ones((self.n_envs, self.n_targets), dtype=bool)
        self.target_discovered = np.zeros((self.n_envs, self.n_targets), dtype=bool)

        # 威胁区中心和半径
        self.threat_pos = np.random.randint(5, gs - 5, (self.n_envs, self.n_threats, 2)).astype(np.float32)
        self.threat_radius = np.full((self.n_envs, self.n_threats), 5.0, dtype=np.float32)

        # 已探索地图
        self.explored = np.zeros((self.n_envs, gs, gs), dtype=bool)

        # 通信时间矩阵
        self.last_comm_time = np.zeros((self.n_envs, self.n_uavs, self.n_uavs), dtype=np.float32)

        return self._get_obs()

    def _get_obs(self):
        """构建观测：局部网格 [n_envs, n_uavs, 5, S, S] + 自身状态 [n_envs, n_uavs, 6]（S=obs_patch_size）

        使用 numpy 切片替代逐像素 Python 循环，性能提升约 50-100x。
        """
        gs = self.grid_size
        r = self.obs_radius
        sz = 2 * r + 1

        grids = np.zeros((self.n_envs, self.n_uavs, 5, sz, sz), dtype=np.float32)

        # 预填充 padding 版的 explored 和 valid mask（一次性构建，避免重复 pad）
        padded_explored = np.pad(self.explored, ((0, 0), (r, r), (r, r)),
                                 mode='constant', constant_values=0)
        padded_valid = np.pad(np.ones((self.n_envs, gs, gs), dtype=np.float32),
                              ((0, 0), (r, r), (r, r)), mode='constant', constant_values=0)

        local_offsets = self._threat_local_offsets  # 构造期缓存，避免每步分配 mgrid

        for e in range(self.n_envs):
            for i in range(self.n_uavs):
                if not self.uav_alive[e, i]:
                    continue
                cx, cy = int(self.uav_pos[e, i, 0]), int(self.uav_pos[e, i, 1])
                px, py = cx + r, cy + r  # padded 坐标

                # 通道 0 (障碍) & 通道 1 (已探索)：numpy 切片
                window_valid = padded_valid[e, px - r:px + r + 1, py - r:py + r + 1]
                window_explored = padded_explored[e, px - r:px + r + 1, py - r:py + r + 1]
                grids[e, i, 0] = 1.0 - window_valid
                grids[e, i, 1] = window_explored

                # 通道 2: 友方 UAV（向量化距离计算）
                alive_others = self.uav_alive[e].copy()
                alive_others[i] = False
                if alive_others.any():
                    other_pos = self.uav_pos[e, alive_others]  # [K, 2]
                    ox = (other_pos[:, 0] - cx + r).astype(int)
                    oy = (other_pos[:, 1] - cy + r).astype(int)
                    valid = (ox >= 0) & (ox < sz) & (oy >= 0) & (oy < sz)
                    if valid.any():
                        grids[e, i, 2, ox[valid], oy[valid]] = 1.0

                # 通道 3: 已发现且存活的目标
                vis = self.target_discovered[e] & self.target_alive[e]
                if vis.any():
                    tpos = self.target_pos[e, vis]
                    ox = (tpos[:, 0] - cx + r).astype(int)
                    oy = (tpos[:, 1] - cy + r).astype(int)
                    valid = (ox >= 0) & (ox < sz) & (oy >= 0) & (oy < sz)
                    if valid.any():
                        grids[e, i, 3, ox[valid], oy[valid]] = 1.0

                # 通道 4: 威胁区（向量化距离矩阵）
                if self.n_threats > 0 and local_offsets is not None:
                    global_coords = local_offsets + np.array([cx, cy], dtype=np.float32)  # [sz, sz, 2]
                    for t in range(self.n_threats):
                        d = np.sqrt(np.sum((global_coords - self.threat_pos[e, t]) ** 2, axis=-1))
                        grids[e, i, 4] = np.maximum(grids[e, i, 4], (d <= self.threat_radius[e, t]).astype(np.float32))

        # 自身状态：完全向量化
        states = np.zeros((self.n_envs, self.n_uavs, 6), dtype=np.float32)
        states[:, :, 0] = self.uav_pos[:, :, 0] / gs
        states[:, :, 1] = self.uav_pos[:, :, 1] / gs
        states[:, :, 2] = self.uav_vel[:, :, 0]
        states[:, :, 3] = self.uav_vel[:, :, 1]
        states[:, :, 4] = self.uav_energy
        states[:, :, 5] = self.uav_mode.astype(np.float32)
        # 死亡 agent 的状态置零
        dead = ~self.uav_alive
        states[dead] = 0.0

        return {
            "grids": grids,
            "states": states,
            "positions": self.uav_pos.copy(),
            "modes": self.uav_mode.copy(),
            "alive": self.uav_alive.copy(),
            "last_comm_time": self.last_comm_time.copy(),
        }

    def step(self, mode_actions, move_actions):
        """执行动作，返回 (obs, rewards, dones, infos)

        使用向量化 numpy 操作替代 Python 循环，关键优化：
        - 探索标记用 slice 赋值替代逐像素循环
        - 威胁/目标距离用 broadcasting 批量计算
        """
        gs = self.grid_size
        r = self.obs_radius
        rewards = np.zeros((self.n_envs, self.n_uavs), dtype=np.float32)
        infos = [{"targets_destroyed": 0, "coverage": 0.0} for _ in range(self.n_envs)]

        dx_arr = np.array(self._dx, dtype=np.float32)
        dy_arr = np.array(self._dy, dtype=np.float32)

        # --- 向量化移动 ---
        self.uav_mode = mode_actions.copy()
        move_mask = (move_actions < 9) & self.uav_alive  # [E, N]

        new_x = self.uav_pos[:, :, 0] + dx_arr[move_actions]
        new_y = self.uav_pos[:, :, 1] + dy_arr[move_actions]
        new_x = np.clip(new_x, 0, gs - 1)
        new_y = np.clip(new_y, 0, gs - 1)

        vel_x = np.where(move_mask, new_x - self.uav_pos[:, :, 0], 0)
        vel_y = np.where(move_mask, new_y - self.uav_pos[:, :, 1], 0)
        self.uav_vel[:, :, 0] = vel_x
        self.uav_vel[:, :, 1] = vel_y
        self.uav_pos[:, :, 0] = np.where(move_mask, new_x, self.uav_pos[:, :, 0])
        self.uav_pos[:, :, 1] = np.where(move_mask, new_y, self.uav_pos[:, :, 1])

        # --- 向量化能量消耗 ---
        self.uav_energy -= self.energy_per_step * self.uav_alive.astype(np.float32)
        energy_dead = (self.uav_energy <= 0) & self.uav_alive
        rewards[energy_dead] += self.rw["uav_lost_penalty"]
        self.uav_alive[energy_dead] = False

        # --- 探索标记（numpy slice 赋值，消除内层 dx/dy 循环） ---
        for e in range(self.n_envs):
            for i in range(self.n_uavs):
                if not self.uav_alive[e, i]:
                    continue
                cx, cy = int(self.uav_pos[e, i, 0]), int(self.uav_pos[e, i, 1])
                x0, x1 = max(0, cx - r), min(gs, cx + r + 1)
                y0, y1 = max(0, cy - r), min(gs, cy + r + 1)
                patch = self.explored[e, x0:x1, y0:y1]
                n_new = int((~patch).sum())
                if n_new > 0:
                    self.explored[e, x0:x1, y0:y1] = True
                    rewards[e, i] += n_new * self.rw["search_coverage"]

        # --- 威胁检测（向量化距离） ---
        if self.n_threats > 0:
            for e in range(self.n_envs):
                alive_idx = np.where(self.uav_alive[e])[0]
                if len(alive_idx) == 0:
                    continue
                uav_p = self.uav_pos[e, alive_idx]  # [K, 2]
                threat_p = self.threat_pos[e]  # [T, 2]
                dists = np.sqrt(((uav_p[:, None, :] - threat_p[None, :, :]) ** 2).sum(axis=-1))  # [K, T]
                in_threat = dists <= self.threat_radius[e][None, :]  # [K, T]
                for ki, i in enumerate(alive_idx):
                    if in_threat[ki].any():
                        self.uav_energy[e, i] -= self.threat_damage * in_threat[ki].sum()
                        rewards[e, i] += self.rw["threat_penalty"] * in_threat[ki].sum()
                        if self.uav_energy[e, i] <= 0:
                            self.uav_alive[e, i] = False
                            rewards[e, i] += self.rw["uav_lost_penalty"]

        # --- 目标发现和攻击 ---
        for e in range(self.n_envs):
            alive_idx = np.where(self.uav_alive[e])[0]
            if len(alive_idx) == 0 or self.n_targets == 0:
                continue
            uav_p = self.uav_pos[e, alive_idx]  # [K, 2]
            tgt_p = self.target_pos[e]  # [T, 2]
            dists = np.sqrt(((uav_p[:, None, :] - tgt_p[None, :, :]) ** 2).sum(axis=-1))  # [K, T]

            for ki, i in enumerate(alive_idx):
                # 发现目标
                for t in range(self.n_targets):
                    if self.target_alive[e, t] and not self.target_discovered[e, t]:
                        if dists[ki, t] <= r:
                            self.target_discovered[e, t] = True
                            rewards[e, i] += self.rw["target_discover"]

                # 攻击
                if move_actions[e, i] == 9 and mode_actions[e, i] == 1:
                    for t in range(self.n_targets):
                        if not self.target_alive[e, t] or not self.target_discovered[e, t]:
                            continue
                        if dists[ki, t] <= r:
                            n_attackers = 1
                            for kj, j in enumerate(alive_idx):
                                if j == i:
                                    continue
                                if move_actions[e, j] == 9 and mode_actions[e, j] == 1 and dists[kj, t] <= r:
                                    n_attackers += 1
                            self.target_alive[e, t] = False
                            rewards[e, i] += self.rw["target_destroy"]
                            if n_attackers >= 2:
                                rewards[e, i] += self.rw["cooperative_bonus"]
                            infos[e]["targets_destroyed"] += 1
                            break

            # 时间惩罚（只对存活 agent）
            rewards[e, alive_idx] += self.rw["time_penalty"]

        # --- 通信时间更新（向量化距离矩阵） ---
        for e in range(self.n_envs):
            alive_mask = self.uav_alive[e]
            if alive_mask.sum() < 2:
                continue
            uav_p = self.uav_pos[e]  # [N, 2]
            d_mat = np.sqrt(((uav_p[:, None, :] - uav_p[None, :, :]) ** 2).sum(axis=-1))  # [N, N]
            comm = (d_mat <= self.R_comm) & alive_mask[:, None] & alive_mask[None, :]
            np.fill_diagonal(comm, False)
            self.last_comm_time[e][comm] = self.step_count[e]

            infos[e]["coverage"] = float(self.explored[e].sum()) / (gs * gs)

        self.step_count += 1

        # 结束条件（向量化）
        all_targets_done = ~self.target_alive.any(axis=1)
        all_uavs_dead = ~self.uav_alive.any(axis=1)
        timeout = self.step_count >= self.max_steps
        dones = all_targets_done | all_uavs_dead | timeout

        obs = self._get_obs()
        return obs, rewards, dones, infos

    def reset_single(self, e):
        """重置单个并行环境，保持其他环境不变"""
        gs = self.grid_size
        self.step_count[e] = 0
        self.uav_pos[e] = np.random.randint(0, gs, (self.n_uavs, 2)).astype(np.float32)
        self.uav_vel[e] = np.zeros((self.n_uavs, 2), dtype=np.float32)
        self.uav_energy[e] = np.ones(self.n_uavs, dtype=np.float32)
        self.uav_mode[e] = np.zeros(self.n_uavs, dtype=np.int32)
        self.uav_alive[e] = np.ones(self.n_uavs, dtype=bool)
        self.target_pos[e] = np.random.randint(10, gs - 10, (self.n_targets, 2)).astype(np.float32)
        self.target_alive[e] = np.ones(self.n_targets, dtype=bool)
        self.target_discovered[e] = np.zeros(self.n_targets, dtype=bool)
        self.threat_pos[e] = np.random.randint(5, gs - 5, (self.n_threats, 2)).astype(np.float32)
        self.threat_radius[e] = np.full(self.n_threats, 5.0, dtype=np.float32)
        self.explored[e] = np.zeros((gs, gs), dtype=bool)
        self.last_comm_time[e] = np.zeros((self.n_uavs, self.n_uavs), dtype=np.float32)

    def get_metrics(self):
        """返回当前环境的评估指标"""
        metrics = []
        gs = self.grid_size
        for e in range(self.n_envs):
            coverage = self.explored[e].sum() / (gs * gs)
            destroyed = (~self.target_alive[e]).sum()
            survival = self.uav_alive[e].sum() / self.n_uavs
            metrics.append({
                "search_coverage": float(coverage),
                "target_destroy_rate": float(destroyed / max(self.n_targets, 1)),
                "uav_survival_rate": float(survival),
                "steps": int(self.step_count[e]),
            })
        return metrics


# ============================================================
# Rollout Buffer
# ============================================================
class RolloutBuffer:
    """存储 rollout 数据，支持 GAE 计算和 mini-batch 采样"""

    def __init__(self, n_envs, n_agents, rollout_steps, device="cpu", obs_patch_size=11):
        self.n_envs = n_envs
        self.n_agents = n_agents
        self.rollout_steps = rollout_steps
        self.device = device
        self.obs_patch_size = obs_patch_size
        self.reset()

    def reset(self):
        """清空缓冲区"""
        self.grids = []       # 局部网格观测
        self.states = []      # 自身状态
        self.positions = []   # UAV 位置
        self.modes = []       # UAV 模式
        self.alive_masks = [] # 存活掩码
        self.last_comm = []   # 通信时间矩阵
        self.features = []    # 完整特征（含 GAT）
        self.mode_actions = []
        self.move_actions = []
        self.mode_log_probs = []
        self.move_log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.ptr = 0

    def add(self, obs, mode_act, move_act, mode_lp, move_lp, reward, done, value, feats=None):
        """添加一步数据"""
        self.grids.append(obs["grids"])
        self.states.append(obs["states"])
        self.positions.append(obs["positions"])
        self.modes.append(obs["modes"])
        self.alive_masks.append(obs["alive"])
        self.last_comm.append(obs["last_comm_time"])
        if feats is not None:
            self.features.append(feats)
        self.mode_actions.append(mode_act)
        self.move_actions.append(move_act)
        self.mode_log_probs.append(mode_lp)
        self.move_log_probs.append(move_lp)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.ptr += 1

    def compute_gae(self, last_value, gamma, lambda_gae):
        """计算 GAE 优势估计和回报

        参数:
            last_value: [n_envs, n_agents] 最后一步的价值估计
            gamma: 折扣因子
            lambda_gae: GAE lambda
        """
        rewards = np.array(self.rewards)       # [T, n_envs, n_agents]
        values = np.array(self.values)         # [T, n_envs, n_agents]
        dones = np.array(self.dones)           # [T, n_envs]
        masks = np.array(self.alive_masks)     # [T, n_envs, n_agents]

        T = self.ptr
        advantages = np.zeros_like(rewards)
        last_gae = np.zeros((self.n_envs, self.n_agents), dtype=np.float32)

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            # done 掩码扩展到 agent 维度
            done_mask = 1.0 - dones[t][:, None]  # [n_envs, 1]
            delta = rewards[t] + gamma * next_value * done_mask - values[t]
            last_gae = delta + gamma * lambda_gae * done_mask * last_gae
            # 只对存活 agent 计算优势
            advantages[t] = last_gae * masks[t]

        returns = advantages + values
        self.advantages = advantages
        self.returns = returns

    def get_batches(self, mini_batch_size):
        """生成 mini-batch 迭代器

        将 [T, n_envs, n_agents, ...] 展平为 [T*n_envs*n_agents, ...] 后随机采样
        """
        T = self.ptr
        N = T * self.n_envs * self.n_agents

        # 转换为 tensor 并展平
        def to_flat(arr, extra_dims=None):
            a = np.array(arr)
            if extra_dims:
                a = a.reshape(N, *extra_dims)
            else:
                a = a.reshape(N)
            return torch.tensor(a, dtype=torch.float32, device=self.device)

        ps = self.obs_patch_size
        grids = to_flat(self.grids, (5, ps, ps))
        states_flat = to_flat(self.states, (6,))
        mode_acts = to_flat(self.mode_actions).long()
        move_acts = to_flat(self.move_actions).long()
        mode_lps = to_flat(self.mode_log_probs)
        move_lps = to_flat(self.move_log_probs)
        advantages = to_flat(self.advantages)
        returns = to_flat(self.returns)
        alive = to_flat(self.alive_masks).bool()

        # 存储的完整特征（含 GAT）
        has_features = len(self.features) > 0
        if has_features:
            feat_dim = np.array(self.features[0]).shape[-1]
            stored_feats = to_flat(self.features, (feat_dim,))

        # 归一化优势（只对存活 agent）
        valid_adv = advantages[alive]
        if len(valid_adv) > 1:
            adv_mean = valid_adv.mean()
            adv_std = valid_adv.std() + 1e-8
            advantages = (advantages - adv_mean) / adv_std

        # 随机打乱索引
        indices = torch.randperm(N, device=self.device)
        for start in range(0, N, mini_batch_size):
            end = min(start + mini_batch_size, N)
            idx = indices[start:end]
            batch = {
                "grids": grids[idx],
                "states": states_flat[idx],
                "mode_actions": mode_acts[idx],
                "move_actions": move_acts[idx],
                "old_mode_log_probs": mode_lps[idx],
                "old_move_log_probs": move_lps[idx],
                "advantages": advantages[idx],
                "returns": returns[idx],
                "alive": alive[idx],
            }
            if has_features:
                batch["stored_features"] = stored_feats[idx]
            yield batch


# ============================================================
# 奖励归一化器
# ============================================================
class RewardNormalizer:
    """奖励归一化器（Running Mean/Var）"""

    def __init__(self, shape=(), clip=10.0):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4
        self.clip = clip

    def update(self, x):
        """更新统计量"""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if hasattr(x, 'shape') else 1

        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x):
        """归一化"""
        self.update(x)
        return np.clip((x - self.mean) / (np.sqrt(self.var) + 1e-8), -self.clip, self.clip)


# ============================================================
# 课程学习调度器
# ============================================================
class CurriculumScheduler:
    """课程学习调度器

    三阶段课程（与 `configs/default.yaml` 对齐），基于性能指标自适应切换：
      Stage 1: 3 UAV, 2 目标, 0 威胁, 30×30
      Stage 2: 6 UAV, 3 目标, 3 威胁, 50×50
      Stage 3: 10 UAV, 5 目标, 7 威胁, 100×100
    """

    def __init__(self, stages_cfg):
        self.stages = stages_cfg
        self.current_stage = 0
        self.stage_updates = 0
        self.performance_history = deque(maxlen=50)

    @property
    def current_config(self):
        """返回当前阶段的环境配置"""
        return self.stages[self.current_stage]

    @property
    def stage_name(self):
        return self.stages[self.current_stage]["name"]

    def report_performance(self, coverage):
        """报告当前性能，判断是否切换阶段"""
        self.performance_history.append(coverage)
        self.stage_updates += 1

        if self.current_stage >= len(self.stages) - 1:
            return False  # 已经是最后阶段

        stage = self.stages[self.current_stage]
        min_updates = stage.get("min_updates", 500)
        threshold = stage.get("advance_threshold", 0.7)

        if self.stage_updates >= min_updates and len(self.performance_history) >= 20:
            avg_perf = np.mean(list(self.performance_history)[-20:])
            if avg_perf >= threshold:
                print(f"\n{'='*60}")
                print(f"[课程学习] 阶段 {self.current_stage + 1} → {self.current_stage + 2}")
                print(f"  平均覆盖率: {avg_perf:.3f} >= 阈值 {threshold:.3f}")
                print(f"  已训练 {self.stage_updates} 步")
                print(f"{'='*60}\n")
                self.current_stage += 1
                self.stage_updates = 0
                self.performance_history.clear()
                return True  # 阶段切换
        return False

    def make_env_config(self, base_cfg):
        """根据当前阶段生成环境配置"""
        stage = self.current_config
        cfg = dict(base_cfg)
        cfg["n_uavs"] = stage["n_uavs"]
        cfg["n_targets"] = stage["n_targets"]
        cfg["n_threats"] = stage["n_threats"]
        cfg["grid_size"] = stage["grid_size"]
        return cfg


# ============================================================
# MAPPO 训练器
# ============================================================
class MAPPOTrainer:
    """MAPPO 训练器

    包含完整的训练循环：rollout 采集 → GAE 计算 → PPO 更新
    """

    def __init__(self, cfg, device="cpu"):
        self.cfg = cfg
        self.device = device
        self.mappo_cfg = cfg["mappo"]
        self.model_cfg = cfg["model"]

        # 导入模型
        from model.madrl_gat import ObservationEncoder, TAGALayer

        # 初始化课程调度器
        self.scheduler = CurriculumScheduler(cfg["curriculum"]["stages"])
        env_cfg = self.scheduler.make_env_config(cfg["env"])

        # 创建并行环境
        self.n_envs = self.mappo_cfg["n_parallel_envs"]
        self.env = BattlefieldEnv(env_cfg, cfg["reward"], n_envs=self.n_envs)

        # 当前 agent 数量（随课程变化）
        self.n_agents = env_cfg["n_uavs"]

        # 构建网络
        self._build_networks()

        # 优化器
        all_params = list(self.obs_encoder.parameters()) + \
                     list(self.taga_layers.parameters()) + \
                     list(self.policy_mode.parameters()) + \
                     list(self.policy_action.parameters()) + \
                     list(self.critic.parameters())
        self.optimizer = optim.Adam(all_params, lr=self.mappo_cfg["lr"], eps=1e-5)

        # 奖励归一化
        self.reward_normalizer = RewardNormalizer()

        # 日志
        self.total_updates = 0
        self.episode_rewards = deque(maxlen=100)

    def _build_networks(self):
        """构建所有网络模块"""
        from model.madrl_gat import ObservationEncoder, TAGALayer

        mcfg = self.model_cfg

        # 观测编码器
        self.obs_encoder = ObservationEncoder(
            grid_channels=mcfg["obs_encoder"]["grid_channels"],
            state_dim=mcfg["obs_encoder"]["state_dim"],
        ).to(self.device)

        # TAGA 层
        taga_cfg = mcfg["taga"]
        self.taga_layers = nn.ModuleList([
            TAGALayer(
                in_dim=taga_cfg["in_dim"] if i == 0 else taga_cfg["out_dim"],
                out_dim=taga_cfg["out_dim"],
                n_heads=taga_cfg["n_heads"],
                R_comm=taga_cfg["R_comm"],
            ).to(self.device)
            for i in range(taga_cfg["n_layers"])
        ])

        # 分层策略：模式选择头 + 动作选择头
        feat_dim = mcfg["obs_encoder"]["cnn_out_dim"] + taga_cfg["out_dim"]  # 128 + 64 = 192
        hidden = mcfg["policy"]["hidden_dim"]
        n_modes = mcfg["policy"]["n_modes"]
        n_actions = mcfg["policy"]["n_actions_per_mode"]

        self.policy_mode = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_modes),
        ).to(self.device)

        self.policy_action = nn.Sequential(
            nn.Linear(feat_dim + n_modes, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        ).to(self.device)

        # 中心化 Critic
        # 输入：所有 agent 的特征拼接（简化版：用均值池化）
        self.critic = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        ).to(self.device)

    def _get_features(self, obs_dict, n_agents):
        """从观测字典提取特征

        返回:
            agent_feats: [n_envs * n_agents, feat_dim]
            gat_feats:   [n_envs * n_agents, gat_dim]
        """
        grids = torch.tensor(obs_dict["grids"], dtype=torch.float32, device=self.device)
        states = torch.tensor(obs_dict["states"], dtype=torch.float32, device=self.device)
        positions = torch.tensor(obs_dict["positions"], dtype=torch.float32, device=self.device)
        modes = torch.tensor(obs_dict["modes"], dtype=torch.long, device=self.device)
        last_comm = torch.tensor(obs_dict["last_comm_time"], dtype=torch.float32, device=self.device)

        B = grids.shape[0]  # n_envs
        N = n_agents

        # 编码观测 [B*N, 128]
        ps = self.env.obs_patch_size
        grids_flat = grids.reshape(B * N, 5, ps, ps)
        states_flat = states.reshape(B * N, 6)
        obs_feat = self.obs_encoder(grids_flat, states_flat)  # [B*N, 128]

        # TAGA 聚合（逐环境处理）
        gat_feats_list = []
        for e in range(B):
            h = obs_feat[e * N:(e + 1) * N]  # [N, 128]
            # 投影到 TAGA 输入维度
            if h.shape[-1] != self.model_cfg["taga"]["in_dim"]:
                # 简单截断/填充到 in_dim
                h_proj = h[:, :self.model_cfg["taga"]["in_dim"]]
            else:
                h_proj = h

            pos_e = positions[e]  # [N, 2]
            mode_e = modes[e]     # [N]
            lc_e = last_comm[e]   # [N, N]
            t = float(self.total_updates)

            for taga in self.taga_layers:
                h_proj = taga(h_proj, pos_e, mode_e, lc_e, t)

            gat_feats_list.append(h_proj)

        gat_feats = torch.cat(gat_feats_list, dim=0)  # [B*N, gat_dim]

        # 拼接观测特征和 GAT 特征
        combined = torch.cat([obs_feat, gat_feats], dim=-1)  # [B*N, 192]
        return combined

    def select_actions(self, obs_dict, n_agents):
        """选择动作，返回动作、log_prob 和完整特征"""
        with torch.no_grad():
            feats = self._get_features(obs_dict, n_agents)  # [B*N, feat_dim]
            B = self.n_envs
            N = n_agents

            # 模式选择
            mode_logits = self.policy_mode(feats)  # [B*N, n_modes]
            mode_dist = torch.distributions.Categorical(logits=mode_logits)
            mode_actions = mode_dist.sample()  # [B*N]
            mode_log_probs = mode_dist.log_prob(mode_actions)

            # 动作选择（条件于模式）
            n_modes = self.model_cfg["policy"]["n_modes"]
            mode_onehot = torch.zeros(B * N, n_modes, device=self.device)
            mode_onehot.scatter_(1, mode_actions.unsqueeze(1), 1.0)
            action_input = torch.cat([feats, mode_onehot], dim=-1)
            action_logits = self.policy_action(action_input)
            action_dist = torch.distributions.Categorical(logits=action_logits)
            move_actions = action_dist.sample()
            move_log_probs = action_dist.log_prob(move_actions)

            # 价值估计
            values = self.critic(feats).squeeze(-1)  # [B*N]

            # 保存完整特征（含 GAT）用于 evaluate_actions
            stored_feats = feats.cpu().numpy().reshape(B, N, -1)

        return (
            mode_actions.cpu().numpy().reshape(B, N),
            move_actions.cpu().numpy().reshape(B, N),
            mode_log_probs.cpu().numpy().reshape(B, N),
            move_log_probs.cpu().numpy().reshape(B, N),
            values.cpu().numpy().reshape(B, N),
            stored_feats,
        )

    def evaluate_actions(self, batch):
        """评估动作（用于 PPO 更新）

        使用 rollout 时存储的完整特征（含 GAT），避免与 select_actions 的表示不一致。
        对观测编码器部分重新前向传播以获得梯度，GAT 部分使用存储值作为常量拼接。
        """
        grids = batch["grids"]
        states = batch["states"]

        # 重新编码观测以获得梯度
        obs_feat = self.obs_encoder(grids, states)  # [batch, 128]

        if "stored_features" in batch:
            # 从存储的完整特征中提取 GAT 部分（后 gat_dim 维）
            gat_dim = self.model_cfg["taga"]["out_dim"]
            obs_dim = batch["stored_features"].shape[-1] - gat_dim
            gat_feat = batch["stored_features"][:, obs_dim:].detach()
            feats = torch.cat([obs_feat, gat_feat], dim=-1)
        else:
            gat_dim = self.model_cfg["taga"]["out_dim"]
            gat_feat = torch.zeros(obs_feat.shape[0], gat_dim, device=self.device)
            feats = torch.cat([obs_feat, gat_feat], dim=-1)

        # 模式 log_prob 和 entropy
        mode_logits = self.policy_mode(feats)
        mode_dist = torch.distributions.Categorical(logits=mode_logits)
        mode_log_probs = mode_dist.log_prob(batch["mode_actions"])
        mode_entropy = mode_dist.entropy()

        # 动作 log_prob 和 entropy
        n_modes = self.model_cfg["policy"]["n_modes"]
        mode_onehot = torch.zeros(feats.shape[0], n_modes, device=self.device)
        mode_onehot.scatter_(1, batch["mode_actions"].unsqueeze(1), 1.0)
        action_input = torch.cat([feats, mode_onehot], dim=-1)
        action_logits = self.policy_action(action_input)
        action_dist = torch.distributions.Categorical(logits=action_logits)
        move_log_probs = action_dist.log_prob(batch["move_actions"])
        move_entropy = action_dist.entropy()

        # 价值估计
        values = self.critic(feats).squeeze(-1)

        return mode_log_probs, move_log_probs, values, mode_entropy + move_entropy

    def ppo_update(self, buffer):
        """PPO 更新（actor + critic）"""
        cfg = self.mappo_cfg
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_batches = 0

        for epoch in range(cfg["ppo_epochs"]):
            for batch in buffer.get_batches(cfg["mini_batch_size"]):
                alive = batch["alive"]
                if not alive.any():
                    continue

                mode_lp, move_lp, values, entropy = self.evaluate_actions(batch)

                # 联合 log_prob
                new_log_prob = mode_lp + move_lp
                old_log_prob = batch["old_mode_log_probs"] + batch["old_move_log_probs"]

                # 重要性采样比
                ratio = torch.exp(new_log_prob - old_log_prob)

                # PPO clip
                adv = batch["advantages"]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - cfg["epsilon_clip"], 1 + cfg["epsilon_clip"]) * adv
                policy_loss = -torch.min(surr1, surr2)[alive].mean()

                # 价值损失
                value_loss = F.mse_loss(values[alive], batch["returns"][alive])

                # 熵奖励
                entropy_loss = -entropy[alive].mean()

                # 总损失
                loss = (policy_loss
                        + cfg["value_loss_coef"] * value_loss
                        + cfg["entropy_coef"] * entropy_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.obs_encoder.parameters()) +
                    list(self.taga_layers.parameters()) +
                    list(self.policy_mode.parameters()) +
                    list(self.policy_action.parameters()) +
                    list(self.critic.parameters()),
                    cfg["max_grad_norm"]
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy[alive].mean().item()
                n_batches += 1

        if n_batches > 0:
            return {
                "policy_loss": total_policy_loss / n_batches,
                "value_loss": total_value_loss / n_batches,
                "entropy": total_entropy / n_batches,
            }
        return {"policy_loss": 0, "value_loss": 0, "entropy": 0}

    def _rebuild_env(self):
        """课程切换后重建环境"""
        env_cfg = self.scheduler.make_env_config(self.cfg["env"])
        self.n_agents = env_cfg["n_uavs"]
        self.env = BattlefieldEnv(env_cfg, self.cfg["reward"], n_envs=self.n_envs)

    def save_checkpoint(self, path, extra=None):
        """保存模型检查点"""
        ckpt = {
            "obs_encoder": self.obs_encoder.state_dict(),
            "taga_layers": self.taga_layers.state_dict(),
            "policy_mode": self.policy_mode.state_dict(),
            "policy_action": self.policy_action.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_updates": self.total_updates,
            "curriculum_stage": self.scheduler.current_stage,
        }
        if extra:
            ckpt.update(extra)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(ckpt, path)
        print(f"[保存] 检查点已保存: {path}")

    def train(self):
        """主训练循环"""
        cfg = self.mappo_cfg
        total_updates = cfg["total_updates"]
        rollout_steps = cfg["rollout_steps"]
        log_interval = cfg["log_interval"]
        save_interval = cfg["save_interval"]
        ckpt_dir = self.cfg["paths"]["checkpoint_dir"]

        print("=" * 60)
        print("TAGA-MAPPO 训练开始")
        print(f"  总更新次数: {total_updates}")
        print(f"  并行环境数: {self.n_envs}")
        print(f"  Rollout 步数: {rollout_steps}")
        print(f"  设备: {self.device}")
        print("=" * 60)

        start_time = time.time()
        obs = self.env.reset()

        # episode 级别的覆盖率追踪（用于课程学习判断）
        episode_coverages = deque(maxlen=100)

        for update in range(1, total_updates + 1):
            self.total_updates = update

            # --- Rollout 采集 ---
            buffer = RolloutBuffer(
                self.n_envs,
                self.n_agents,
                rollout_steps,
                self.device,
                obs_patch_size=self.env.obs_patch_size,
            )
            ep_rewards = np.zeros(self.n_envs)

            for step in range(rollout_steps):
                mode_act, move_act, mode_lp, move_lp, values, feats = self.select_actions(obs, self.n_agents)
                next_obs, rewards, dones, infos = self.env.step(mode_act, move_act)

                # 奖励归一化
                flat_rewards = rewards.reshape(-1)
                norm_rewards = self.reward_normalizer.normalize(flat_rewards).reshape(rewards.shape)

                buffer.add(obs, mode_act, move_act, mode_lp, move_lp, norm_rewards, dones, values, feats)
                ep_rewards += rewards.mean(axis=1)

                # 处理 done 的环境：逐个重置，记录 episode 覆盖率
                for e in range(self.n_envs):
                    if dones[e]:
                        self.episode_rewards.append(ep_rewards[e])
                        episode_coverages.append(infos[e]["coverage"])
                        ep_rewards[e] = 0
                        self.env.reset_single(e)

                obs = self.env._get_obs()

            # --- GAE 计算 ---
            with torch.no_grad():
                feats = self._get_features(obs, self.n_agents)
                last_values = self.critic(feats).squeeze(-1)
                last_values = last_values.cpu().numpy().reshape(self.n_envs, self.n_agents)
            buffer.compute_gae(last_values, cfg["gamma"], cfg["lambda_gae"])

            # --- PPO 更新 ---
            losses = self.ppo_update(buffer)

            # --- 课程学习检查 ---
            # 使用 episode 完成时的覆盖率（而非 snapshot）
            if len(episode_coverages) > 0:
                avg_coverage = np.mean(list(episode_coverages)[-20:])
            else:
                metrics = self.env.get_metrics()
                avg_coverage = np.mean([m["search_coverage"] for m in metrics])

            stage_changed = self.scheduler.report_performance(avg_coverage)
            if stage_changed:
                self._rebuild_env()
                obs = self.env.reset()
                episode_coverages.clear()
                self.save_checkpoint(
                    os.path.join(ckpt_dir, f"stage_{self.scheduler.current_stage}.pt")
                )

            # --- 日志 ---
            if update % log_interval == 0:
                elapsed = time.time() - start_time
                avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
                print(f"[更新 {update:5d}/{total_updates}] "
                      f"阶段={self.scheduler.stage_name} | "
                      f"奖励={avg_reward:7.2f} | "
                      f"覆盖率={avg_coverage:.3f} | "
                      f"策略损失={losses['policy_loss']:.4f} | "
                      f"价值损失={losses['value_loss']:.4f} | "
                      f"熵={losses['entropy']:.4f} | "
                      f"耗时={elapsed:.0f}s")

            # --- 定期保存 ---
            if update % save_interval == 0:
                self.save_checkpoint(
                    os.path.join(ckpt_dir, f"update_{update}.pt")
                )

        # 训练结束，保存最终模型
        total_time = time.time() - start_time
        self.save_checkpoint(os.path.join(ckpt_dir, "final.pt"))
        print(f"\n训练完成！总耗时: {total_time / 60:.1f} 分钟")


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="TAGA-MAPPO 训练")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--device", type=str, default="auto", help="设备 (cpu/cuda/auto)")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（覆盖配置文件）")
    parser.add_argument("--total_updates", type=int, default=None, help="总更新次数（覆盖配置文件）")
    args = parser.parse_args()

    _prog_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(_prog_root)
    if not os.path.isabs(args.config):
        args.config = os.path.normpath(os.path.join(_prog_root, args.config))

    # 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 覆盖参数
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.total_updates is not None:
        cfg["mappo"]["total_updates"] = args.total_updates

    # 设备选择
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # 设置随机种子
    seed = cfg.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    print(f"随机种子: {seed}")
    print(f"设备: {device}")

    # 创建输出目录
    os.makedirs(cfg["paths"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["log_dir"], exist_ok=True)

    # 开始训练
    trainer = MAPPOTrainer(cfg, device=device)
    trainer.train()


if __name__ == "__main__":
    main()
