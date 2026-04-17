"""
uav_env.py — UAV集群搜索-攻击仿真环境
兼容 gym.Env 接口，支持多智能体、分层动作、动作mask、并行环境
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import gym
from gym import spaces

from utils import (
    euclidean_distance, pairwise_distances, direction_to_offset,
    signal_quality, signal_quality_batch, clip_position,
    in_threat_zone_batch, render_grid_text, create_matplotlib_frame,
)


# ============================================================
# 环境配置（默认参数）
# ============================================================
DEFAULT_CONFIG = {
    "grid_size": 100,       # 网格大小 100×100
    "n_uavs": 10,           # UAV数量
    "n_targets": 5,         # 目标数量
    "n_threats": 7,         # 威胁区域数量
    "r_comm": 20.0,         # 通信半径
    "r_search": 3,          # 搜索半径
    "r_attack": 1,          # 攻击半径
    "ammo": 2,              # 每架UAV弹药量
    "t_max": 500,           # 最大时间步
    "obs_radius": 5,        # 局部观测半径（11×11 → radius=5）
    "threat_radius_range": (5, 12),  # 威胁区域半径范围
    "target_hp": 1,         # 目标生命值（被攻击1次即摧毁）
}


# ============================================================
# 单环境类
# ============================================================
class UAVSearchAttackEnv(gym.Env):
    """UAV集群搜索-攻击仿真环境

    观测空间（每个智能体）：
        - local_grid: (5, 11, 11) — 5通道局部网格
            通道0: 已访问标记
            通道1: 目标位置（已发现的）
            通道2: 威胁区域
            通道3: 友方UAV位置
            通道4: 地形/边界
        - self_state: (6,) — [x, y, ammo, mode, timestep_norm, in_threat]

    动作空间（分层联合，共20维）：
        模式0（搜索）: 动作 0~8  → 9种移动方向
        模式1（攻击）: 动作 9~18 → 攻击目标索引(0~M-1) 或 不攻击
        动作19: 切换模式
    """

    metadata = {"render.modes": ["human", "text", "rgb_array"]}

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        # 合并配置
        self.cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.grid_size = self.cfg["grid_size"]
        self.n_uavs = self.cfg["n_uavs"]
        self.n_targets = self.cfg["n_targets"]
        self.n_threats = self.cfg["n_threats"]
        self.r_comm = self.cfg["r_comm"]
        self.r_search = self.cfg["r_search"]
        self.r_attack = self.cfg["r_attack"]
        self.max_ammo = self.cfg["ammo"]
        self.t_max = self.cfg["t_max"]
        self.obs_radius = self.cfg["obs_radius"]
        self.obs_size = 2 * self.obs_radius + 1  # 11

        # --- 动作空间 ---
        # 搜索动作: 0~8 (9个), 攻击动作: 9~18 (10个=目标数+不攻击), 切换模式: 19
        self.n_search_actions = 9
        self.n_attack_actions = self.n_targets + 1  # 最后一个=不攻击
        self.n_mode_switch = 1
        self.n_actions = self.n_search_actions + self.n_attack_actions + self.n_mode_switch  # 20

        # gym空间定义（单个智能体）
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Dict({
            "local_grid": spaces.Box(0, 1, shape=(5, self.obs_size, self.obs_size), dtype=np.float32),
            "self_state": spaces.Box(-1, 1, shape=(6,), dtype=np.float32),
        })

        # 内部状态（在reset中初始化）
        self._step_count = 0
        self._initialized = False

    # ---- 占位：当前 MAPPO 训练请使用 train_madrl.BattlefieldEnv ----
    def _not_impl(self, name: str) -> None:
        raise NotImplementedError(
            f"{name} 尚未实现。可训练路径为 train.train_madrl.BattlefieldEnv。"
        )

    def reset(self):
        self._not_impl("UAVSearchAttackEnv.reset")

    def step(self, actions):
        self._not_impl("UAVSearchAttackEnv.step")

    def _get_obs(self):
        self._not_impl("UAVSearchAttackEnv._get_obs")

    def _compute_rewards(self):
        self._not_impl("UAVSearchAttackEnv._compute_rewards")

    def get_action_mask(self):
        self._not_impl("UAVSearchAttackEnv.get_action_mask")

    def get_neighbor_info(self, agent_id):
        self._not_impl("UAVSearchAttackEnv.get_neighbor_info")

    def get_comm_topology(self):
        self._not_impl("UAVSearchAttackEnv.get_comm_topology")

    def render(self, mode="text"):
        self._not_impl("UAVSearchAttackEnv.render")


# ============================================================
# 并行环境包装器
# ============================================================
class VectorizedUAVEnv:
    """简易并行环境：内部维护多个独立的 UAVSearchAttackEnv 实例"""

    def __init__(self, n_envs: int = 4, config: Optional[Dict] = None):
        self.n_envs = n_envs
        self.envs = [UAVSearchAttackEnv(config) for _ in range(n_envs)]

    def reset(self):
        raise NotImplementedError(
            "VectorizedUAVEnv.reset 尚未实现；请使用 train.train_madrl.BattlefieldEnv。"
        )

    def step(self, actions_list):
        raise NotImplementedError(
            "VectorizedUAVEnv.step 尚未实现；请使用 train.train_madrl.BattlefieldEnv。"
        )

    def get_action_masks(self):
        raise NotImplementedError(
            "VectorizedUAVEnv.get_action_masks 尚未实现；请使用 train.train_madrl.BattlefieldEnv。"
        )
