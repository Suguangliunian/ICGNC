"""
TAGA-MAPPO 模型（阶段A）
=========================
包含：
  - TAGALayer: 拓扑自适应图注意力层
  - ObservationEncoder: 观测编码器（CNN + MLP 融合）
  - HierarchicalPolicy: 分层策略网络（模式选择 + 动作选择）
  - CentralizedCritic: 中心化价值网络
  - TAGAMAPPO: 完整的 TAGA-MAPPO 智能体
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ============================================================
# TAGALayer: Topology-Adaptive Graph Attention 层
# ============================================================
class TAGALayer(nn.Module):
    """拓扑自适应图注意力层

    三个自适应因子：
      - lambda_d: 距离衰减因子，通信半径外的邻居权重归零
      - lambda_m: 任务感知因子，相同模式的邻居获得更高权重
      - lambda_t: 信息新鲜度因子，长时间未通信的邻居权重衰减

    参数:
        in_dim:  输入节点特征维度
        out_dim: 输出节点特征维度
        n_heads: 注意力头数（默认4）
        R_comm:  通信半径（默认20）
    """

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4, R_comm: float = 20.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.R_comm = R_comm
        self.d_k = out_dim // n_heads  # 每个头的维度

        # 线性投影：Q, K, V
        self.W_q = nn.Linear(in_dim, out_dim, bias=False)
        self.W_k = nn.Linear(in_dim, out_dim, bias=False)
        self.W_v = nn.Linear(in_dim, out_dim, bias=False)

        # 自适应因子的可学习权重
        self.lambda_d = nn.Parameter(torch.tensor(1.0))  # 距离衰减系数
        self.lambda_m = nn.Parameter(torch.tensor(1.0))  # 任务感知系数
        self.lambda_t = nn.Parameter(torch.tensor(0.1))  # 信息新鲜度衰减速率

        # 输出投影
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        h: torch.Tensor,           # [N, in_dim] 节点特征
        positions: torch.Tensor,    # [N, 2]      节点位置
        modes: torch.Tensor,        # [N]         节点模式（0=搜索, 1=攻击）
        last_comm_time: torch.Tensor,  # [N, N]   最后通信时间矩阵
        current_time: float,        # 当前时间步
    ) -> torch.Tensor:
        """前向传播，返回聚合后的节点特征 [N, out_dim]"""
        N = h.size(0)

        # --- 1. 计算 Q, K, V 并拆分多头 ---
        Q = self.W_q(h).view(N, self.n_heads, self.d_k)  # [N, H, d_k]
        K = self.W_k(h).view(N, self.n_heads, self.d_k)
        V = self.W_v(h).view(N, self.n_heads, self.d_k)

        # 标准缩放点积注意力分数 [N, N, H]
        attn_logits = torch.einsum("ihd,jhd->ijh", Q, K) / math.sqrt(self.d_k)

        # --- 2. 距离衰减因子 lambda_d ---
        # 计算节点间欧氏距离 [N, N]
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # [N, N, 2]
        dist = torch.norm(diff, dim=-1)  # [N, N]
        # 通信半径外的邻居直接 mask 掉
        comm_mask = (dist <= self.R_comm).float()  # [N, N]
        # 距离衰减：exp(-lambda_d * d / R_comm)
        dist_factor = torch.exp(-self.lambda_d * dist / self.R_comm) * comm_mask  # [N, N]

        # --- 3. 任务感知因子 lambda_m ---
        # 相同模式的邻居获得额外加成
        same_mode = (modes.unsqueeze(0) == modes.unsqueeze(1)).float()  # [N, N]
        mode_factor = 1.0 + self.lambda_m * same_mode  # [N, N]

        # --- 4. 信息新鲜度因子 lambda_t ---
        # 时间差越大，新鲜度越低
        time_delta = current_time - last_comm_time  # [N, N]
        time_factor = torch.exp(-self.lambda_t * time_delta)  # [N, N]

        # --- 5. 综合自适应权重 ---
        adaptive_weight = dist_factor * mode_factor * time_factor  # [N, N]
        # 扩展到多头维度 [N, N, H]
        adaptive_weight = adaptive_weight.unsqueeze(-1).expand_as(attn_logits)

        # 将自适应权重加到注意力分数上（乘性调制）
        attn_logits = attn_logits * adaptive_weight

        # 通信半径外设为 -inf
        mask_3d = comm_mask.unsqueeze(-1).expand_as(attn_logits)
        attn_logits = attn_logits.masked_fill(mask_3d == 0, float("-inf"))

        # Softmax 归一化
        attn_weights = F.softmax(attn_logits, dim=1)  # 对邻居维度归一化 [N, N, H]
        # 处理全 -inf 的行（孤立节点）
        attn_weights = attn_weights.nan_to_num(0.0)

        # --- 6. 加权聚合 ---
        # [N, N, H] x [N, H, d_k] -> [N, H, d_k]
        out = torch.einsum("ijh,jhd->ihd", attn_weights, V)
        out = out.reshape(N, self.out_dim)  # [N, out_dim]

        # 输出投影 + 残差 + LayerNorm
        out = self.out_proj(out)
        # 如果输入输出维度一致，加残差连接
        if self.in_dim == self.out_dim:
            out = self.layer_norm(out + h)
        else:
            out = self.layer_norm(out)

        return out


# ============================================================
# ObservationEncoder: 观测编码器
# ============================================================
class ObservationEncoder(nn.Module):
    """观测编码器

    两个分支：
      - CNN分支：处理 11×11×5 的局部网格观测 → 128维
      - MLP分支：处理 6维自身状态向量 → 64维
    融合后输出 128维特征

    局部网格通道含义（5通道）：
      0: 障碍物层
      1: 已探索区域
      2: 友方UAV位置
      3: 已知目标位置
      4: 威胁/敌方区域
    自身状态（6维）：
      [x, y, vx, vy, 剩余能量, 当前模式]
    """

    def __init__(self, grid_channels: int = 5, state_dim: int = 6):
        super().__init__()

        # CNN 分支：2层 5×5 卷积
        self.cnn = nn.Sequential(
            nn.Conv2d(grid_channels, 32, kernel_size=5, padding=2),  # [B, 32, 11, 11]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),             # [B, 64, 11, 11]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                                  # [B, 64, 1, 1]
            nn.Flatten(),                                             # [B, 64]
            nn.Linear(64, 128),
            nn.ReLU(),
        )

        # MLP 分支：自身状态编码
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # 融合层：128 + 64 = 192 → 128
        self.fusion = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
        )

    def forward(
        self,
        grid: torch.Tensor,   # [B, 5, 11, 11] 局部网格
        state: torch.Tensor,  # [B, 6]          自身状态
    ) -> torch.Tensor:
        """返回融合后的观测特征 [B, 128]"""
        cnn_feat = self.cnn(grid)     # [B, 128]
        mlp_feat = self.mlp(state)    # [B, 64]
        fused = torch.cat([cnn_feat, mlp_feat], dim=-1)  # [B, 192]
        return self.fusion(fused)     # [B, 128]


# ============================================================
# HierarchicalPolicy: 分层策略网络
# ============================================================
class HierarchicalPolicy(nn.Module):
    """分层策略网络

    第一层：模式选择（搜索 / 攻击）
    第二层：根据模式选择具体动作
      - 搜索模式：9个动作（8方向 + 停留）
      - 攻击模式：10个动作（8方向 + 停留 + 攻击）

    输入维度：obs_feat(128) + taga_feat(64) = 192
    """

    def __init__(self, obs_dim: int = 128, taga_dim: int = 64):
        super().__init__()
        input_dim = obs_dim + taga_dim  # 192

        # 模式选择头：搜索(0) / 攻击(1)
        self.mode_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

        # 搜索动作头：8方向 + 停留 = 9
        self.search_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 9),
        )

        # 攻击动作头：8方向 + 停留 + 攻击 = 10
        self.attack_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(
        self,
        features: torch.Tensor,                # [B, 192] 拼接后的特征
        mode_mask: torch.Tensor | None = None,  # [B, 2]  模式mask
        search_mask: torch.Tensor | None = None, # [B, 9]  搜索动作mask
        attack_mask: torch.Tensor | None = None,  # [B, 10] 攻击动作mask
    ) -> tuple[Categorical, Categorical]:
        """返回 (模式分布, 动作分布)"""
        # --- 模式选择 ---
        mode_logits = self.mode_head(features)  # [B, 2]
        if mode_mask is not None:
            mode_logits = mode_logits.masked_fill(~mode_mask.bool(), float("-inf"))
        mode_dist = Categorical(logits=mode_logits)

        # --- 动作选择（两个头都计算，后续按模式取用） ---
        search_logits = self.search_head(features)  # [B, 9]
        if search_mask is not None:
            search_logits = search_logits.masked_fill(~search_mask.bool(), float("-inf"))

        attack_logits = self.attack_head(features)  # [B, 10]
        if attack_mask is not None:
            attack_logits = attack_logits.masked_fill(~attack_mask.bool(), float("-inf"))

        return mode_dist, search_logits, attack_logits

    def get_action_dist(
        self,
        features: torch.Tensor,
        mode: torch.Tensor,  # [B] 已选定的模式
        search_mask: torch.Tensor | None = None,
        attack_mask: torch.Tensor | None = None,
    ) -> Categorical:
        """根据已选定的模式，返回对应的动作分布"""
        B = features.size(0)

        search_logits = self.search_head(features)  # [B, 9]
        attack_logits = self.attack_head(features)  # [B, 10]

        if search_mask is not None:
            search_logits = search_logits.masked_fill(~search_mask.bool(), float("-inf"))
        if attack_mask is not None:
            attack_logits = attack_logits.masked_fill(~attack_mask.bool(), float("-inf"))

        # 统一到 10 维：搜索模式补一个 -inf（第10个动作"攻击"不可用）
        search_padded = F.pad(search_logits, (0, 1), value=float("-inf"))  # [B, 10]

        # 按每个 agent 的模式选择对应的 logits
        is_attack = (mode == 1).unsqueeze(-1).expand(B, 10)  # [B, 10] bool
        combined_logits = torch.where(is_attack, attack_logits, search_padded)
        return Categorical(logits=combined_logits)


# ============================================================
# CentralizedCritic: 中心化价值网络
# ============================================================
class CentralizedCritic(nn.Module):
    """中心化价值网络

    输入所有 UAV 的 TAGA 特征拼接 [N * taga_dim]，输出标量 V 值。
    用于 MAPPO 的中心化训练。

    参数:
        n_agents:  UAV 数量
        taga_dim:  每个 UAV 的 TAGA 特征维度（默认64）
    """

    def __init__(self, n_agents: int, taga_dim: int = 64):
        super().__init__()
        input_dim = n_agents * taga_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # 标量 V 值
        )

    def forward(self, all_taga_features: torch.Tensor) -> torch.Tensor:
        """
        参数:
            all_taga_features: [B, N * taga_dim] 所有UAV的TAGA特征拼接
        返回:
            value: [B, 1] 状态价值
        """
        return self.net(all_taga_features)


# ============================================================
# TAGAMAPPO: 完整的 TAGA-MAPPO 智能体
# ============================================================
class TAGAMAPPO(nn.Module):
    """完整的 TAGA-MAPPO 智能体

    组件：
      - encoder: ObservationEncoder（CNN + MLP 融合）
      - taga_layers: 2层 TAGALayer（图注意力聚合邻居信息）
      - policy: HierarchicalPolicy（分层策略）
      - critic: CentralizedCritic（中心化价值网络）

    参数:
        n_agents:      UAV 数量
        grid_channels: 局部网格通道数（默认5）
        state_dim:     自身状态维度（默认6）
        taga_dim:      TAGA 层输出维度（默认64）
        n_heads:       注意力头数（默认4）
        R_comm:        通信半径（默认20）
    """

    def __init__(
        self,
        n_agents: int = 5,
        grid_channels: int = 5,
        state_dim: int = 6,
        taga_dim: int = 64,
        n_heads: int = 4,
        R_comm: float = 20.0,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.taga_dim = taga_dim

        # 观测编码器
        self.encoder = ObservationEncoder(grid_channels, state_dim)

        # 2层 TAGA 图注意力
        self.taga_layers = nn.ModuleList([
            TAGALayer(128, taga_dim, n_heads, R_comm),   # 第1层：128 → 64
            TAGALayer(taga_dim, taga_dim, n_heads, R_comm),  # 第2层：64 → 64
        ])

        # 分层策略网络
        self.policy = HierarchicalPolicy(obs_dim=128, taga_dim=taga_dim)

        # 中心化价值网络
        self.critic = CentralizedCritic(n_agents, taga_dim)

    def _encode_and_taga(
        self,
        grids: torch.Tensor,           # [N, 5, 11, 11]
        states: torch.Tensor,          # [N, 6]
        positions: torch.Tensor,       # [N, 2]
        modes: torch.Tensor,           # [N]
        last_comm_time: torch.Tensor,  # [N, N]
        current_time: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """编码观测并通过 TAGA 层聚合，返回 (obs_feat, taga_feat)"""
        # 编码每个 UAV 的局部观测
        obs_feat = self.encoder(grids, states)  # [N, 128]

        # 通过 TAGA 层聚合邻居信息
        h = obs_feat
        for taga_layer in self.taga_layers:
            h = taga_layer(h, positions, modes, last_comm_time, current_time)
        taga_feat = h  # [N, 64]

        return obs_feat, taga_feat

    def act(
        self,
        grids: torch.Tensor,           # [N, 5, 11, 11]
        states: torch.Tensor,          # [N, 6]
        positions: torch.Tensor,       # [N, 2]
        modes: torch.Tensor,           # [N]
        last_comm_time: torch.Tensor,  # [N, N]
        current_time: float,
        mode_mask: torch.Tensor | None = None,
        search_mask: torch.Tensor | None = None,
        attack_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> dict[str, torch.Tensor]:
        """给定观测，返回动作字典

        返回:
            dict: {
                "mode": [N],          选择的模式
                "action": [N],        选择的动作
                "mode_log_prob": [N],  模式的 log 概率
                "action_log_prob": [N], 动作的 log 概率
            }
        """
        obs_feat, taga_feat = self._encode_and_taga(
            grids, states, positions, modes, last_comm_time, current_time
        )

        # 拼接特征
        features = torch.cat([obs_feat, taga_feat], dim=-1)  # [N, 192]

        # 模式选择
        mode_dist, _, _ = self.policy(features, mode_mask, search_mask, attack_mask)
        if deterministic:
            selected_mode = mode_dist.probs.argmax(dim=-1)
        else:
            selected_mode = mode_dist.sample()

        # 动作选择
        action_dist = self.policy.get_action_dist(
            features, selected_mode, search_mask, attack_mask
        )
        if deterministic:
            selected_action = action_dist.probs.argmax(dim=-1)
        else:
            selected_action = action_dist.sample()

        return {
            "mode": selected_mode,
            "action": selected_action,
            "mode_log_prob": mode_dist.log_prob(selected_mode),
            "action_log_prob": action_dist.log_prob(selected_action),
        }

    def evaluate(
        self,
        grids: torch.Tensor,           # [N, 5, 11, 11]
        states: torch.Tensor,          # [N, 6]
        positions: torch.Tensor,       # [N, 2]
        modes: torch.Tensor,           # [N]
        last_comm_time: torch.Tensor,  # [N, N]
        current_time: float,
        selected_mode: torch.Tensor,   # [N] 已执行的模式
        selected_action: torch.Tensor, # [N] 已执行的动作
        mode_mask: torch.Tensor | None = None,
        search_mask: torch.Tensor | None = None,
        attack_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """给定观测和已执行的动作，返回评估信息

        返回:
            dict: {
                "mode_log_prob": [N],
                "action_log_prob": [N],
                "value": [N, 1],
                "mode_entropy": [N],
                "action_entropy": [N],
            }
        """
        obs_feat, taga_feat = self._encode_and_taga(
            grids, states, positions, modes, last_comm_time, current_time
        )

        features = torch.cat([obs_feat, taga_feat], dim=-1)  # [N, 192]

        # 模式分布
        mode_dist, _, _ = self.policy(features, mode_mask, search_mask, attack_mask)

        # 动作分布
        action_dist = self.policy.get_action_dist(
            features, selected_mode, search_mask, attack_mask
        )

        # 中心化价值：拼接所有 UAV 的 TAGA 特征
        all_taga = taga_feat.view(1, -1)  # [1, N*64]
        value = self.critic(all_taga)     # [1, 1]
        # 扩展到每个 agent
        value = value.expand(features.size(0), -1)  # [N, 1]

        return {
            "mode_log_prob": mode_dist.log_prob(selected_mode),
            "action_log_prob": action_dist.log_prob(selected_action),
            "value": value,
            "mode_entropy": mode_dist.entropy(),
            "action_entropy": action_dist.entropy(),
        }


# ============================================================
# 快速自测
# ============================================================
if __name__ == "__main__":
    # ---- 参数 ----
    N = 5           # UAV 数量
    device = "cpu"

    print("=" * 60)
    print("TAGA-MAPPO 模型自测")
    print("=" * 60)

    # ---- 构造虚拟输入 ----
    grids = torch.randn(N, 5, 11, 11, device=device)          # 局部网格
    states = torch.randn(N, 6, device=device)                  # 自身状态
    positions = torch.rand(N, 2, device=device) * 40           # 位置 [0, 40]
    modes = torch.randint(0, 2, (N,), device=device)           # 模式
    last_comm_time = torch.rand(N, N, device=device) * 10      # 最后通信时间
    current_time = 10.0

    # ---- 创建模型 ----
    model = TAGAMAPPO(n_agents=N).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # ---- 测试 act() ----
    with torch.no_grad():
        result = model.act(grids, states, positions, modes, last_comm_time, current_time)
    print(f"\nact() 输出:")
    print(f"  mode:            {result['mode']}")
    print(f"  action:          {result['action']}")
    print(f"  mode_log_prob:   {result['mode_log_prob']}")
    print(f"  action_log_prob: {result['action_log_prob']}")

    # ---- 测试 evaluate() ----
    eval_result = model.evaluate(
        grids, states, positions, modes, last_comm_time, current_time,
        selected_mode=result["mode"],
        selected_action=result["action"],
    )
    print(f"\nevaluate() 输出:")
    print(f"  mode_log_prob:   {eval_result['mode_log_prob']}")
    print(f"  action_log_prob: {eval_result['action_log_prob']}")
    print(f"  value:           {eval_result['value'].squeeze()}")
    print(f"  mode_entropy:    {eval_result['mode_entropy']}")
    print(f"  action_entropy:  {eval_result['action_entropy']}")

    print("\n[OK] madrl_gat.py 自测通过")
