"""
utils.py — UAV集群仿真辅助函数
包含：距离计算、方向映射、信号质量、网格可视化
"""

import numpy as np
from typing import Tuple, Optional, List

# ============================================================
# 1. 距离计算
# ============================================================

def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """计算两点之间的欧氏距离
    Args:
        p1: 点1坐标 (x, y)
        p2: 点2坐标 (x, y)
    Returns:
        欧氏距离（浮点数）
    """
    return float(np.linalg.norm(np.asarray(p1, dtype=np.float64)
                                - np.asarray(p2, dtype=np.float64)))


def manhattan_distance(p1: np.ndarray, p2: np.ndarray) -> int:
    """计算两点之间的曼哈顿距离"""
    d = np.abs(np.asarray(p1, dtype=np.int32) - np.asarray(p2, dtype=np.int32))
    return int(d.sum())


def chebyshev_distance(p1: np.ndarray, p2: np.ndarray) -> int:
    """计算两点之间的切比雪夫距离（棋盘距离）"""
    d = np.abs(np.asarray(p1, dtype=np.int32) - np.asarray(p2, dtype=np.int32))
    return int(d.max())


def pairwise_distances(positions: np.ndarray) -> np.ndarray:
    """计算所有智能体之间的欧氏距离矩阵
    Args:
        positions: (N, 2) 坐标数组
    Returns:
        (N, N) 距离矩阵
    """
    diff = positions[:, None, :] - positions[None, :, :]  # (N, N, 2)
    return np.sqrt((diff ** 2).sum(axis=-1))               # (N, N)


# ============================================================
# 2. 方向映射（8方向 + 停留 = 9种搜索动作）
# ============================================================

# 动作索引 → (dx, dy) 偏移
# 0=停留, 1=上, 2=下, 3=左, 4=右, 5=左上, 6=右上, 7=左下, 8=右下
DIRECTION_MAP = {
    0: (0, 0),    # 停留
    1: (0, -1),   # 上（y减小）
    2: (0, 1),    # 下（y增大）
    3: (-1, 0),   # 左
    4: (1, 0),    # 右
    5: (-1, -1),  # 左上
    6: (1, -1),   # 右上
    7: (-1, 1),   # 左下
    8: (1, 1),    # 右下
}

DIRECTION_NAMES = {
    0: "停留", 1: "上", 2: "下", 3: "左", 4: "右",
    5: "左上", 6: "右上", 7: "左下", 8: "右下",
}


def direction_to_offset(action: int) -> Tuple[int, int]:
    """将方向动作索引转换为 (dx, dy) 偏移"""
    return DIRECTION_MAP.get(action, (0, 0))


def offset_to_direction(dx: int, dy: int) -> int:
    """将 (dx, dy) 偏移转换为方向动作索引，找不到则返回0（停留）"""
    inv = {v: k for k, v in DIRECTION_MAP.items()}
    return inv.get((dx, dy), 0)


def angle_to_direction(angle_rad: float) -> int:
    """将弧度角转换为最近的8方向索引（0=停留不参与）
    角度定义：0=右, π/2=下, π=左, 3π/2=上
    """
    # 归一化到 [0, 2π)
    angle = angle_rad % (2 * np.pi)
    # 8方向对应角度（从右开始顺时针）
    dirs = [4, 8, 2, 7, 3, 5, 1, 6]  # 右,右下,下,左下,左,左上,上,右上
    idx = int(round(angle / (np.pi / 4))) % 8
    return dirs[idx]


# ============================================================
# 3. 信号质量计算
# ============================================================

def signal_quality(distance: float, r_comm: float = 20.0,
                   alpha: float = 2.0) -> float:
    """基于距离的通信信号质量（0~1）
    使用简化的路径损耗模型：q = max(0, 1 - (d / R_comm)^alpha)
    Args:
        distance: 两节点间距离
        r_comm:   通信半径
        alpha:    衰减指数（默认2.0，平方衰减）
    Returns:
        信号质量 [0, 1]
    """
    if distance <= 0:
        return 1.0
    ratio = distance / r_comm
    if ratio >= 1.0:
        return 0.0
    return float(max(0.0, 1.0 - ratio ** alpha))


def signal_quality_batch(distances: np.ndarray, r_comm: float = 20.0,
                         alpha: float = 2.0) -> np.ndarray:
    """批量计算信号质量
    Args:
        distances: 任意形状的距离数组
    Returns:
        同形状的信号质量数组
    """
    ratio = np.clip(distances / r_comm, 0.0, None)
    q = np.maximum(0.0, 1.0 - ratio ** alpha)
    q[distances <= 0] = 1.0
    return q


# ============================================================
# 4. 网格可视化辅助
# ============================================================

def render_grid_text(grid_size: int,
                     uav_positions: np.ndarray,
                     target_positions: np.ndarray,
                     target_alive: np.ndarray,
                     threat_centers: np.ndarray,
                     threat_radii: np.ndarray,
                     visited: Optional[np.ndarray] = None) -> str:
    """生成文本形式的网格可视化（用于终端调试）
    符号说明：
        U = UAV, T = 存活目标, X = 已摧毁目标, ! = 威胁区域中心
        . = 未访问, ~ = 已访问
    仅渲染关键元素，大网格时缩放到最大50×50显示
    """
    # 确定显示尺寸
    max_display = 50
    scale = max(1, grid_size // max_display)
    display_size = grid_size // scale

    canvas = np.full((display_size, display_size), '.', dtype='<U1')

    # 标记已访问区域
    if visited is not None:
        for y in range(display_size):
            for x in range(display_size):
                # 检查缩放块内是否有已访问格子
                block = visited[y*scale:(y+1)*scale, x*scale:(x+1)*scale]
                if block.any():
                    canvas[y, x] = '~'

    # 标记威胁区域中心
    for i in range(len(threat_centers)):
        tx, ty = int(threat_centers[i][0]) // scale, int(threat_centers[i][1]) // scale
        if 0 <= tx < display_size and 0 <= ty < display_size:
            canvas[ty, tx] = '!'

    # 标记目标
    for i in range(len(target_positions)):
        tx = int(target_positions[i][0]) // scale
        ty = int(target_positions[i][1]) // scale
        if 0 <= tx < display_size and 0 <= ty < display_size:
            canvas[ty, tx] = 'T' if target_alive[i] else 'X'

    # 标记UAV
    for i in range(len(uav_positions)):
        ux = int(uav_positions[i][0]) // scale
        uy = int(uav_positions[i][1]) // scale
        if 0 <= ux < display_size and 0 <= uy < display_size:
            canvas[uy, ux] = 'U'

    # 拼接字符串
    lines = []
    header = "  " + "".join(f"{j % 10}" for j in range(display_size))
    lines.append(header)
    for y in range(display_size):
        row = f"{y % 100:2d}" + "".join(canvas[y])
        lines.append(row)
    return "\n".join(lines)


def create_matplotlib_frame(grid_size: int,
                            uav_positions: np.ndarray,
                            target_positions: np.ndarray,
                            target_alive: np.ndarray,
                            threat_centers: np.ndarray,
                            threat_radii: np.ndarray,
                            visited: Optional[np.ndarray] = None,
                            comm_adj: Optional[np.ndarray] = None):
    """生成 matplotlib 图像帧（用于可视化/录制）
    返回 matplotlib figure 对象，调用方可 savefig 或转为数组
    注意：需要安装 matplotlib，仅在调用时导入
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        raise ImportError("可视化需要安装 matplotlib: pip install matplotlib")

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title("UAV集群搜索-攻击仿真")

    # 已访问区域（浅绿色背景）
    if visited is not None:
        ax.imshow(visited.T, origin='upper', extent=(-0.5, grid_size-0.5, grid_size-0.5, -0.5),
                  cmap='Greens', alpha=0.3, vmin=0, vmax=1)

    # 威胁区域（红色半透明圆）
    for i in range(len(threat_centers)):
        circle = patches.Circle(threat_centers[i], threat_radii[i],
                                color='red', alpha=0.15, linewidth=1,
                                edgecolor='red', linestyle='--')
        ax.add_patch(circle)

    # 目标
    alive_mask = target_alive.astype(bool)
    if alive_mask.any():
        ax.scatter(target_positions[alive_mask, 0], target_positions[alive_mask, 1],
                   marker='*', s=200, c='orange', edgecolors='black', zorder=5,
                   label='存活目标')
    dead_mask = ~alive_mask
    if dead_mask.any():
        ax.scatter(target_positions[dead_mask, 0], target_positions[dead_mask, 1],
                   marker='x', s=100, c='gray', zorder=4, label='已摧毁目标')

    # UAV
    ax.scatter(uav_positions[:, 0], uav_positions[:, 1],
               marker='^', s=120, c='blue', edgecolors='black', zorder=6,
               label='UAV')

    # 通信链路
    if comm_adj is not None:
        n = len(uav_positions)
        for i in range(n):
            for j in range(i + 1, n):
                if comm_adj[i, j] > 0:
                    ax.plot([uav_positions[i, 0], uav_positions[j, 0]],
                            [uav_positions[i, 1], uav_positions[j, 1]],
                            'b-', alpha=0.2, linewidth=0.5)

    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig


# ============================================================
# 5. 其他工具函数
# ============================================================

def clip_position(pos: np.ndarray, grid_size: int) -> np.ndarray:
    """将位置裁剪到网格范围 [0, grid_size-1]"""
    return np.clip(pos, 0, grid_size - 1)


def in_threat_zone(pos: np.ndarray, threat_centers: np.ndarray,
                   threat_radii: np.ndarray) -> bool:
    """判断某位置是否在任意威胁区域内"""
    for i in range(len(threat_centers)):
        if euclidean_distance(pos, threat_centers[i]) <= threat_radii[i]:
            return True
    return False


def in_threat_zone_batch(positions: np.ndarray, threat_centers: np.ndarray,
                         threat_radii: np.ndarray) -> np.ndarray:
    """批量判断多个位置是否在威胁区域内
    Args:
        positions:      (N, 2)
        threat_centers: (K, 2)
        threat_radii:   (K,)
    Returns:
        (N,) bool数组
    """
    # (N, K, 2)
    diff = positions[:, None, :] - threat_centers[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))  # (N, K)
    return (dists <= threat_radii[None, :]).any(axis=1)  # (N,)
