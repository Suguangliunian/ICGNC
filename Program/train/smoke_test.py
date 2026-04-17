"""极简 smoke test：逐步验证每个组件是否正常工作"""
import os, sys, time

_PROG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROG_ROOT)
os.chdir(_PROG_ROOT)

import numpy as np
import torch
import yaml

print("[1/6] 导入模块... ", end="", flush=True)
from train.train_madrl import BattlefieldEnv, CurriculumScheduler, MAPPOTrainer, RolloutBuffer
print("OK")

print("[2/6] 加载配置... ", end="", flush=True)
_cfg_path = os.path.join(_PROG_ROOT, "configs", "default.yaml")
with open(_cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print("OK")

# 用极小参数
cfg["mappo"]["n_parallel_envs"] = 2
cfg["mappo"]["rollout_steps"] = 16
cfg["mappo"]["ppo_epochs"] = 2
cfg["mappo"]["mini_batch_size"] = 32

print("[3/6] 创建环境 (stage_1: 3 UAV, 30x30)... ", end="", flush=True)
t0 = time.time()
env_cfg = {**cfg["env"], "n_uavs": 3, "n_targets": 2, "n_threats": 0, "grid_size": 30}
env = BattlefieldEnv(env_cfg, cfg["reward"], n_envs=2)
obs = env.reset()
print(f"OK ({time.time()-t0:.2f}s)")

print("[4/6] 环境 step x 16... ", flush=True)
t0 = time.time()
for step in range(16):
    mode_act = np.random.randint(0, 2, (2, 3))
    move_act = np.random.randint(0, 10, (2, 3))
    obs, rewards, dones, infos = env.step(mode_act, move_act)
    if step % 4 == 0:
        print(f"  step {step}: cov={infos[0]['coverage']:.3f} reward_mean={rewards.mean():.3f}", flush=True)
    if dones.any():
        for e in range(2):
            if dones[e]:
                env.reset_single(e)
        obs = env._get_obs()
print(f"  环境 step 完成 ({time.time()-t0:.2f}s)")

print("[5/6] 创建 Trainer... ", end="", flush=True)
t0 = time.time()
np.random.seed(42)
torch.manual_seed(42)
trainer = MAPPOTrainer(cfg, device="cpu")
print(f"OK ({time.time()-t0:.2f}s)")

print("[6/6] 跑 1 个 update (rollout + GAE + PPO)... ", flush=True)
t0 = time.time()
obs = trainer.env.reset()
buffer = RolloutBuffer(
    trainer.n_envs,
    trainer.n_agents,
    16,
    trainer.device,
    obs_patch_size=trainer.env.obs_patch_size,
)

for step in range(16):
    mode_act, move_act, mode_lp, move_lp, values, feats = trainer.select_actions(obs, trainer.n_agents)
    next_obs, rewards, dones, infos = trainer.env.step(mode_act, move_act)
    flat_r = rewards.reshape(-1)
    norm_r = trainer.reward_normalizer.normalize(flat_r).reshape(rewards.shape)
    buffer.add(obs, mode_act, move_act, mode_lp, move_lp, norm_r, dones, values, feats)
    for e in range(trainer.n_envs):
        if dones[e]:
            trainer.env.reset_single(e)
    obs = trainer.env._get_obs()
    if step % 4 == 0:
        print(f"  rollout step {step} OK", flush=True)

print("  计算 GAE...", flush=True)
with torch.no_grad():
    feats_last = trainer._get_features(obs, trainer.n_agents)
    last_values = trainer.critic(feats_last).squeeze(-1)
    last_values = last_values.cpu().numpy().reshape(trainer.n_envs, trainer.n_agents)
buffer.compute_gae(last_values, cfg["mappo"]["gamma"], cfg["mappo"]["lambda_gae"])

print("  PPO 更新...", flush=True)
losses = trainer.ppo_update(buffer)
print(f"  losses: policy={losses['policy_loss']:.4f} value={losses['value_loss']:.4f} entropy={losses['entropy']:.4f}")
print(f"  1 update 完成 ({time.time()-t0:.2f}s)")

print("\n=== SMOKE TEST PASSED ===")
