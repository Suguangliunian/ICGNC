"""
Qwen LoRA 微调脚本（阶段B）
============================
功能：
  - 加载 Qwen2.5-4B + LoRA(r=16, alpha=32)
  - 从 JSON 文件加载训练数据
  - Loss mask：只对 [思考] 和 [A] 之后的 token 计算 loss
  - 训练日志 & 模型保存

用法：
  python train/train_qwen.py --config configs/default.yaml --data_path data/generated/train.json
"""

import os
import sys
import json
import time
import argparse
import yaml
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("[警告] 未安装 transformers/peft，请运行: pip install transformers peft accelerate")

import numpy as np


# ============================================================
# 数据集
# ============================================================
class TAGADataset(Dataset):
    """TAGA 训练数据集

    每条数据格式：
    {
        "prompt": "场景描述 + 观测信息",
        "cot": "[思考] 推理过程...",
        "action": "[A] 具体动作决策",
        "quality_score": 0.85
    }

    拼接为：{prompt}\n{cot}\n{action}
    """

    def __init__(self, data_path, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 加载数据
        print(f"[数据] 加载训练数据: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        print(f"[数据] 共 {len(self.data)} 条样本")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 拼接完整文本
        prompt = item["prompt"]
        cot = item.get("cot", "")
        action = item.get("action", "")
        full_text = f"{prompt}\n{cot}\n{action}"

        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # 计算 loss mask：只对 [思考] 和 [A] 之后的 token 计算 loss
        loss_mask = self._compute_loss_mask(prompt, full_text, input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
            "loss_mask": loss_mask,
        }

    def _compute_loss_mask(self, prompt, full_text, input_ids):
        """计算 loss mask

        只对 [思考] 和 [A] 标记之后的 token 计算 loss，
        prompt 部分的 token 不参与 loss 计算。
        """
        # 找到 prompt 结束位置
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_tokens)

        mask = torch.zeros_like(input_ids, dtype=torch.float32)

        # [思考] 和 [A] 之后的所有 token 都计算 loss
        # 简化实现：prompt 之后的所有 token 都计算 loss
        mask[prompt_len:] = 1.0

        # 对 padding 位置不计算 loss
        mask[input_ids == self.tokenizer.pad_token_id] = 0.0

        return mask


# ============================================================
# Loss Mask 工具
# ============================================================
class LossMaskHelper:
    """Loss Mask 辅助工具

    精确定位 [思考] 和 [A] 标记，只对其后续 token 计算 loss。
    """

    # 特殊标记
    COT_MARKER = "[思考]"
    ACTION_MARKER = "[A]"

    @staticmethod
    def find_marker_positions(text, tokenizer):
        """找到特殊标记在 token 序列中的位置"""
        positions = {}
        for marker in [LossMaskHelper.COT_MARKER, LossMaskHelper.ACTION_MARKER]:
            char_pos = text.find(marker)
            if char_pos >= 0:
                # 将字符位置转换为 token 位置
                prefix = text[:char_pos]
                prefix_tokens = tokenizer(prefix, add_special_tokens=False)["input_ids"]
                positions[marker] = len(prefix_tokens)
        return positions

    @staticmethod
    def create_mask(input_ids, text, tokenizer):
        """创建精确的 loss mask"""
        mask = torch.zeros_like(input_ids, dtype=torch.float32)
        positions = LossMaskHelper.find_marker_positions(text, tokenizer)

        # 对 [思考] 之后到 [A] 之前的 token 计算 loss
        cot_start = positions.get(LossMaskHelper.COT_MARKER, None)
        action_start = positions.get(LossMaskHelper.ACTION_MARKER, None)

        if cot_start is not None:
            # [思考] 之后的 token 都计算 loss
            mask[cot_start:] = 1.0

        if action_start is not None and cot_start is None:
            # 如果没有 [思考] 但有 [A]，从 [A] 开始计算
            mask[action_start:] = 1.0

        # padding 不计算
        mask[input_ids == tokenizer.pad_token_id] = 0.0
        return mask


# ============================================================
# 训练器
# ============================================================
class QwenLoRATrainer:
    """Qwen LoRA 微调训练器"""

    def __init__(self, cfg, data_path, device="auto"):
        if not HAS_TRANSFORMERS:
            raise RuntimeError("需要安装 transformers 和 peft 库")

        self.cfg = cfg
        self.qwen_cfg = cfg["qwen"]
        self.train_cfg = self.qwen_cfg["training"]
        self.data_path = data_path

        # 设备
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # 加载 tokenizer
        model_name = self.qwen_cfg["model_name"]
        print(f"[模型] 加载 tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, padding_side="right"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载基座模型
        print(f"[模型] 加载基座模型: {model_name}")
        dtype = torch.bfloat16 if self.train_cfg.get("bf16", True) else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto" if self.device == "cuda" else None,
        )

        # 启用 gradient checkpointing
        if self.train_cfg.get("gradient_checkpointing", True):
            self.model.gradient_checkpointing_enable()
            print("[模型] 已启用 gradient checkpointing")

        # 配置 LoRA
        lora_cfg = self.qwen_cfg["lora"]
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg.get("dropout", 0.05),
            target_modules=lora_cfg["target_modules"],
            bias="none",
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

        # 加载数据集
        self.dataset = TAGADataset(
            data_path, self.tokenizer,
            max_length=self.train_cfg.get("max_seq_length", 2048)
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.train_cfg["batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_cfg["lr"],
            weight_decay=self.train_cfg.get("weight_decay", 0.01),
        )

        # 学习率调度器（线性 warmup + cosine decay）
        total_steps = (len(self.dataloader) // self.train_cfg["gradient_accumulation_steps"]) * self.train_cfg["num_epochs"]
        warmup_steps = int(total_steps * self.train_cfg.get("warmup_ratio", 0.05))
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.train_cfg["lr"],
            total_steps=total_steps,
            pct_start=warmup_steps / max(total_steps, 1),
            anneal_strategy="cos",
        )

    def train(self):
        """主训练循环"""
        cfg = self.train_cfg
        grad_accum = cfg["gradient_accumulation_steps"]
        num_epochs = cfg["num_epochs"]
        logging_steps = cfg.get("logging_steps", 10)
        save_steps = cfg.get("save_steps", 200)
        save_dir = os.path.join(self.cfg["paths"]["checkpoint_dir"], "qwen_lora")
        os.makedirs(save_dir, exist_ok=True)

        print("=" * 60)
        print("Qwen LoRA 微调开始")
        print(f"  基座模型: {self.qwen_cfg['model_name']}")
        print(f"  LoRA r={self.qwen_cfg['lora']['r']}, alpha={self.qwen_cfg['lora']['alpha']}")
        print(f"  Batch size: {cfg['batch_size']} × {grad_accum} = {cfg['batch_size'] * grad_accum}")
        print(f"  Epochs: {num_epochs}")
        print(f"  学习率: {cfg['lr']}")
        print(f"  数据量: {len(self.dataset)} 条")
        print("=" * 60)

        self.model.train()
        start_time = time.time()
        global_step = 0
        total_loss = 0
        log_loss = 0

        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
            epoch_loss = 0
            n_batches = 0

            for batch_idx, batch in enumerate(self.dataloader):
                # 移动到设备
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                loss_mask = batch["loss_mask"].to(self.device)

                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                # 计算带 mask 的 loss
                logits = outputs.logits[:, :-1, :]  # 去掉最后一个位置
                targets = labels[:, 1:]              # 去掉第一个位置
                mask = loss_mask[:, 1:]              # 对齐 mask

                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                token_loss = loss_fct(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                ).reshape(targets.shape)

                # 应用 loss mask
                masked_loss = (token_loss * mask).sum() / (mask.sum() + 1e-8)
                loss = masked_loss / grad_accum

                loss.backward()

                if (batch_idx + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    step_loss = masked_loss.item()
                    total_loss += step_loss
                    log_loss += step_loss

                    # 日志
                    if global_step % logging_steps == 0:
                        avg_log_loss = log_loss / logging_steps
                        elapsed = time.time() - start_time
                        lr = self.scheduler.get_last_lr()[0]
                        print(f"  [步骤 {global_step:5d}] "
                              f"loss={avg_log_loss:.4f} | "
                              f"lr={lr:.2e} | "
                              f"耗时={elapsed:.0f}s")
                        log_loss = 0

                    # 保存
                    if global_step % save_steps == 0:
                        ckpt_path = os.path.join(save_dir, f"step_{global_step}")
                        self.model.save_pretrained(ckpt_path)
                        self.tokenizer.save_pretrained(ckpt_path)
                        print(f"  [保存] 检查点: {ckpt_path}")

                epoch_loss += masked_loss.item()
                n_batches += 1

            avg_epoch_loss = epoch_loss / max(n_batches, 1)
            print(f"  Epoch {epoch + 1} 平均 loss: {avg_epoch_loss:.4f}")

        # 保存最终模型
        final_path = os.path.join(save_dir, "final")
        self.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)

        total_time = time.time() - start_time
        avg_loss = total_loss / max(global_step, 1)
        print(f"\n微调完成！")
        print(f"  总步数: {global_step}")
        print(f"  平均 loss: {avg_loss:.4f}")
        print(f"  总耗时: {total_time / 60:.1f} 分钟")
        print(f"  模型保存: {final_path}")


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Qwen LoRA 微调")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--data_path", type=str, default="data/generated/train.json", help="训练数据路径")
    parser.add_argument("--device", type=str, default="auto", help="设备 (cpu/cuda/auto)")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    args = parser.parse_args()

    # 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 设置随机种子
    seed = args.seed or cfg.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"随机种子: {seed}")

    # 创建输出目录
    os.makedirs(cfg["paths"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["log_dir"], exist_ok=True)

    # 开始训练
    trainer = QwenLoRATrainer(cfg, args.data_path, device=args.device)
    trainer.train()


if __name__ == "__main__":
    main()
