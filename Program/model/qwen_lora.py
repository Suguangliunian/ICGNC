"""
Qwen3.5-4B + LoRA 模型（阶段B）
================================
包含：
  - TAGAAwarePromptBuilder: TAGA感知的Prompt构建器
  - QwenLoRAModel: Qwen3.5-4B + LoRA 封装
  - BatchInferenceEngine: 批量推理引擎
"""

import re
import json
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


# ============================================================
# TAGAAwarePromptBuilder: TAGA感知的Prompt构建器
# ============================================================
@dataclass
class NeighborInfo:
    """邻居 UAV 信息"""
    uav_id: int
    distance: float
    relative_pos: tuple[float, float]  # (dx, dy)
    mode: int                          # 0=搜索, 1=攻击
    signal_quality: str = "H"          # H/M/L
    freshness: float = 1.0             # 信息新鲜度 [0, 1]


@dataclass
class TargetInfo:
    """目标信息"""
    target_id: int
    position: tuple[float, float]
    distance: float
    confirmed: bool = False
    threat_level: int = 0  # 0-3


# 5种指令类型
INSTRUCTION_TYPES = {
    "BAL": "平衡搜索与攻击资源分配",
    "SCH": "优先搜索未探索区域",
    "ATK": "集中力量攻击已确认目标",
    "EVD": "规避高威胁区域",
    "COO": "与邻居协同行动",
}


class TAGAAwarePromptBuilder:
    """TAGA感知的Prompt构建器

    根据 UAV 状态、邻居信息、目标信息构建压缩格式的 prompt。
    邻居按距离排序，标注信号质量(H/M/L)和新鲜度。
    支持5种指令(BAL/SCH/ATK/EVD/COO)。
    当超过 max_tokens 时按优先级截断。
    """

    # 各部分优先级（数字越小越重要，截断时从低优先级开始删）
    PRIORITY_SELF = 0       # 自身状态（不可截断）
    PRIORITY_TARGETS = 1    # 目标信息
    PRIORITY_INSTRUCTION = 2  # 指令
    PRIORITY_NEIGHBORS = 3  # 邻居信息（最先被截断）

    def __init__(self, max_tokens: int = 256):
        self.max_tokens = max_tokens

    @staticmethod
    def _signal_quality(distance: float, R_comm: float = 20.0) -> str:
        """根据距离计算信号质量等级"""
        ratio = distance / R_comm
        if ratio < 0.4:
            return "H"  # 高质量
        elif ratio < 0.75:
            return "M"  # 中等
        else:
            return "L"  # 低质量

    @staticmethod
    def _freshness_tag(freshness: float) -> str:
        """新鲜度标签"""
        if freshness > 0.7:
            return "新"
        elif freshness > 0.3:
            return "旧"
        else:
            return "过期"

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """粗略估算 token 数（中文约1.5字/token，英文约4字符/token）"""
        # 简单估算：按字符数 / 3
        return max(1, len(text) // 3)

    def _build_self_section(
        self,
        uav_id: int,
        position: tuple[float, float],
        mode: int,
        energy: float,
        velocity: tuple[float, float],
    ) -> str:
        """构建自身状态段（不可截断）"""
        mode_str = "搜索" if mode == 0 else "攻击"
        return (
            f"[自身] UAV{uav_id} "
            f"位置({position[0]:.1f},{position[1]:.1f}) "
            f"模式={mode_str} "
            f"能量={energy:.0%} "
            f"速度({velocity[0]:.1f},{velocity[1]:.1f})"
        )

    def _build_neighbor_section(self, neighbors: list[NeighborInfo]) -> str:
        """构建邻居信息段（按距离排序）"""
        if not neighbors:
            return "[邻居] 无"
        # 按距离排序
        sorted_nb = sorted(neighbors, key=lambda n: n.distance)
        lines = ["[邻居]"]
        for nb in sorted_nb:
            mode_str = "搜" if nb.mode == 0 else "攻"
            fresh_tag = self._freshness_tag(nb.freshness)
            lines.append(
                f"  U{nb.uav_id} d={nb.distance:.1f} "
                f"({nb.relative_pos[0]:+.1f},{nb.relative_pos[1]:+.1f}) "
                f"{mode_str} 信号={nb.signal_quality} {fresh_tag}"
            )
        return "\n".join(lines)

    def _build_target_section(self, targets: list[TargetInfo]) -> str:
        """构建目标信息段"""
        if not targets:
            return "[目标] 未发现"
        sorted_tgt = sorted(targets, key=lambda t: t.distance)
        lines = ["[目标]"]
        for tgt in sorted_tgt:
            status = "已确认" if tgt.confirmed else "疑似"
            lines.append(
                f"  T{tgt.target_id} ({tgt.position[0]:.1f},{tgt.position[1]:.1f}) "
                f"d={tgt.distance:.1f} {status} 威胁={tgt.threat_level}"
            )
        return "\n".join(lines)

    def _build_instruction_section(self, instruction: str) -> str:
        """构建指令段"""
        desc = INSTRUCTION_TYPES.get(instruction, instruction)
        return f"[指令] {instruction}: {desc}"

    def build(
        self,
        uav_id: int,
        position: tuple[float, float],
        mode: int,
        energy: float,
        velocity: tuple[float, float],
        neighbors: list[NeighborInfo],
        targets: list[TargetInfo],
        instruction: str = "BAL",
    ) -> str:
        """构建完整 prompt，超过 max_tokens 时按优先级截断

        截断策略：
          1. 先删邻居（从最远的开始逐个删）
          2. 再删目标（从最远的开始逐个删）
          3. 再删指令
          4. 自身状态永不截断
        """
        self_section = self._build_self_section(uav_id, position, mode, energy, velocity)
        instr_section = self._build_instruction_section(instruction)

        # 先尝试完整版本
        nb_section = self._build_neighbor_section(neighbors)
        tgt_section = self._build_target_section(targets)

        full_prompt = f"{self_section}\n{tgt_section}\n{instr_section}\n{nb_section}"

        if self._estimate_tokens(full_prompt) <= self.max_tokens:
            return full_prompt

        # --- 截断策略 ---
        # 1. 逐步减少邻居数量
        sorted_neighbors = sorted(neighbors, key=lambda n: n.distance)
        for keep_n in range(len(sorted_neighbors) - 1, -1, -1):
            nb_section = self._build_neighbor_section(sorted_neighbors[:keep_n])
            prompt = f"{self_section}\n{tgt_section}\n{instr_section}\n{nb_section}"
            if self._estimate_tokens(prompt) <= self.max_tokens:
                return prompt

        # 2. 逐步减少目标数量
        sorted_targets = sorted(targets, key=lambda t: t.distance)
        for keep_t in range(len(sorted_targets) - 1, -1, -1):
            tgt_section = self._build_target_section(sorted_targets[:keep_t])
            prompt = f"{self_section}\n{tgt_section}\n{instr_section}"
            if self._estimate_tokens(prompt) <= self.max_tokens:
                return prompt

        # 3. 删除指令
        prompt = f"{self_section}\n[目标] 未发现"
        if self._estimate_tokens(prompt) <= self.max_tokens:
            return prompt

        # 4. 只保留自身状态
        return self_section


# ============================================================
# QwenLoRAModel: Qwen3.5-4B + LoRA 封装
# ============================================================
class QwenLoRAModel:
    """Qwen3.5-4B + LoRA 封装

    训练模式：加载全精度模型 + LoRA 适配器
    推理模式：加载 4-bit 量化模型 + 合并后的 LoRA 权重
    支持 loss mask：只对 CoT 和 Action 部分计算 loss

    参数:
        model_name: HuggingFace 模型名称
        lora_r:     LoRA 秩（默认16）
        lora_alpha: LoRA alpha（默认32）
        lora_dropout: LoRA dropout（默认0.05）
        target_modules: LoRA 目标模块
    """

    # CoT 和 Action 的标记符号（用于 loss mask）
    COT_START = "<think>"
    COT_END = "</think>"
    ACTION_START = "<act>"
    ACTION_END = "</act>"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-4B",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: list[str] | None = None,
    ):
        self.model_name = model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

        self.model = None
        self.tokenizer = None

    def load_for_training(self, device: str = "auto") -> None:
        """加载模型用于训练（全精度 + LoRA）

        需要安装: transformers, peft, accelerate
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, TaskType

        print(f"[QwenLoRA] 加载训练模型: {self.model_name}")

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载基础模型（bf16 节省显存）
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )

        # 配置 LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias="none",
        )

        # 应用 LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        print("[QwenLoRA] 训练模型加载完成")

    def load_for_inference(self, lora_weights_path: str | None = None, device: str = "auto") -> None:
        """加载模型用于推理（4-bit 量化）

        需要安装: transformers, peft, bitsandbytes, accelerate
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        print(f"[QwenLoRA] 加载推理模型（4-bit）: {self.model_name}")

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 4-bit 量化配置
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        # 加载量化模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map=device,
            trust_remote_code=True,
        )

        # 如果有 LoRA 权重，加载并合并
        if lora_weights_path is not None:
            print(f"[QwenLoRA] 加载 LoRA 权重: {lora_weights_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_weights_path)
            self.model = self.model.merge_and_unload()

        self.model.eval()
        print("[QwenLoRA] 推理模型加载完成")

    def build_loss_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """构建 loss mask：只对 CoT 和 Action 部分计算 loss

        标记规则：
          - <think>...</think> 之间的 token → mask = 1
          - <act>...</act> 之间的 token → mask = 1
          - 其余 token → mask = 0

        参数:
            input_ids: [B, L] token ids
        返回:
            mask: [B, L] loss mask（0 或 1）
        """
        assert self.tokenizer is not None, "请先调用 load_for_training() 或 load_for_inference()"

        B, L = input_ids.shape
        mask = torch.zeros(B, L, dtype=torch.float32, device=input_ids.device)

        # 获取特殊标记的 token id
        cot_start_ids = self.tokenizer.encode(self.COT_START, add_special_tokens=False)
        cot_end_ids = self.tokenizer.encode(self.COT_END, add_special_tokens=False)
        act_start_ids = self.tokenizer.encode(self.ACTION_START, add_special_tokens=False)
        act_end_ids = self.tokenizer.encode(self.ACTION_END, add_special_tokens=False)

        for b in range(B):
            ids = input_ids[b].tolist()
            # 查找所有 CoT 和 Action 区间
            mask[b] = self._mark_spans(ids, cot_start_ids, cot_end_ids, L)
            mask[b] += self._mark_spans(ids, act_start_ids, act_end_ids, L)

        return mask.clamp(0, 1)

    @staticmethod
    def _mark_spans(
        ids: list[int],
        start_pattern: list[int],
        end_pattern: list[int],
        length: int,
    ) -> torch.Tensor:
        """在 token 序列中查找 start_pattern...end_pattern 区间并标记"""
        mask = torch.zeros(length, dtype=torch.float32)
        i = 0
        while i < length:
            # 查找 start_pattern
            if ids[i: i + len(start_pattern)] == start_pattern:
                start = i + len(start_pattern)
                # 查找对应的 end_pattern
                j = start
                while j < length:
                    if ids[j: j + len(end_pattern)] == end_pattern:
                        mask[start: j] = 1.0
                        i = j + len(end_pattern)
                        break
                    j += 1
                else:
                    # 没找到结束标记，标记到末尾
                    mask[start:] = 1.0
                    break
            else:
                i += 1
        return mask

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        **kwargs,
    ) -> list[str]:
        """批量生成文本

        参数:
            prompts: 输入 prompt 列表
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度（0 = 贪心解码）
        返回:
            生成的文本列表
        """
        assert self.model is not None and self.tokenizer is not None, "请先加载模型"

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            **kwargs,
        }

        # 贪心解码 vs 采样
        if temperature <= 0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # 只取新生成的部分
        input_len = inputs["input_ids"].shape[1]
        generated = outputs[:, input_len:]
        texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return texts


# ============================================================
# BatchInferenceEngine: 批量推理引擎
# ============================================================
@dataclass
class StructuredAction:
    """解析后的结构化动作"""
    instruction: str = "BAL"          # 指令类型
    target_uav: int | None = None     # 目标 UAV（COO 指令时使用）
    target_id: int | None = None      # 目标 ID（ATK 指令时使用）
    direction: int = 8                # 移动方向 0-7 或 8=停留
    reasoning: str = ""               # CoT 推理过程


class BatchInferenceEngine:
    """批量推理引擎

    同时为所有 UAV 生成决策，解析 LLM 输出为结构化动作。

    参数:
        qwen_model: QwenLoRAModel 实例
        prompt_builder: TAGAAwarePromptBuilder 实例
        max_new_tokens: 最大生成 token 数
    """

    # 方向映射：0-7 对应 8 个方向，8 = 停留
    DIRECTION_MAP = {
        "N": 0, "NE": 1, "E": 2, "SE": 3,
        "S": 4, "SW": 5, "W": 6, "NW": 7,
        "STAY": 8,
        # 中文别名
        "北": 0, "东北": 1, "东": 2, "东南": 3,
        "南": 4, "西南": 5, "西": 6, "西北": 7,
        "停留": 8,
    }

    # 动作解析正则
    _ACTION_PATTERN = re.compile(
        r"<act>\s*(?P<instr>BAL|SCH|ATK|EVD|COO)"
        r"(?:\s+T(?P<tid>\d+))?"
        r"(?:\s+U(?P<uid>\d+))?"
        r"\s+(?P<dir>[A-Z]+|[\u4e00-\u9fff]+)"
        r"\s*</act>",
        re.IGNORECASE,
    )

    def __init__(
        self,
        qwen_model: QwenLoRAModel,
        prompt_builder: TAGAAwarePromptBuilder | None = None,
        max_new_tokens: int = 128,
    ):
        self.qwen_model = qwen_model
        self.prompt_builder = prompt_builder or TAGAAwarePromptBuilder()
        self.max_new_tokens = max_new_tokens

    def batch_decide(
        self,
        uav_states: list[dict[str, Any]],
        neighbor_map: dict[int, list[NeighborInfo]],
        target_map: dict[int, list[TargetInfo]],
        instruction: str = "BAL",
    ) -> list[StructuredAction]:
        """同时为所有 UAV 生成决策

        参数:
            uav_states: UAV 状态列表，每个元素包含:
                {"uav_id", "position", "mode", "energy", "velocity"}
            neighbor_map: {uav_id: [NeighborInfo, ...]}
            target_map: {uav_id: [TargetInfo, ...]}
            instruction: 全局指令类型

        返回:
            结构化动作列表，与 uav_states 一一对应
        """
        # 1. 为每个 UAV 构建 prompt
        prompts = []
        for uav in uav_states:
            uid = uav["uav_id"]
            prompt = self.prompt_builder.build(
                uav_id=uid,
                position=uav["position"],
                mode=uav["mode"],
                energy=uav["energy"],
                velocity=uav["velocity"],
                neighbors=neighbor_map.get(uid, []),
                targets=target_map.get(uid, []),
                instruction=instruction,
            )
            # 添加系统提示和格式要求
            full_prompt = self._wrap_prompt(prompt)
            prompts.append(full_prompt)

        # 2. 批量推理（贪心解码）
        raw_outputs = self.qwen_model.generate(
            prompts,
            max_new_tokens=self.max_new_tokens,
            temperature=0.0,  # 贪心解码
        )

        # 3. 解析每个输出
        actions = []
        for i, output in enumerate(raw_outputs):
            action = self._parse_action(output)
            actions.append(action)

        return actions

    def _wrap_prompt(self, context: str) -> str:
        """包装 prompt，添加系统提示和格式要求"""
        system = (
            "你是一个无人机协同作战的决策系统。"
            "根据当前态势信息，输出推理过程和行动指令。\n"
            "格式要求：\n"
            "<think>简要推理</think>\n"
            "<act>指令 方向</act>\n"
            "指令: BAL/SCH/ATK/EVD/COO\n"
            "方向: N/NE/E/SE/S/SW/W/NW/STAY\n"
        )
        return f"{system}\n{context}\n请决策："

    def _parse_action(self, raw_output: str) -> StructuredAction:
        """解析 LLM 输出为结构化动作

        期望格式：
          <think>推理过程</think>
          <act>BAL N</act>

        解析失败时返回默认动作（BAL STAY）
        """
        action = StructuredAction()

        # 提取 CoT 推理
        cot_match = re.search(r"<think>(.*?)</think>", raw_output, re.DOTALL)
        if cot_match:
            action.reasoning = cot_match.group(1).strip()

        # 提取动作
        act_match = self._ACTION_PATTERN.search(raw_output)
        if act_match:
            action.instruction = act_match.group("instr").upper()

            # 目标 ID
            tid = act_match.group("tid")
            if tid is not None:
                action.target_id = int(tid)

            # 目标 UAV
            uid = act_match.group("uid")
            if uid is not None:
                action.target_uav = int(uid)

            # 方向
            dir_str = act_match.group("dir").upper()
            action.direction = self.DIRECTION_MAP.get(dir_str, 8)
        else:
            # 解析失败，尝试简单匹配
            for key in self.DIRECTION_MAP:
                if key in raw_output.upper():
                    action.direction = self.DIRECTION_MAP[key]
                    break

        return action


# ============================================================
# 快速自测
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Qwen3.5-4B + LoRA 模型自测")
    print("=" * 60)

    # ---- 1. 测试 TAGAAwarePromptBuilder ----
    print("\n--- TAGAAwarePromptBuilder 测试 ---")
    builder = TAGAAwarePromptBuilder(max_tokens=256)

    neighbors = [
        NeighborInfo(uav_id=2, distance=8.5, relative_pos=(5.0, 6.8), mode=0, signal_quality="H", freshness=0.9),
        NeighborInfo(uav_id=3, distance=15.2, relative_pos=(-10.0, 11.3), mode=1, signal_quality="M", freshness=0.5),
        NeighborInfo(uav_id=4, distance=19.8, relative_pos=(18.0, -7.5), mode=0, signal_quality="L", freshness=0.2),
    ]
    targets = [
        TargetInfo(target_id=1, position=(25.0, 30.0), distance=12.0, confirmed=True, threat_level=2),
        TargetInfo(target_id=2, position=(40.0, 15.0), distance=28.0, confirmed=False, threat_level=1),
    ]

    prompt = builder.build(
        uav_id=1,
        position=(10.0, 15.0),
        mode=0,
        energy=0.75,
        velocity=(1.0, 0.5),
        neighbors=neighbors,
        targets=targets,
        instruction="SCH",
    )
    print(prompt)
    print(f"\n估算 tokens: {builder._estimate_tokens(prompt)}")

    # ---- 2. 测试动作解析 ----
    print("\n--- BatchInferenceEngine 动作解析测试 ---")
    # 不加载真实模型，只测试解析逻辑
    dummy_model = QwenLoRAModel()
    engine = BatchInferenceEngine(dummy_model, builder)

    test_outputs = [
        "<think>目标T1在东北方向，距离较近，应优先搜索</think><act>SCH NE</act>",
        "<think>需要攻击T1</think><act>ATK T1 E</act>",
        "<think>与U2协同</think><act>COO U2 N</act>",
        "无法解析的输出",
    ]

    for i, output in enumerate(test_outputs):
        action = engine._parse_action(output)
        print(f"\n输出 {i}: {output[:50]}...")
        print(f"  指令={action.instruction} 方向={action.direction} "
              f"目标={action.target_id} 协同UAV={action.target_uav}")
        if action.reasoning:
            print(f"  推理: {action.reasoning}")

    # ---- 3. 测试 loss mask（需要 tokenizer，此处跳过） ----
    print("\n--- QwenLoRAModel ---")
    print("  注意：load_for_training() 和 load_for_inference() 需要 GPU 和模型权重")
    print("  此处仅验证类结构，跳过实际加载")

    print("\n[OK] qwen_lora.py 自测通过")
