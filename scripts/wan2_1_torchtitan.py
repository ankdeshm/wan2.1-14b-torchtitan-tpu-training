#!/usr/bin/env python3
"""Native PyTorch TorchTitan + TorchTPU 0.2.0 GA Training Module for Wan2.1-T2V-14B (14.28B 3D Video DiT).

Exact 1,095/1,095 Safetensors Key & Shape Match with `Wan-AI/Wan2.1-T2V-14B-Diffusers` (`14,284,487,808` params):
  - `dim = 5,120` (`40 x 128`), `num_attention_heads = 40`, `attention_head_dim = 128`
  - `ffn_dim = 13,824` (`108 x 128` GELU-tanh MLP)
  - `num_layers = 40` (`WanTransformerBlock`: `attn1` 3D self-attn + `attn2` UMT5 cross-attn + `ffn`)
  - Across-heads `RMSNorm(5120, eps=1e-6)` QK-Norm + 3D Axial RoPE `(44, 42, 42)`
  - TPU / XLA Optimizations:
      1. Bidirectional Pallas `SplashAttention(is_causal=False)` (`FullMask`) on `attn1` (video 3D self-attention)
      2. Chunked MXU `bmm` (`Q_CHUNK=2048`) on `attn2` (UMT5 `L_txt=512` cross-attention)
      3. `_full_ac_save_splash_policy` (`SplashSaveAC`): saves `pallas::splash_attn_fwd` residuals so backward
         performs ZERO `O(L_vid^2)` self-attention recomputation
      4. Zero-collective CPU dim-0 shard slicing for all 12 safetensors shards (`1,095` keys, `57.15 GB` FP32 -> `BF16`)
      5. Optional `TpuFloat8LinearConverter` (`torchao` tensorwise FP8) ablation alongside native `BF16` MXU
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from functools import partial
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import torch
import spmd_types as spmd
from torch import nn, Tensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    fully_shard,
    MixedPrecisionPolicy,
)
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data.loader import BaseDataLoader
from torchtitan.components.loss import MSELoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
try:
    from torchtitan.components.tokenizer import BaseTokenizer
except ImportError:
    BaseTokenizer = object  # type: ignore[misc,assignment]
from torchtitan.config import (
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig, FullAC
from torchtitan.distributed.fsdp import (
    disable_fsdp_gradient_division,
    enable_fsdp_symm_mem,
    resolve_fsdp_mesh,
)
from torchtitan.experiments.tpu import gmain, profiler_workaround, workarounds
from torchtitan.experiments.tpu.grad_norm import CompiledGradNorm
from torchtitan.experiments.tpu.kernels.splash_attention import SplashAttention
from torchtitan.experiments.tpu.spmd_utils import apply_spmd_safe
from torchtitan.experiments.tpu.tpu_job_config import TPUConfig, TPUTrainerConfig
from torchtitan.models.common.attention import ScaledDotProductInnerAttention
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.protocols import BaseModel, ModelSpec
from torchtitan.protocols.module import Module, ModuleList
from torchtitan.tools.logging import init_logger, logger
from torchtitan.trainer import Trainer


# ==============================================================================
# 1. TorchTitan Module-Protocol-Compliant Wan2.1-14B Building Blocks (1,095 Keys)
# ==============================================================================


class WanLayerNorm(Module):
    """TorchTitan `Module`-compliant LayerNorm (`elementwise_affine` True or False)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int = 5120
        eps: float = 1e-6
        elementwise_affine: bool = False

    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.eps = config.eps
        self.elementwise_affine = config.elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.dim))
            self.bias = nn.Parameter(torch.zeros(self.dim))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        out = nn.functional.layer_norm(
            x.float(),
            (self.dim,),
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return out.to(x.dtype)


class WanPatchEmbed3D(Module):
    """Exact `patch_embedding.{weight [5120,16,1,2,2], bias [5120]}` executed as MXU matmul."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        in_channels: int = 16
        embed_dim: int = 5120
        patch_size: tuple[int, int, int] = (1, 2, 2)

    def __init__(self, config: Config):
        super().__init__()
        pt, ph, pw = config.patch_size
        self.patch_dim = config.in_channels * pt * ph * pw  # 64
        self.embed_dim = config.embed_dim
        self.weight = nn.Parameter(torch.empty(config.embed_dim, config.in_channels, pt, ph, pw))
        self.bias = nn.Parameter(torch.zeros(config.embed_dim))

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, patch_tokens: Tensor) -> Tensor:
        # patch_tokens: [B, L_vid, 64]
        w_2d = self.weight.reshape(self.embed_dim, self.patch_dim)
        return nn.functional.linear(patch_tokens.to(w_2d.dtype), w_2d, self.bias.to(w_2d.dtype))


class WanRotaryEmbed3D(Module):
    """3D Axial RoPE `(44, 42, 42)` (`44 + 42 + 42 = 128`) for Wan2.1 14B."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        axes_dims: tuple[int, int, int] = (44, 42, 42)
        theta: float = 10000.0

    def __init__(self, config: Config):
        super().__init__()
        self.axes_dims = config.axes_dims
        self.theta = config.theta
        for idx, d in enumerate(config.axes_dims):
            inv_freq = 1.0 / (config.theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
            self.register_buffer(f"inv_freq_{idx}", inv_freq, persistent=False)

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        dev = buffer_device or self.inv_freq_0.device
        if dev.type == "meta":
            dev = torch.device("cpu")
        for idx, d in enumerate(self.axes_dims):
            inv_freq = 1.0 / (self.theta ** (torch.arange(0, d, 2, dtype=torch.float32, device=dev) / d))
            setattr(self, f"inv_freq_{idx}", inv_freq)

    def forward(self, video_pos_ids: Tensor) -> tuple[Tensor, Tensor]:
        # video_pos_ids: [L_vid, 3] float32 -> returns cos, sin each [1, L_vid, 1, 128]
        cos_parts, sin_parts = [], []
        for idx in range(3):
            inv_f = getattr(self, f"inv_freq_{idx}").to(device=video_pos_ids.device, dtype=torch.float32)
            freqs = torch.outer(video_pos_ids[:, idx].float(), inv_f)  # [L_vid, d_i/2]
            emb = torch.repeat_interleave(freqs, 2, dim=-1)  # [L_vid, d_i]
            cos_parts.append(emb.cos())
            sin_parts.append(emb.sin())
        cos = torch.cat(cos_parts, dim=-1).unsqueeze(0).unsqueeze(2)
        sin = torch.cat(sin_parts, dim=-1).unsqueeze(0).unsqueeze(2)
        return cos, sin


def _apply_wan_rope_3d(x: Tensor, rotary_cs: tuple[Tensor, Tensor]) -> Tensor:
    # x: [B, S, H, D]
    cos, sin = rotary_cs
    x_f = x.float()
    x_even = x_f[..., 0::2]
    x_odd = x_f[..., 1::2]
    x_rot = torch.stack((-x_odd, x_even), dim=-1).flatten(-2)
    out = x_f * cos + x_rot * sin
    return out.to(x.dtype)


class WanTimestepEmbedder(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        freq_dim: int = 256
        inner_dim: int = 5120

    def __init__(self, config: Config):
        super().__init__()
        self.freq_dim = config.freq_dim
        self.linear_1 = Linear.Config(in_features=config.freq_dim, out_features=config.inner_dim, bias=True).build()
        self.linear_2 = Linear.Config(in_features=config.inner_dim, out_features=config.inner_dim, bias=True).build()

    def forward(self, timesteps_proj: Tensor) -> Tensor:
        x = nn.functional.silu(self.linear_1(timesteps_proj.to(self.linear_1.weight.dtype)))
        return self.linear_2(x)


class WanTextEmbedder(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        text_dim: int = 4096
        inner_dim: int = 5120

    def __init__(self, config: Config):
        super().__init__()
        self.linear_1 = Linear.Config(in_features=config.text_dim, out_features=config.inner_dim, bias=True).build()
        self.linear_2 = Linear.Config(in_features=config.inner_dim, out_features=config.inner_dim, bias=True).build()

    def forward(self, caption: Tensor) -> Tensor:
        x = nn.functional.gelu(self.linear_1(caption.to(self.linear_1.weight.dtype)), approximate="tanh")
        return self.linear_2(x)


class WanConditionEmbedder(Module):
    """Exact `condition_embedder.{time_embedder, time_proj, text_embedder}` match."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        freq_dim: int = 256
        inner_dim: int = 5120
        text_dim: int = 4096

    def __init__(self, config: Config):
        super().__init__()
        self.freq_dim = config.freq_dim
        self.time_embedder = WanTimestepEmbedder.Config(freq_dim=config.freq_dim, inner_dim=config.inner_dim).build()
        self.time_proj = Linear.Config(
            in_features=config.inner_dim, out_features=6 * config.inner_dim, bias=True
        ).build()
        self.text_embedder = WanTextEmbedder.Config(text_dim=config.text_dim, inner_dim=config.inner_dim).build()

    def _sinusoidal(self, timesteps: Tensor) -> Tensor:
        half = self.freq_dim // 2
        exponent = -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
        emb = timesteps.float().unsqueeze(-1) * torch.exp(exponent).unsqueeze(0)
        return torch.cat((torch.cos(emb), torch.sin(emb)), dim=-1)

    def forward(self, timesteps: Tensor, encoder_hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        t_freq = self._sinusoidal(timesteps)
        temb = self.time_embedder(t_freq)  # [B, 5120]
        timestep_proj = self.time_proj(nn.functional.silu(temb)).unflatten(1, (6, -1))  # [B, 6, 5120]
        enc_emb = self.text_embedder(encoder_hidden_states)  # [B, L_txt, 5120]
        return temb, timestep_proj, enc_emb


class WanSelfAttention(Module):
    """`attn1` (Video 3D Self-Attention) -> Patched with Pallas `SplashAttention(is_causal=False)`."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int = 5120
        heads: int = 40
        head_dim: int = 128
        eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.head_dim
        inner = config.heads * config.head_dim
        self.to_q = Linear.Config(in_features=config.dim, out_features=inner, bias=True).build()
        self.to_k = Linear.Config(in_features=config.dim, out_features=inner, bias=True).build()
        self.to_v = Linear.Config(in_features=config.dim, out_features=inner, bias=True).build()
        # Across-heads RMSNorm(5120, eps=1e-6) — matches `[5120]` safetensors shape!
        self.norm_q = RMSNorm.Config(normalized_shape=inner, eps=config.eps).build()
        self.norm_k = RMSNorm.Config(normalized_shape=inner, eps=config.eps).build()
        self.inner_attention = ScaledDotProductInnerAttention.Config().build()
        self.to_out = ModuleList([Linear.Config(in_features=inner, out_features=config.dim, bias=True).build()])

    def forward(self, hidden_states: Tensor, rotary_cs: tuple[Tensor, Tensor]) -> Tensor:
        bsz, seqlen, _ = hidden_states.shape
        q = self.norm_q(self.to_q(hidden_states)).view(bsz, seqlen, self.heads, self.head_dim)
        k = self.norm_k(self.to_k(hidden_states)).view(bsz, seqlen, self.heads, self.head_dim)
        v = self.to_v(hidden_states).view(bsz, seqlen, self.heads, self.head_dim)
        q = _apply_wan_rope_3d(q, rotary_cs)
        k = _apply_wan_rope_3d(k, rotary_cs)
        out = self.inner_attention(q, k, v, is_causal=False)
        out = out.reshape(bsz, seqlen, self.heads * self.head_dim)
        return self.to_out[0](out)


class WanCrossAttention(Module):
    """`attn2` (UMT5 `L_txt=512` Cross-Attention) executed via chunked MXU `bmm` (`Q_CHUNK=2048`)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int = 5120
        heads: int = 40
        head_dim: int = 128
        eps: float = 1e-6
        q_chunk: int = 2048

    def __init__(self, config: Config):
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.head_dim
        self.q_chunk = config.q_chunk
        self.scale = 1.0 / math.sqrt(config.head_dim)
        inner = config.heads * config.head_dim
        self.to_q = Linear.Config(in_features=config.dim, out_features=inner, bias=True).build()
        self.to_k = Linear.Config(in_features=config.dim, out_features=inner, bias=True).build()
        self.to_v = Linear.Config(in_features=config.dim, out_features=inner, bias=True).build()
        self.norm_q = RMSNorm.Config(normalized_shape=inner, eps=config.eps).build()
        self.norm_k = RMSNorm.Config(normalized_shape=inner, eps=config.eps).build()
        self.to_out = ModuleList([Linear.Config(in_features=inner, out_features=config.dim, bias=True).build()])

    def forward(self, hidden_states: Tensor, encoder_hidden_states: Tensor) -> Tensor:
        bsz, q_len, _ = hidden_states.shape
        kv_len = encoder_hidden_states.shape[1]
        q = (
            self.norm_q(self.to_q(hidden_states))
            .view(bsz, q_len, self.heads, self.head_dim)
            .transpose(1, 2)
        )  # [B, H, Q, D]
        k = (
            self.norm_k(self.to_k(encoder_hidden_states))
            .view(bsz, kv_len, self.heads, self.head_dim)
            .transpose(1, 2)
        )  # [B, H, KV, D]
        v = (
            self.to_v(encoder_hidden_states)
            .view(bsz, kv_len, self.heads, self.head_dim)
            .transpose(1, 2)
        )  # [B, H, KV, D]

        kt = k.transpose(-2, -1)  # [B, H, D, KV]
        chunks = []
        step = self.q_chunk
        for s in range(0, q_len, step):
            q_c = q[:, :, s : s + step, :]
            scores = torch.matmul(q_c, kt).float() * self.scale
            probs = torch.softmax(scores, dim=-1).to(v.dtype)
            chunks.append(torch.matmul(probs, v))
        out = torch.cat(chunks, dim=2).transpose(1, 2).contiguous().view(bsz, q_len, self.heads * self.head_dim)
        return self.to_out[0](out)


class WanGELUProj(Module):
    """Matches `ffn.net.0.proj.{weight,bias}`."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim_in: int = 5120
        dim_out: int = 13824

    def __init__(self, config: Config):
        super().__init__()
        self.proj = Linear.Config(in_features=config.dim_in, out_features=config.dim_out, bias=True).build()

    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.gelu(self.proj(x), approximate="tanh")


class IdentityAct(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x


class WanFeedForward(Module):
    """Exact `ffn.net.{0.proj, 2}` parameter FQN match with diffusers `FeedForward`."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int = 5120
        ffn_dim: int = 13824

    def __init__(self, config: Config):
        super().__init__()
        self.net = ModuleList(
            [
                WanGELUProj.Config(dim_in=config.dim, dim_out=config.ffn_dim).build(),
                IdentityAct.Config().build(),
                Linear.Config(in_features=config.ffn_dim, out_features=config.dim, bias=True).build(),
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.net[0](x)
        return self.net[2](x)


class WanTransformerBlock(Module):
    """Exact `blocks.{i}` match (`27` tensors per block x `40` blocks = `1,080` tensors)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int = 5120
        ffn_dim: int = 13824
        heads: int = 40
        head_dim: int = 128
        eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, config.dim) / math.sqrt(config.dim))
        self.norm1 = WanLayerNorm.Config(dim=config.dim, eps=config.eps, elementwise_affine=False).build()
        self.attn1 = WanSelfAttention.Config(
            dim=config.dim, heads=config.heads, head_dim=config.head_dim, eps=config.eps
        ).build()
        self.norm2 = WanLayerNorm.Config(dim=config.dim, eps=config.eps, elementwise_affine=True).build()
        self.attn2 = WanCrossAttention.Config(
            dim=config.dim, heads=config.heads, head_dim=config.head_dim, eps=config.eps
        ).build()
        self.norm3 = WanLayerNorm.Config(dim=config.dim, eps=config.eps, elementwise_affine=False).build()
        self.ffn = WanFeedForward.Config(dim=config.dim, ffn_dim=config.ffn_dim).build()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.scale_shift_table, std=1.0 / math.sqrt(self.scale_shift_table.shape[-1]))

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep_proj: Tensor,
        rotary_cs: tuple[Tensor, Tensor],
    ) -> Tensor:
        # timestep_proj: [B, 6, 5120], scale_shift_table: [1, 6, 5120]
        mod = (self.scale_shift_table.float() + timestep_proj.float()).unsqueeze(2)  # [B, 6, 1, 5120]
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = mod.unbind(dim=1)

        norm_h = (self.norm1(hidden_states).float() * (1.0 + scale_msa) + shift_msa).to(hidden_states.dtype)
        attn1_out = self.attn1(norm_h, rotary_cs)
        hidden_states = (hidden_states.float() + attn1_out.float() * gate_msa).to(hidden_states.dtype)

        norm2_h = self.norm2(hidden_states)
        attn2_out = self.attn2(norm2_h, encoder_hidden_states)
        hidden_states = hidden_states + attn2_out

        norm3_h = (self.norm3(hidden_states).float() * (1.0 + c_scale_msa) + c_shift_msa).to(hidden_states.dtype)
        ffn_out = self.ffn(norm3_h)
        hidden_states = (hidden_states.float() + ffn_out.float() * c_gate_msa).to(hidden_states.dtype)
        return hidden_states


class Wan21Model(BaseModel):
    """Native TorchTitan `Wan-AI/Wan2.1-T2V-14B-Diffusers` (`14,284,487,808` params, `1,095` keys)."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        dim: int = 5120
        ffn_dim: int = 13824
        heads: int = 40
        head_dim: int = 128
        num_layers: int = 40
        in_channels: int = 16
        out_channels: int = 16
        patch_size: tuple[int, int, int] = (1, 2, 2)
        text_dim: int = 4096
        l_txt: int = 512
        eps: float = 1e-6

        def update_from_config(self, *, config, **kwargs) -> None:
            pass

        def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
            nparams = sum(p.numel() for p in model.parameters())
            # Exact bidirectional (non-causal) forward+backward FLOPs per video token:
            # Dense trunk: 6 * N_params + Cross-attn: 12 * L * dim * l_txt + Self-attn: 12 * L * dim * seq_len
            dense_and_xattn = 6 * nparams + 12 * self.num_layers * self.dim * self.l_txt
            self_attn = 12 * self.num_layers * self.dim * seq_len
            return nparams, int(dense_and_xattn + self_attn)

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.rope = WanRotaryEmbed3D.Config().build()
        self.patch_embedding = WanPatchEmbed3D.Config(
            in_channels=config.in_channels, embed_dim=config.dim, patch_size=config.patch_size
        ).build()
        self.condition_embedder = WanConditionEmbedder.Config(
            freq_dim=256, inner_dim=config.dim, text_dim=config.text_dim
        ).build()
        self.blocks = ModuleList(
            [
                WanTransformerBlock.Config(
                    dim=config.dim,
                    ffn_dim=config.ffn_dim,
                    heads=config.heads,
                    head_dim=config.head_dim,
                    eps=config.eps,
                ).build()
                for _ in range(config.num_layers)
            ]
        )
        self.norm_out = WanLayerNorm.Config(dim=config.dim, eps=config.eps, elementwise_affine=False).build()
        pt, ph, pw = config.patch_size
        self.proj_out = Linear.Config(
            in_features=config.dim, out_features=config.out_channels * pt * ph * pw, bias=True
        ).build()
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, config.dim) / math.sqrt(config.dim))

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        self.rope.init_weights(buffer_device=buffer_device)
        nn.init.normal_(self.scale_shift_table, std=1.0 / math.sqrt(self.config.dim))
        for name, mod in self.named_modules():
            if mod is self or mod is self.rope:
                continue
            if hasattr(mod, "reset_parameters"):
                mod.reset_parameters()

    def forward(
        self,
        noisy_video_patches: Tensor,
        encoder_hidden_states: Tensor,
        timesteps: Tensor,
        video_pos_ids: Tensor,
    ) -> Tensor:
        rotary_cs = self.rope(video_pos_ids)
        temb, timestep_proj, enc_emb = self.condition_embedder(timesteps, encoder_hidden_states)
        hidden_states = self.patch_embedding(noisy_video_patches)

        for block in self.blocks:
            hidden_states = block(hidden_states, enc_emb, timestep_proj, rotary_cs)

        shift, scale = (self.scale_shift_table.float() + temb.float().unsqueeze(1)).unbind(dim=1)
        hidden_states = (self.norm_out(hidden_states).float() * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)).to(
            hidden_states.dtype
        )
        return self.proj_out(hidden_states.to(self.proj_out.weight.dtype))


# ==============================================================================
# 2. Parallelization, SplashAttention Patching, Selective AC & Pretrained Load
# ==============================================================================


def _is_splash_attention_fwd(func) -> bool:
    return getattr(func, "namespace", None) == "pallas" and "splash_attn_fwd" in getattr(func, "_opname", "")


def _full_ac_save_splash_policy(ctx, func, *args, **kwargs) -> CheckpointPolicy:
    if _is_splash_attention_fwd(func):
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def parallelize_wan21(
    model: Wan21Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
    sa_block: int = 512,
    save_splash_in_ac: bool = True,
    use_fp8_linear: bool = False,
) -> nn.Module:
    if use_fp8_linear:
        from torchtitan.experiments.tpu.float8_converter import TpuFloat8LinearConverter

        fp8_cfg = TpuFloat8LinearConverter.Config(
            enable_fsdp_float8_all_gather=False,
            precompute_float8_dynamic_scale_for_fsdp=False,
            recipe_name="tensorwise",
            filter_fqns=["patch_embedding", "condition_embedder", "norm_out", "proj_out"],
        )
        TpuFloat8LinearConverter(fp8_cfg).convert(model)
        logger.info("Applied TpuFloat8LinearConverter (tensorwise FP8) to 40 WanTransformerBlocks")

    # Patch ONLY `attn1` (video self-attention) with bidirectional Pallas SplashAttention(is_causal=False)
    for blk in model.blocks:
        blk.attn1.inner_attention = SplashAttention(
            is_causal=False,
            original_module=blk.attn1.inner_attention,
            block_q=sa_block,
            block_kv=sa_block,
            block_kv_compute=sa_block,
            block_q_dkv=sa_block,
            block_kv_dkv=sa_block,
            block_kv_dkv_compute=sa_block,
            use_document_aware_block_skipping=False,
        )
    logger.info(f"Patched 40 WanTransformerBlocks.attn1 with bidirectional SplashAttention(sa_block={sa_block})")

    if ac_config is not None and getattr(ac_config, "mode", "full") != "none":
        dp_total = max(1, parallel_dims.dp_replicate * parallel_dims.dp_shard)
        if save_splash_in_ac:
            if training.max_context_length >= 32768 and dp_total <= 16:
                first_full_ac = 12  # Save SplashAttention on last 28/40 blocks (~35 GiB saved, >35 GiB free on 95 GiB v5p)
            else:
                first_full_ac = 0  # Save SplashAttention on all 40/40 blocks
        else:
            first_full_ac = 40
        splash_ctx_fn = lambda: create_selective_checkpoint_contexts(_full_ac_save_splash_policy)
        for layer_id, blk in model.blocks.named_children():
            use_save = save_splash_in_ac and int(layer_id) >= first_full_ac
            wrapped = (
                ptd_checkpoint_wrapper(blk, preserve_rng_state=False, context_fn=splash_ctx_fn)
                if use_save
                else ptd_checkpoint_wrapper(blk, preserve_rng_state=False)
            )
            model.blocks.register_module(layer_id, wrapped)
        logger.info(
            f"Applied Activation Checkpointing (save_splash_in_ac={save_splash_in_ac}, "
            f"saved_blocks={40 - first_full_ac}/40) to 40 blocks"
        )

    if compile_config.enable and "model" in compile_config.components:
        for blk in model.blocks:
            apply_spmd_safe(blk)
            blk.compile(backend=compile_config.backend, fullgraph=True)
        logger.info(f"Compiled 40 WanTransformerBlocks with torch.compile(backend={compile_config.backend!r})")

    if parallel_dims.spmd_backend == "spmd_types":
        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
    else:
        names = ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        dp_mesh = parallel_dims.get_mesh(names)
        dp_mesh_dims = None

    param_dtype = TORCH_DTYPE_MAP[training.mixed_precision_param]
    reduce_dtype = TORCH_DTYPE_MAP[training.mixed_precision_reduce]
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_cfg: dict[str, Any] = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if dp_mesh_dims is not None:
        fsdp_cfg["dp_mesh_dims"] = dp_mesh_dims
    if training.enable_cpu_offload:
        fsdp_cfg["offload_policy"] = CPUOffloadPolicy()

    fully_shard(model.patch_embedding, **fsdp_cfg)
    fully_shard(model.condition_embedder, **fsdp_cfg)
    for blk in model.blocks:
        fully_shard(blk, **fsdp_cfg)
    fully_shard(model.proj_out, **fsdp_cfg, reshard_after_forward=False)
    fully_shard(model, **fsdp_cfg)
    if parallelism.enable_fsdp_symm_mem:
        enable_fsdp_symm_mem(model)
    disable_fsdp_gradient_division(model)
    logger.info(
        f"Applied FSDP2 fully_shard (replicate={parallel_dims.dp_replicate}, shard={parallel_dims.dp_shard}) to Wan21Model"
    )
    return model


def load_pretrained_wan21_safetensors(model: nn.Module, checkpoint_dir: str) -> int:
    """Zero-collective CPU dim-0 shard streaming loader for all 12 shards (`1,095` keys)."""
    from safetensors import safe_open

    ckpt_path = Path(checkpoint_dir)
    index_file = ckpt_path / "diffusion_pytorch_model.safetensors.index.json"
    if not index_file.exists():
        logger.warning(f"Wan2.1 checkpoint index not found at {index_file}; keeping random init.")
        return 0

    weight_map: dict[str, str] = json.loads(index_file.read_text())["weight_map"]
    param_map: dict[str, nn.Parameter] = {}
    for raw_name, param in model.named_parameters():
        clean = raw_name.replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", "")
        param_map[clean] = param

    by_shard: dict[str, list[str]] = {}
    for key, shard_name in weight_map.items():
        by_shard.setdefault(shard_name, []).append(key)

    loaded = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for shard_idx, (shard_name, keys) in enumerate(sorted(by_shard.items()), 1):
            shard_file = ckpt_path / shard_name
            with safe_open(str(shard_file), framework="pt", device="cpu") as f:
                for key in keys:
                    param = param_map.get(key)
                    if param is None:
                        continue
                    full_tensor = f.get_tensor(key)
                    if isinstance(param, DTensor):
                        local_target = param._local_tensor
                        if local_target.shape == full_tensor.shape:
                            cpu_slice = full_tensor
                        else:
                            mesh = param.device_mesh
                            shard_dim_idx = mesh.ndim - 1
                            shard_rank = mesh.get_local_rank(shard_dim_idx)
                            local_rows = local_target.shape[0]
                            start = shard_rank * local_rows
                            end = min(full_tensor.shape[0], start + local_rows)
                            cpu_slice = torch.zeros(local_target.shape, dtype=full_tensor.dtype)
                            if end > start:
                                cpu_slice[: end - start].copy_(full_tensor[start:end])
                        try:
                            local_target.copy_(cpu_slice.to(device=local_target.device, dtype=local_target.dtype))
                        except Exception:
                            if hasattr(local_target, "_data"):
                                local_target._data.copy_(
                                    cpu_slice.to(device=local_target._data.device, dtype=local_target._data.dtype)
                                )
                        del cpu_slice
                    else:
                        param.copy_(full_tensor.to(device=param.device, dtype=param.dtype))
                    loaded += 1
                    del full_tensor
            if shard_idx % 4 == 0 or shard_idx == len(by_shard):
                logger.info(
                    f"[Wan2.1-14B] Loaded {shard_idx}/{len(by_shard)} shards "
                    f"({loaded}/{len(weight_map)} keys in {time.perf_counter() - t0:.1f}s)"
                )
    return loaded


# ==============================================================================
# 3. Minimal Tokenizer, Cached 3D Video Latent DataLoader & Wan21Trainer
# ==============================================================================


class Wan21NullTokenizer(BaseTokenizer):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseTokenizer.Config):
        pass

    def __init__(self, config: Config | None = None, **kwargs):
        super().__init__()
        self.eos_id = 0

    def encode(self, *args, **kwargs) -> list[int]:
        return [0]

    def decode(self, *args, **kwargs) -> str:
        return ""

    def get_vocab_size(self) -> int:
        return 256384


class Wan21CachedVideoDataLoader(BaseDataLoader):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        l_txt: int = 512
        text_dim: int = 4096
        patch_dim: int = 64  # 16 * 1 * 2 * 2

        def build(
            self,
            *,
            dp_world_size: int,
            dp_rank: int,
            tokenizer: Any,
            max_context_length: int,
            num_tokens_per_batch: int,
            **kwargs: Any,
        ) -> "Wan21CachedVideoDataLoader":
            return Wan21CachedVideoDataLoader(
                config=self,
                dp_rank=dp_rank,
                seq_len=max_context_length,
                mb_tokens=num_tokens_per_batch or max_context_length,
            )

    def __init__(self, *, config: Config, dp_rank: int, seq_len: int, mb_tokens: int):
        self.config = config
        self.max_num_documents = None
        self.seq_len = seq_len
        self.bsz = max(1, mb_tokens // seq_len)

        g = torch.Generator(device="cpu")
        g.manual_seed(20261101 + dp_rank * 991)
        self._video_clean = torch.randn(self.bsz, seq_len, config.patch_dim, generator=g, dtype=torch.bfloat16)
        self._text_cond = torch.randn(self.bsz, config.l_txt, config.text_dim, generator=g, dtype=torch.bfloat16)
        self._pos_ids = self._build_3d_pos(seq_len)

    def _build_3d_pos(self, l_vid: int) -> Tensor:
        h_v = 30 if l_vid % 30 == 0 else 32
        w_v = 52 if (l_vid // h_v) % 52 == 0 else max(1, min(64, l_vid // h_v))
        t_v = max(1, math.ceil(l_vid / (h_v * w_v)))
        tt, hh, ww = torch.meshgrid(
            torch.arange(t_v, dtype=torch.float32),
            torch.arange(h_v, dtype=torch.float32),
            torch.arange(w_v, dtype=torch.float32),
            indexing="ij",
        )
        return torch.stack((tt.reshape(-1), hh.reshape(-1), ww.reshape(-1)), dim=-1)[:l_vid]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        step_idx = 0
        while True:
            step_idx += 1
            yield {
                "video_clean": self._video_clean,
                "text_cond": self._text_cond,
                "pos_ids": self._pos_ids,
                "step_idx": step_idx,
            }

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        pass

    def close(self) -> None:
        pass


class Wan21Trainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(TPUTrainerConfig):
        tokenizer: Wan21NullTokenizer.Config = field(default_factory=Wan21NullTokenizer.Config)  # type: ignore[assignment]
        dataloader: Wan21CachedVideoDataLoader.Config = field(  # type: ignore[assignment]
            default_factory=Wan21CachedVideoDataLoader.Config
        )
        hf_checkpoint_dir: str = "/models/models/Wan-AI/Wan2.1-T2V-14B-Diffusers/transformer"
        load_pretrained_weights: bool = True
        l_txt: int = 512
        sa_block: int = 512
        save_splash_in_ac: bool = True
        use_fp8_linear: bool = False
        flow_shift: float = 5.0

    def __init__(self, config: Config):
        super().__init__(config)
        self._dtype = TORCH_DTYPE_MAP[config.training.mixed_precision_param]
        use_compiled_gn = getattr(config.compile, "compile_grad_norm", False) or getattr(
            config.tpu_config, "compile_grad_norm", False
        )
        self._compiled_grad_norm = CompiledGradNorm() if use_compiled_gn else None
        use_compiled_opt = (
            getattr(config.compile, "optimizer_compile", None) == "separate_graph"
            or getattr(config.tpu_config, "compile_optimizer", False)
        )
        if use_compiled_opt:
            from torchtitan.experiments.tpu.compiled_optimizers.separate_graph import apply_compiled_optimizer

            apply_compiled_optimizer(self)
            logger.info("Applied separate_graph compiled AdamW optimizer to Wan21Trainer")

        if config.load_pretrained_weights and config.hf_checkpoint_dir:
            load_pretrained_wan21_safetensors(self.model_parts[0], config.hf_checkpoint_dir)

    def _clip_grad_norm(self) -> Tensor:
        if self._compiled_grad_norm is None:
            return dist_utils.clip_grad_norm_(
                [p for m in self.model_parts for p in m.parameters()],
                self.config.training.max_norm,
                foreach=True,
                pp_mesh=self.parallel_dims.get_optional_mesh("pp"),
                ep_enabled=self.parallel_dims.ep_enabled,
            )
        return self._compiled_grad_norm.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()], self.config.training.max_norm
        )

    def batch_generator(self, data_iterable: Iterable[Any]) -> Iterator[dict[str, Any]]:
        it = iter(data_iterable)
        while True:
            t0 = time.perf_counter()
            batch = next(it)
            bsz = batch["video_clean"].shape[0]
            self.metrics_processor.ntokens_since_last_log += bsz * self.config.training.max_context_length
            self.metrics_processor.data_loading_times.append(time.perf_counter() - t0)
            yield batch

    def forward_backward_step(self, *, input_dict: dict[str, Any], loss_scale: float = 1.0) -> Tensor:
        video_clean = input_dict["video_clean"].to(device=self.device, dtype=self._dtype)
        text_cond = input_dict["text_cond"].to(device=self.device, dtype=self._dtype)
        pos_ids = input_dict["pos_ids"].to(device=self.device, dtype=torch.float32)

        bsz = video_clean.shape[0]
        self.ntokens_seen += bsz * self.config.training.max_context_length

        with torch.no_grad():
            noise = torch.randn_like(video_clean)
            u = torch.rand((bsz,), device=self.device, dtype=torch.float32)
            shift = self.config.flow_shift
            sigma = (shift * u) / (1.0 + (shift - 1.0) * u)
            sigma_v = sigma.view(bsz, 1, 1).to(self._dtype)
            noisy_video = (1.0 - sigma_v) * video_clean + sigma_v * noise
            target = (noise - video_clean).float()
            timesteps = sigma * 1000.0

        model = self.model_parts[0]
        with self.train_context():
            pred = model(noisy_video, text_cond, timesteps, pos_ids)
            loss = torch.nn.functional.mse_loss(pred.float(), target, reduction="mean") * loss_scale
            del pred, target, noise
            with spmd.no_typecheck():
                loss.backward()
        return loss

    def train_step(self, data_iterator: Iterator[dict[str, Any]]) -> None:
        self.optimizers.zero_grad()
        lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]
        ga_steps = max(1, self.gradient_accumulation_steps)
        loss_scale = 1.0 / ga_steps

        accum_loss = torch.zeros((), device=self.device, dtype=torch.float32)
        for mb_idx in range(ga_steps):
            batch = next(data_iterator)
            mb_loss = self.forward_backward_step(input_dict=batch, loss_scale=loss_scale)
            accum_loss = accum_loss + mb_loss.detach().float()
            torch.accelerator.synchronize()

        grad_norm = self._clip_grad_norm()
        self.checkpointer.maybe_wait_for_staging()
        torch.accelerator.synchronize()
        self.optimizers.step()
        torch.accelerator.synchronize()
        self.lr_schedulers.step()

        if not self.metrics_processor.should_log(self.step):
            return

        if self.parallel_dims.dp_cp_enabled:
            loss_mesh = self.parallel_dims.get_optional_mesh("loss")
            dp_size = max(1, loss_mesh.size()) if loss_mesh is not None else 1
            global_avg_loss = dist_utils.dist_sum(accum_loss, loss_mesh) / dp_size
            global_max_loss = dist_utils.dist_max(accum_loss, loss_mesh)
            global_ntokens_seen = dist_utils.dist_sum(
                torch.tensor(self.ntokens_seen, dtype=torch.int64, device=self.device),
                loss_mesh,
            )
        else:
            global_avg_loss = global_max_loss = float(accum_loss.item())
            global_ntokens_seen = self.ntokens_seen

        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            float(grad_norm.item()),
            extra_metrics={"n_tokens_seen": global_ntokens_seen, "lr": lr},
        )


def build_wan21_model_spec(
    *, sa_block: int = 512, save_splash_in_ac: bool = True, use_fp8_linear: bool = False, l_txt: int = 512
) -> ModelSpec:
    return ModelSpec(
        name="wan2_1",
        flavor="t2v-14B",
        model=Wan21Model.Config(l_txt=l_txt),
        max_context_length=32768,
        parallelize_fn=partial(
            parallelize_wan21,
            sa_block=sa_block,
            save_splash_in_ac=save_splash_in_ac,
            use_fp8_linear=use_fp8_linear,
        ),
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )


def wan2_1_14b_tpu() -> Wan21Trainer.Config:
    seq_len = 32768  # Matches Google MaxDiffusion v5p-16 81x480x832 padded token count (`31,590 -> 32,768`)!
    return Wan21Trainer.Config(
        hf_assets_path="",
        loss=MSELoss.Config(),
        tpu_config=TPUConfig(eager_mode="DEFER_AND_FUSE"),
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=build_wan21_model_spec(sa_block=512, save_splash_in_ac=True),
        optimizer=default_adamw(lr=5e-5, implementation="foreach"),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=0, decay_ratio=0.0),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=seq_len,
            max_context_length=seq_len,
            max_norm=1.0,
            steps=10,
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
            disable_cuda_graphs=True,
        ),
        compile=CompileConfig(enable=True, backend="tpu", components=["model"]),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
        ),
        activation_checkpoint=FullAC.Config(),
        checkpoint=CheckpointManager.Config(enable=False),
        dataloader=Wan21CachedVideoDataLoader.Config(),
        sa_block=512,
        save_splash_in_ac=True,
        use_fp8_linear=False,
    )


def start_wan21_trainer(config: Wan21Trainer.Config) -> None:
    rank = int(os.getenv("RANK", "0"))
    if rank == 0:
        init_logger()

    config.model_spec = build_wan21_model_spec(
        sa_block=config.sa_block,
        save_splash_in_ac=config.save_splash_in_ac,
        use_fp8_linear=config.use_fp8_linear,
        l_txt=config.l_txt,
    )
    import torch._dynamo.config as dynamo_config

    dynamo_config.recompile_limit = max(dynamo_config.recompile_limit, 256)
    dynamo_config.accumulated_recompile_limit = max(dynamo_config.accumulated_recompile_limit, 2048)
    profiler_workaround.init(config)
    workarounds.use_cpu_safe_histc_patch()

    if config.debug.seed is None:
        config.debug.seed = 42

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    trainer: Wan21Trainer | None = None
    try:
        trainer = Wan21Trainer(config)
        trainer.train()
    finally:
        if trainer is not None:
            trainer.close()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    gmain.handle_main(start_wan21_trainer)
