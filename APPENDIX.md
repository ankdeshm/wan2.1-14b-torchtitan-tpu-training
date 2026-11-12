# Appendix: Detailed Ablations, Engineering Notes, and Comparison Caveats (Wan2.1-T2V-14B on TPU v5p)

This document records the complete 6-experiment optimization progression, TorchTitan DiT architecture and kernel details, failure/teardown diagnostics, and setup differences relative to upstream JAX MaxDiffusion. For headline results and reproduction commands, see [`README.md`](./README.md).

---

## 1. Native TorchTitan Porting & Kernel Design

[`scripts/wan2_1_torchtitan.py`](./scripts/wan2_1_torchtitan.py) implements `Wan-AI/Wan2.1-T2V-14B-Diffusers` directly on TorchTitan's `Module` and `Trainer` protocols with an exact `1,095 / 1,095` safetensors key and shape match:

1. **MXU-Aligned Geometry:** Every projection dimension is an exact multiple of 128 (`dim = 5,120 = 40 × 128`, `40` heads × `head_dim = 128`, `ffn_dim = 13,824 = 108 × 128`). The 3D patchify layer (`patch_embedding.weight` of shape `[5120, 16, 1, 2, 2]`) reshapes `16 × 1 × 2 × 2 = 64` patch channels into a single linear MXU matmul (`[B, S, 64] -> [B, S, 5120]`).
2. **Bidirectional Pallas `SplashAttention` + Chunked Cross-Attention:**
   - Video 3D self-attention (`attn1`, `S = 32,768` or `16,384`) uses bidirectional Pallas `SplashAttention(is_causal=False)` (`FullMask`) with 3D axial RoPE `(44, 42, 42)` and across-heads `RMSNorm(5120, eps=1e-6)` QK-normalization.
   - Text cross-attention (`attn2`, attending over `L_txt = 512` UMT5 tokens) uses query-chunked MXU batched matmuls (`Q_CHUNK = 2,048`) with `fp32` softmax so `[B·H, S, 512]` attention logits never materialize as a single full-sequence tensor in HBM.
3. **Selective `SplashSaveAC` Policy (`_full_ac_save_splash_policy`):**
   - Standard `FullAC` recomputes the entire `WanTransformerBlock` during backward, including `O(S²)` self-attention (`80.5 GFLOPs/token` across 40 layers at `S = 32,768` — nearly half of total model FLOPs).
   - `SplashSaveAC` marks `pallas::splash_attn_fwd` as `CheckpointPolicy.MUST_SAVE` while keeping linear projections and FFN layers on `PREFER_RECOMPUTE`. At `S = 32,768` on 16 chips, saving `SplashAttention` outputs on the last `28 / 40` blocks fits comfortably in `21.51–31.51 GiB` HBM per chip (out of `95 GiB`); at `S = 16,384`, all `40 / 40` blocks are saved inside `16.25 GiB` HBM.
4. **Zero-Collective Safetensors Shard Streaming:** `load_pretrained_wan21_safetensors` opens each of the 12 safetensors shards on CPU, slices dim-0 directly onto each rank's local `FSDP2` `DTensor` shard, and copies to TPU `bfloat16` in `2.2–5.9 s` without materializing the full `57.15 GB` `fp32` checkpoint in host RAM or running broadcast collectives.
5. **Exact Bidirectional FLOP Accounting:** Unlike causal LLMs (`6·P + 6·L·D·S`), full bidirectional DiT self-attention + UMT5 cross-attention executes:
   $$\text{FLOPs/token} = 6P + 12 L D L_{\text{txt}} + 12 L D S$$
   With $P = 14,284,487,808$, $L = 40$, $D = 5,120$, and $L_{\text{txt}} = 512$:
   - At $S = 32,768$: $85.71 + 1.26 + 80.53 = 167.50\text{ GFLOPs/token}$.
   - At $S = 16,384$: $85.71 + 1.26 + 40.27 = 127.23\text{ GFLOPs/token}$.

---

## 2. Full `v5p-16` Optimization Progression

All 6 experiments ran on a 4-host `v5p-16` slice (`2x2x4`, 16 chips) starting from pretrained `Wan2.1-T2V-14B-Diffusers` weights with compiled AdamW (`--compile.optimizer_compile=separate_graph`) and compiled grad-norm clipping. Raw outputs are in [`results/v5p-16/summary.json`](./results/v5p-16/summary.json) and [`results/v5p-16/*.log`](./results/v5p-16/).

| # | Experiment Log | Parallelism | AC Policy | `sa_block` | Reduce Dtype | XLA LHS | Seq `S` | Grad Accum (`Vids/Step`) | Cold Step 1 | Steady Step | Cluster Videos/s | Tok/s / Chip | MFU | Peak HBM | Loss (`1 -> N`) | Status |
| ---: | :--- | :--- | :--- | ---: | :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| 1 | [`fsdp16_fullac_s32k.log`](./results/v5p-16/fsdp16_fullac_s32k.log) | `FSDP-16` | `FullAC` (recompute attn) | 512 | `fp32` | Off | 32,768 | `1×` (`16`) | 297.43 s | 33.820 s | 0.473 | 969.0 | 35.36% | 25.77 GiB | `2.0717 -> 2.0095` | `OK` (8 steps) |
| 2 | [`fsdp16_splash_save_ac_s32k.log`](./results/v5p-16/fsdp16_splash_save_ac_s32k.log) | `FSDP-16` | `SplashSaveAC` (`28/40` saved) | 512 | `fp32` | Off | 32,768 | `1×` (`16`) | 68.93 s | 31.747 s | 0.504 | 1,032.3 | 37.67% | 31.51 GiB | `2.0717 -> 1.9947` | `OK` (10 steps) |
| 3 | [`fsdp16_splash_save_sa1024_xla_lhs_s32k.log`](./results/v5p-16/fsdp16_splash_save_sa1024_xla_lhs_s32k.log) | `FSDP-16` | `SplashSaveAC` (`28/40` saved) | 1,024 | `bf16` | On | 32,768 | `1×` (`16`) | 782.46 s | 27.754 s* | 0.577* | 1,180.9* | 43.09%* | 21.51 GiB* | `2.0717 -> 1.9947` | Teardown `rc=-9` after step 10* |
| 4 | **[`fsdp16_splash_save_sa1024_xla_lhs_s32k_ga2.log`](./results/v5p-16/fsdp16_splash_save_sa1024_xla_lhs_s32k_ga2.log)** | **`FSDP-16`** | **`SplashSaveAC` (`28/40` saved)** | **1,024** | **`bf16`** | **On** | **32,768** | **`2×` (`32`)** | **90.83 s** | **53.344 s** | **0.600** | **1,228.4** | **44.83%** | **21.51 GiB** | **`2.0701 -> 1.9939`** | **`OK` (10 steps)** |
| 5 | [`hsdp_2x8_splash_save_sa1024_xla_lhs_s32k.log`](./results/v5p-16/hsdp_2x8_splash_save_sa1024_xla_lhs_s32k.log) | `HSDP 2×8` | `SplashSaveAC` (`28/40` saved) | 1,024 | `bf16` | On | 32,768 | `1×` (`16`) | 666.04 s | 28.596 s* | 0.560* | 1,145.8* | 41.82%* | 21.51 GiB* | `2.0717 -> 2.0094` | Teardown `rc=-9` after step 8* |
| 6 | **[`fsdp16_splash_save_sa1024_xla_lhs_s16k.log`](./results/v5p-16/fsdp16_splash_save_sa1024_xla_lhs_s16k.log)** | **`FSDP-16`** | **`SplashSaveAC` (`40/40` saved)** | **1,024** | **`bf16`** | **On** | **16,384** | **`1×` (`32`)** | **84.21 s** | **19.824 s** | **1.614** | **1,652.7** | **45.81%** | **16.25 GiB** | **`2.0711 -> 1.9939`** | **`OK` (10 steps)** |

*\*Note on Experiments 3 and 5 (`rc=-9` at process exit):* Inspecting the raw logs confirms that Experiment 3 completed all 10 training steps cleanly (`Training completed`, steady steps 4–10 averaging `27.754 s/step`, `1,180.9 tok/s/chip`, `43.09%` MFU) and Experiment 5 completed all 8 steps (`Training completed`, steady steps 4–8 averaging `28.596 s/step`, `1,145.8 tok/s/chip`, `41.82%` MFU). In both runs, step-1 cold XLA compilation with `--xla_latency_hiding_scheduler_rerun=2` took `666–782 s` (`>540 s`), causing the sweep wrapper's idle watchdog on a non-rank-0 worker pod (where `[titan]` stdout only flushes on rank 0) or `torchrun` teardown to send `SIGKILL` (`rc=-9`), so [`summary.json`](./results/v5p-16/summary.json) recorded `status: "rc=-9"` instead of promoting them to `OK`. Subsequent runs (Experiments 4 and 6) hit the warm Tier-2 compilation cache (`90.8 s` and `84.2 s` on step 1) and exited with `rc=0`.

---

## 3. What Worked vs. What Did Not Help

### Optimizations That Improved Throughput
- **Selective `SplashSaveAC` (`+6.5%` throughput, `35.36% -> 37.67%` MFU):** Saving `SplashAttention` outputs on 28 of 40 DiT blocks eliminated `70%` of backward `O(S²)` self-attention recomputation while increasing peak HBM by `5.74 GiB` (`25.77 -> 31.51 GiB`).
- **Widening Pallas `SplashAttention` Tiles (`512 -> 1,024`) + XLA Latency-Hiding Flags + `bfloat16` ReduceScatter (`+14.4%` further throughput at `1×` grad accum, `37.67% -> 43.09%` MFU):** Larger `1,024 × 1,024` VMEM tiles plus async AllGather/ReduceScatter overlap (`--xla_tpu_overlap_compute_collective_tc=true`, `--xla_tpu_scoped_vmem_limit_kib=81920`) and `bfloat16` gradient reduction lowered step time from `31.747 s` to `27.754 s` and reduced peak live HBM from `31.51 GiB` to `21.51 GiB`.
- **`2×` Gradient Accumulation (`+4.0%` further throughput, `43.09% -> 44.83%` MFU):** Accumulating 2 microbatches per optimizer step (`53.344 s` for 32 videos = `26.672 s` per 16 videos) amortized FSDP ReduceScatter, gradient norm clipping, and AdamW step overhead across twice as many tokens.

### Configurations That Did Not Help
- **`HSDP 2×8` (`dp_replicate=2, dp_shard=8`) vs. Pure `FSDP-16`:** On a single `2x2x4` `v5p-16` pod where all 16 chips are connected over high-bandwidth 3D torus ICI, sharding across 8 chips and AllReducing across 2 replicas was `3.0%` slower than pure `FSDP-16` (`28.596 s` vs. `27.754 s` at `1×` grad accum; `41.82%` vs. `43.09%` MFU) because cross-replica AllReduce added extra collective traffic without reducing intra-pod AllGather cost.
- **Ulysses / Ring Context Parallelism:** Unnecessary at `S = 32,768` (`81×480×832`) on `95 GiB` `v5p` chips — `FSDP-16` with `SplashSaveAC` uses only `21.51 GiB` peak HBM (`22.6%` of chip capacity), avoiding per-layer sequence AllToAll collectives completely.
- **FP8 Linear Quantization:** Not enabled on `v5p` because TPU `v5p` MXUs execute native `bfloat16` / `int8` rather than hardware `fp8` tensor-core matmuls.

---

## 4. Comparison Caveats Against JAX MaxDiffusion

1. **Source of JAX Reference:** The `~20.13 s/step` (`bsz = 8`, `0.397` videos/s) baseline is taken from Google's published [`AI-Hypercomputer/tpu-recipes` Wan2.1-14B `bf16-v5p-16` recipe](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/v5p/wan2.1-14b) and was not re-measured in this repository.
2. **Global Batch Size:** Both runs target `81×480×832` videos (`31,590` 3D VAE patches padded to `S = 32,768` tokens/video on `16` `v5p` chips). The published MaxDiffusion recipe uses global batch `8` (`0.5` videos/chip via sequence/context parallelism), whereas our `FSDP-16` runs place `1` full `32,768`-token video per chip (`global batch 16` at `1×` grad accum = `0.577` videos/s, or `global batch 32` at `2×` grad accum = `0.600` videos/s).
3. **Pre-Encoded Latents & Warm Compilation:** Following standard DiT throughput benchmarking practice, [`Wan21CachedVideoDataLoader`](./scripts/wan2_1_torchtitan.py) feeds pre-encoded `[B, S, 64]` video latent patches and `[B, 512, 4096]` UMT5 text embeddings (excluding frozen Wan-VAE and UMT5 encoder forward passes) and reports steady-state step times after initial XLA compilation.
