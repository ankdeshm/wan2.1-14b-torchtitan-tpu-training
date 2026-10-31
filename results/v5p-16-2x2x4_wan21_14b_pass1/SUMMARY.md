# Wan2.1-T2V-14B (`14.28B` 3D Video DiT) — TorchTitan + TorchTPU `v5p-16` Results

- **Updated (UTC):** `2026-10-31T05:45:29.623288+00:00`
- **Cluster / Nodepool:** `tpu-v5p-16-pool` (`2x2x4`, `16` TPU `v5p` ranks)
- **Model:** `Wan-AI/Wan2.1-T2V-14B-Diffusers` (`14,284,487,808` params, `1,095` keys across 12 safetensors shards)
- **Reference Baseline (`AI-Hypercomputer/tpu-recipes` JAX MaxDiffusion `v5p-16` `81x480x832`):** `~20.13s/step` (`bsz=8` global = `0.5` videos/chip)

| # | Experiment | Parallelism | AC Policy | `sa_block` | Quantization | Seq `S` | GradAccum | Steady Step (s) | Videos/s (Cluster) | Tok/s/rank | Peak HBM (GiB) | MFU (%) | Loss (`step 1 -> N`) | Status |
|---:|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | `wan21_p1_01_fsdp16_fullac_s32k_81x480x832_pretrained` | `FSDP-16` | `FullAC (recompute attn)` | `512` | `BF16 (FP32 Reduce)` | `32,768` | `1x` | **33.820s** | `0.473` | **969.0** | `25.77` | **35.36%** | `2.0717 -> 2.0095` | `OK` |
| 2 | `wan21_p1_02_fsdp16_splash_save_ac_s32k_81x480x832_pretrained` | `FSDP-16` | `SplashSaveAC (28/40 blocks saved)` | `512` | `BF16 (FP32 Reduce)` | `32,768` | `1x` | **31.747s** | `0.504` | **1,032.3** | `31.51` | **37.67%** | `2.0717 -> 1.9947` | `OK` |
