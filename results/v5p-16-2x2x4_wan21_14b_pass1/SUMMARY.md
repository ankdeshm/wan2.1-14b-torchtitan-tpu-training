# Wan2.1-T2V-14B (`14.28B` 3D Video DiT) — TorchTitan + TorchTPU `v5p-16` Results

- **Updated (UTC):** `2026-10-31T06:33:13.552927+00:00`
- **Cluster / Nodepool:** `tpu-v5p-16-pool` (`2x2x4`, `16` TPU `v5p` ranks)
- **Model:** `Wan-AI/Wan2.1-T2V-14B-Diffusers` (`14,284,487,808` params, `1,095` keys across 12 safetensors shards)
- **Reference Baseline (`AI-Hypercomputer/tpu-recipes` JAX MaxDiffusion `v5p-16` `81x480x832`):** `~20.13s/step` (`bsz=8` global = `0.5` videos/chip)

| # | Experiment | Parallelism | AC Policy | `sa_block` | Quantization | Seq `S` | GradAccum | Steady Step (s) | Videos/s (Cluster) | Tok/s/rank | Peak HBM (GiB) | MFU (%) | Loss (`step 1 -> N`) | Status |
|---:|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | `wan21_p1_01_fsdp16_fullac_s32k_81x480x832_pretrained` | `FSDP-16` | `FullAC (recompute attn)` | `512` | `BF16 (FP32 Reduce)` | `32,768` | `1x` | **33.820s** | `0.473` | **969.0** | `25.77` | **35.36%** | `2.0717 -> 2.0095` | `OK` |
| 2 | `wan21_p1_02_fsdp16_splash_save_ac_s32k_81x480x832_pretrained` | `FSDP-16` | `SplashSaveAC (28/40 blocks saved)` | `512` | `BF16 (FP32 Reduce)` | `32,768` | `1x` | **31.747s** | `0.504` | **1,032.3** | `31.51` | **37.67%** | `2.0717 -> 1.9947` | `OK` |
| 3 | `wan21_p1_03_fsdp16_splash_save_sa1024_xla_lhs_s32k_pretrained` | `FSDP-16` | `SplashSaveAC` | `1024` | `BF16 (BF16 Reduce)` | `32,768` | `1x` | - | - | - | - | - | - | `rc=-9` (` | num_cache_hits=10543 {98.6%} | num_compilations=32 | peak`) |
| 4 | `wan21_p1_04_fsdp16_splash_save_sa1024_xla_lhs_s32k_ga2` | `FSDP-16` | `SplashSaveAC` | `1024` | `BF16 (BF16 Reduce)` | `32,768` | `2x` | **53.344s** | `0.600` | **1,228.4** | `21.51` | **44.83%** | `2.0701 -> 1.9939` | `OK` |
| 5 | `wan21_p1_05_hsdp_2x8_splash_save_sa1024_xla_lhs_s32k` | `HSDP 2x8` | `SplashSaveAC` | `1024` | `BF16 (BF16 Reduce)` | `32,768` | `1x` | - | - | - | - | - | - | `rc=-9` (`9688 | num_cache_hits=9506 {98.1%} | num_compilations=30 | p`) |
| 6 | `wan21_p1_06_fsdp16_splash_save_sa1024_xla_lhs_s16k` | `FSDP-16` | `SplashSaveAC` | `1024` | `BF16 (BF16 Reduce)` | `16,384` | `1x` | **19.824s** | `1.614` | **1,652.7** | `16.25` | **45.81%** | `2.0711 -> 1.9939` | `OK` |
