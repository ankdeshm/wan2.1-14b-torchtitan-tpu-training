# `Wan2.1-T2V-14B` (`14.28B` DiT) TorchTitan Training on TPU `v5p-16`

**PyTorch [`TorchTitan`](https://github.com/pytorch/torchtitan) + `TorchTPU 0.2.0` GA** distributed Rectified-Flow video diffusion pre-training and fine-tuning for **[`Wan-AI/Wan2.1-T2V-14B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers)** (`14,284,487,808` parameters, `1,095 / 1,095` safetensors tensors across `12` shards) on **`16` TPU `v5p` chips (`2x2x4`, `1,520 GiB` HBM)** — evaluated on `81×480×832` (`S = 32,768` video latent tokens) matching the hardware and sequence configuration of `tpu-recipes` MaxDiffusion (`~20.13s/step` at `bsz=8`).

- **Sweep Summary & Logs:** [`results/v5p-16-2x2x4_wan21_14b_pass1/SUMMARY.md`](./results/v5p-16-2x2x4_wan21_14b_pass1/SUMMARY.md) · [`summary.json`](./results/v5p-16-2x2x4_wan21_14b_pass1/summary.json)

---

## Key Results

| Configuration | Run ID | Code / Config Change Over Baseline | Steady Step (`Videos/s`) | **MFU (%)** | Peak HBM | Script & Line Anchor | K8s Manifest | Result Files |
|---|---|---|---|---:|---:|---|---|---|
| **Baseline (`FullAC`, `S=32K`)** | `wan21_p1_01_fsdp16_fullac_s32k_81x480x832_pretrained` | `FSDP-16` + `FullAC` (recompute attn) + `Splash sa=512` + `FP32` Reduce (`16` vids/step) | `33.820s` (`0.473` vid/s) | **`35.36%`** | `25.77 GiB` | [`run_wan21_v5p16_sweep.py#L190`](./scripts/run_wan21_v5p16_sweep.py#L190-L208) | [`v5p-16-wan21-sweep.yaml`](./k8s/v5p-16-wan21-sweep.yaml) | [`MD`](./results/v5p-16-2x2x4_wan21_14b_pass1/SUMMARY.md) · [`Log`](./results/v5p-16-2x2x4_wan21_14b_pass1/wan21_p1_01_fsdp16_fullac_s32k_81x480x832_pretrained.log) |
| **`+SplashSaveAC` (`+2.31 pp`)** | `wan21_p1_02_fsdp16_splash_save_ac_s32k_81x480x832_pretrained` | Save `pallas::splash_attn_fwd` residuals ([`_full_ac_save_splash_policy#L555`](./scripts/wan2_1_torchtitan.py#L555-L560)) — avoids `O(S²)` self-attn recompute across `28/40` blocks | `31.747s` (`0.504` vid/s) | **`37.67%`** | `31.51 GiB` | [`run_wan21_v5p16_sweep.py#L209`](./scripts/run_wan21_v5p16_sweep.py#L209-L227) | [`v5p-16-wan21-sweep.yaml`](./k8s/v5p-16-wan21-sweep.yaml) | [`MD`](./results/v5p-16-2x2x4_wan21_14b_pass1/SUMMARY.md) · [`Log`](./results/v5p-16-2x2x4_wan21_14b_pass1/wan21_p1_02_fsdp16_splash_save_ac_s32k_81x480x832_pretrained.log) |
| **Optimized `S=32K` (`81×480×832`, `+9.47 pp`)** | `wan21_p1_04_fsdp16_splash_save_sa1024_xla_lhs_s32k_ga2` | Widen Splash block `512 -> 1024` + `v5p` XLA Latency-Hiding Scheduler (`LHS`) + `BF16` ReduceScatter + `2×` GradAccum (`32` vids/step) + Compiled Opt | **`53.344s`** (`0.600` vid/s = **`1.67s/vid`**) | **`44.83%`** | **`21.51 GiB`** | [`run_wan21_v5p16_sweep.py#L247`](./scripts/run_wan21_v5p16_sweep.py#L247-L265) | [`v5p-16-wan21-sweep.yaml`](./k8s/v5p-16-wan21-sweep.yaml) | [`MD`](./results/v5p-16-2x2x4_wan21_14b_pass1/SUMMARY.md) · [`JSON`](./results/v5p-16-2x2x4_wan21_14b_pass1/summary.json) · [`Log`](./results/v5p-16-2x2x4_wan21_14b_pass1/wan21_p1_04_fsdp16_splash_save_sa1024_xla_lhs_s32k_ga2.log) |
| **Optimized `S=16K` (`41×480×832`)** | `wan21_p1_06_fsdp16_splash_save_sa1024_xla_lhs_s16k` | Same `SplashSaveAC + sa=1024 + XLA LHS + BF16 Reduce` stack on `41×480×832` (`S = 16,384`, `bsz=2`/chip = `32` vids/step) | **`19.824s`** (**`1.614` vid/s**) | **`45.81%`** | **`16.25 GiB`** | [`run_wan21_v5p16_sweep.py#L285`](./scripts/run_wan21_v5p16_sweep.py#L285-L305) | [`v5p-16-wan21-sweep.yaml`](./k8s/v5p-16-wan21-sweep.yaml) | [`MD`](./results/v5p-16-2x2x4_wan21_14b_pass1/SUMMARY.md) · [`JSON`](./results/v5p-16-2x2x4_wan21_14b_pass1/summary.json) · [`Log`](./results/v5p-16-2x2x4_wan21_14b_pass1/wan21_p1_06_fsdp16_splash_save_sa1024_xla_lhs_s16k.log) |

> **Comparison vs. Reference Baseline:** At `81×480×832` (`S = 32,768`), the PyTorch `TorchTitan + TorchTPU` configuration (`0.600` videos/s = `1.67s/video` at `44.83%` MFU) achieves `1.51×` the video training throughput on the same `v5p-16` (`2x2x4`) slice compared to the `AI-Hypercomputer/tpu-recipes` JAX MaxDiffusion baseline (`~20.13s` per `bsz=8` step = `0.397` videos/s = `2.52s/video`).

---

## 1. Architecture & TPU MXU Alignment (`1,095 / 1,095` Safetensors Keys)

| Property | Implementation & Code Anchor |
|---|---|
| **Hugging Face Checkpoint** | [`Wan-AI/Wan2.1-T2V-14B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers) (`12` shards, `57.15 GB` FP32 -> streamed directly to `BF16` `FSDP2` `DTensor` shards in [`load_pretrained_wan21_safetensors#L660`](./scripts/wan2_1_torchtitan.py#L660-L725)) |
| **Hidden / Heads / FFN** | `dim = 5,120` (`40 × 128`), `40` heads × `head_dim = 128`, `ffn_dim = 13,824` (`108 × 128` GELU-tanh MLP), `40` [`WanTransformerBlock`s (`#L406`)](./scripts/wan2_1_torchtitan.py#L406-L458) inside [`Wan21Model` (`#L459`)](./scripts/wan2_1_torchtitan.py#L459-L548) |
| **Attention Kernels** | [`WanSelfAttention` (`#L263`)](./scripts/wan2_1_torchtitan.py#L263-L298): Bidirectional Pallas **`SplashAttention(is_causal=False)` (`FullMask`)** + 3D Axial RoPE `(44, 42, 42)` ([`#L149`](./scripts/wan2_1_torchtitan.py#L149-L195)) · [`WanCrossAttention` (`#L299`)](./scripts/wan2_1_torchtitan.py#L299-L352): Chunked MXU `bmm` (`Q_CHUNK=2048`) |
| **Activation Checkpointing** | [`_full_ac_save_splash_policy` (`#L555`)](./scripts/wan2_1_torchtitan.py#L555-L560) & [`parallelize_wan21` (`#L561`)](./scripts/wan2_1_torchtitan.py#L561-L655) — saves `pallas::splash_attn_fwd` outputs & logsumexp residuals so backward performs `0%` `O(S^2)` self-attention recomputation |
| **16-Chip `v5p` HBM Footprint** | `FSDP-16` persistent state = **`~14.2 GiB/chip`** (`21.51 GiB` peak live HBM at `S = 32,768` with `SplashSaveAC + sa=1024 + XLA LHS`), requiring zero Ulysses/Ring Context Parallelism ICI communication |

---

## 2. Repository Structure & Quickstart

- [`docker/Dockerfile`](./docker/Dockerfile) — Python 3.14 + PyTorch 2.15 + TorchTPU `0.2.0` GA + TorchTitan
- [`scripts/wan2_1_torchtitan.py`](./scripts/wan2_1_torchtitan.py) — Native TorchTitan `Wan21Model` + [`Wan21Trainer` (`#L818`)](./scripts/wan2_1_torchtitan.py#L818-L945) + `SplashAttention` + 12-shard loader
- [`scripts/run_wan21_v5p16_sweep.py`](./scripts/run_wan21_v5p16_sweep.py) — 6-experiment `v5p-16` MFU, `SplashSaveAC`, `sa=1024`, `XLA LHS`, `GradAccum` & `HSDP 2x8` sweep
- [`scripts/stage_hf_to_gcs.py`](./scripts/stage_hf_to_gcs.py) — Parallel Hugging Face -> GCS safetensors stager
- [`k8s/stage-wan21-14b-job.yaml`](./k8s/stage-wan21-14b-job.yaml) & [`k8s/storage-gcsfuse.yaml`](./k8s/storage-gcsfuse.yaml) — Checkpoint staging & GCSFuse manifests
- [`k8s/v5p-16-wan21-sweep.yaml`](./k8s/v5p-16-wan21-sweep.yaml) — 4-host (`16` `v5p` chips, `2x2x4`) Indexed Job manifest
- [`results/v5p-16-2x2x4_wan21_14b_pass1/`](./results/v5p-16-2x2x4_wan21_14b_pass1/SUMMARY.md) — Empirical `SUMMARY.md`, `summary.json`, and all 6 `.log` files

```bash
# 1. Stage Wan-AI/Wan2.1-T2V-14B-Diffusers to GCS
kubectl apply -f k8s/stage-wan21-14b-job.yaml

# 2. Create ConfigMap & Launch 4-Host (16-Chip v5p 2x2x4) TorchTitan Sweep
kubectl create configmap wan21-v5p16-script   --from-file=wan2_1_torchtitan.py=scripts/wan2_1_torchtitan.py   --from-file=run_wan21_v5p16_sweep.py=scripts/run_wan21_v5p16_sweep.py
kubectl apply -f k8s/v5p-16-wan21-sweep.yaml
```
