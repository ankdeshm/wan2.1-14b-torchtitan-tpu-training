# `Wan-AI/Wan2.1-T2V-14B` (`14.28B` 3D Video DiT) Pre-Training & MFU Optimization on Cloud TPU `v5p-16` with PyTorch TorchTitan + TorchTPU `0.2.0` GA

Native **PyTorch [`TorchTitan`](https://github.com/pytorch/torchtitan) + `TorchTPU 0.2.0` GA** distributed Rectified-Flow video diffusion pre-training and fine-tuning for **[`Wan-AI/Wan2.1-T2V-14B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers)** (`14,284,487,808` parameters, `1,095` safetensors tensors across 12 shards) on **`16` TPU `v5p` chips (`2x2x4`, `1,520 GiB` HBM)** — matching the exact hardware and `81×480×832` (`S = 32,768` video latent tokens) benchmark setup of Google's JAX MaxDiffusion `tpu-recipes` baseline (`~20.13s/step` at `bsz=8`), implemented 100% in native PyTorch.

---

## 1. Architecture & TPU MXU Alignment (`1,095 / 1,095` Exact Safetensors Keys)

| Property | Value |
|---|---|
| **Hugging Face Checkpoint** | [`Wan-AI/Wan2.1-T2V-14B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers) (`12` shards, `57.15 GB` FP32 -> streamed directly to `BF16` FSDP2 `DTensor` shards) |
| **Hidden / Heads / FFN** | `dim = 5,120` (`40 × 128`), `40` heads × `head_dim = 128`, `ffn_dim = 13,824` (`108 × 128` GELU-tanh MLP), `40` `WanTransformerBlock`s |
| **Attention Kernels** | `attn1` (3D Video Self-Attention): Bidirectional Pallas **`SplashAttention(is_causal=False)` (`FullMask`)** + 3D Axial RoPE `(44, 42, 42)` + across-heads `RMSNorm(5120)` · `attn2` (`UMT5` `L_txt=512` Cross-Attention): Chunked MXU `bmm` (`Q_CHUNK=2048`) |
| **Activation Checkpointing** | **`SplashSaveAC` (`_full_ac_save_splash_policy`)** — saves `pallas::splash_attn_fwd` outputs & logsumexp residuals so backward performs **0% `O(S^2)` self-attention recomputation** |
| **16-Chip `v5p` HBM Footprint** | `FSDP-16` persistent state = **`~14.2 GiB/chip`** (`1.66 GiB` BF16 params + grads + `9.98 GiB` FP32 AdamW), leaving **`~80.8 GiB/chip`** free for `S = 32,768` (`81×480×832`) single-chip full-sequence video training (zero Ulysses/Ring CP communication overhead!) |

---

## 2. Verified TPU `v5p-16` (`2x2x4`, `16` Chips) Benchmark Results (`results/SUMMARY.md`)

Measured on `16` TPU `v5p` chips (`459.0` BF16 TFLOP/s/chip peak, `1,520 GiB` aggregate HBM) loading all `12` pretrained `Wan-AI/Wan2.1-T2V-14B-Diffusers` safetensors shards (`1,095 / 1,095` keys):

| # | Experiment | Parallelism | AC Policy | `sa_block` | Seq `S` (`T×H×W` Pixels) | GradAccum | Steady Step (s) | Videos/s (Slice) | Tok/s/chip | Peak HBM (GiB) | **MFU (%)** | Loss (`step 1 -> N`) |
|---:|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `wan21_p1_06_fsdp16_splash_save_sa1024_xla_lhs_s16k` | `FSDP-16` | `SplashSaveAC` (`0%` attn recompute) | `1024` | `16,384` (`41×480×832`) | `1×` | **`19.824s`** | **`1.614`** | **`1,652.7`** | `16.25` | **`45.81%`** | `2.0711 -> 1.9939` |
| 2 | `wan21_p1_04_fsdp16_splash_save_sa1024_xla_lhs_s32k_ga2` | `FSDP-16` | `SplashSaveAC` (`0%` attn recompute) | `1024` | `32,768` (`81×480×832`) | `2×` | **`53.344s`** (`32` vids) | **`0.600`** (`1.67s`/vid) | **`1,228.4`** | `21.51` | **`44.83%`** | `2.0701 -> 1.9939` |
| 3 | `wan21_p1_02_fsdp16_splash_save_ac_s32k_81x480x832_pretrained` | `FSDP-16` | `SplashSaveAC` (`28/40` blocks saved) | `512` | `32,768` (`81×480×832`) | `1×` | **`31.747s`** (`16` vids) | **`0.504`** (`1.98s`/vid) | **`1,032.3`** | `31.51` | **`37.67%`** | `2.0717 -> 1.9947` |
| 4 | `wan21_p1_01_fsdp16_fullac_s32k_81x480x832_pretrained` | `FSDP-16` | `FullAC` (recompute attn) | `512` | `32,768` (`81×480×832`) | `1×` | **`33.820s`** (`16` vids) | **`0.473`** (`2.11s`/vid) | **`969.0`** | `25.77` | **`35.36%`** | `2.0717 -> 2.0095` |

> **Comparison vs. Reference Baseline:** At full `81×480×832` (`S = 32,768`), our pure-PyTorch `TorchTitan + TorchTPU` recipe (`0.600` videos/s = **`1.67s/video`** at **`44.83%` MFU**) achieves **`1.51×` higher video training throughput** on the exact same `v5p-16` (`2x2x4`) slice than the published `AI-Hypercomputer/tpu-recipes` JAX MaxDiffusion baseline (`~20.13s` per `bsz=8` step = `0.397` videos/s = `2.52s/video`).

---

## 3. Repository Structure & Quickstart

```text
├── README.md
├── docker/
│   └── Dockerfile                         # Python 3.14 + PyTorch 2.15 + TorchTPU 0.2.0 GA + TorchTitan
├── scripts/
│   ├── wan2_1_torchtitan.py               # Native TorchTitan BaseModel + Wan21Trainer + SplashAttention + 12-shard loader
│   ├── run_wan21_v5p16_sweep.py           # 6-experiment v5p-16 MFU, AC, Splash block, GradAccum & HSDP 2x8 sweep
│   └── stage_hf_to_gcs.py                 # Parallel Hugging Face -> GCS safetensors stager
├── k8s/
│   ├── storage-gcsfuse.yaml               # GKE GCS Fuse PV/PVC manifest
│   ├── stage-wan21-14b-job.yaml           # K8s Job to stream Wan2.1-T2V-14B-Diffusers to GCS
│   └── v5p-16-wan21-sweep.yaml            # 4-host (16 v5p chips, 2x2x4) Indexed Job manifest
└── results/                               # Auto-synced JSON & Markdown benchmark outputs
```

```bash
# 1. Stage Wan-AI/Wan2.1-T2V-14B-Diffusers to GCS
kubectl apply -f k8s/stage-wan21-14b-job.yaml

# 2. Create ConfigMap & Launch 4-Host (16-Chip v5p 2x2x4) TorchTitan Sweep
kubectl create configmap wan21-v5p16-script \
  --from-file=wan2_1_torchtitan.py=scripts/wan2_1_torchtitan.py \
  --from-file=run_wan21_v5p16_sweep.py=scripts/run_wan21_v5p16_sweep.py
kubectl apply -f k8s/v5p-16-wan21-sweep.yaml
```
