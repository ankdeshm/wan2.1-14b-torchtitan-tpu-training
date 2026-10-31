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

## 2. Repository Structure & Quickstart

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
