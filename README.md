# Wan2.1-T2V-14B Training on Google Cloud TPU v5p

Distributed pretraining and fine-tuning benchmarks for [Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers) on Google Cloud TPU `v5p-16` with PyTorch, TorchTitan, and TorchTPU.

**Summary:** With selective Splash Attention checkpointing (`SplashSaveAC`), `1,024`-token Pallas block sizes, XLA latency-hiding scheduling, and `bfloat16` gradient reduction on `v5p-16` (`16` chips), **Wan2.1-T2V-14B reaches 0.600 videos/s (`44.83%` MFU, `21.51 GiB` peak HBM/chip) on `81×480×832` (`32,768`-token) sequences — 1.27× higher throughput than our initial `FullAC` TorchTitan TPU baseline (`0.473` videos/s) and 1.51× higher cluster throughput than the published `AI-Hypercomputer/tpu-recipes` JAX MaxDiffusion `v5p-16` reference (`0.397` videos/s at global batch 8).** On shorter `41×480×832` (`16,384`-token) sequences, the same stack reaches **1.614 videos/s (`45.81%` MFU, `16.25 GiB` peak HBM/chip)**.

---

## 1. Model, Framework, and Hardware

- **Model:** [`Wan-AI/Wan2.1-T2V-14B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers) is an open-source 3D rectified-flow Diffusion Transformer (DiT) for text-to-video generation.
  - **Parameters & Weights:** **14.28B parameters** (`14,284,487,808` safetensors parameters across 1,095 tensors in 12 shards; `14,288,491,584` total module parameters). Pretrained `fp32` safetensors weights (`57.15 GB`) are streamed on CPU directly into `bfloat16` FSDP2 `DTensor` shards; training uses `bfloat16` parameters/activations with `fp32` or `bfloat16` gradient ReduceScatter and `fp32` AdamW state.
  - **Architecture:** 40 DiT blocks (`dim = 5,120`, `40` attention heads × `128` head dim, `13,824` GELU-tanh FFN width). Each block applies bidirectional 3D video self-attention (`attn1`) with across-head RMSNorm QK-norm and 3D axial RoPE `(44, 42, 42)`, followed by cross-attention (`attn2`) over 512 UMT5 text tokens (`4,096` width) and an AdaLN-modulated MLP.
- **Framework:** Python 3.14, PyTorch 2.15.0, `torch_tpu` 0.2.0, `libtpu` 0.0.50, `TorchTitan` (`FSDP2` + `torch.compile(backend="tpu")`), and bidirectional Pallas `SplashAttention(is_causal=False)` ([`docker/Dockerfile`](./docker/Dockerfile)).
- **Hardware:** Google Cloud TPU `v5p-16` slice on GKE (`2x2x4` 3D torus ICI topology: 4 hosts × 4 chips/host = 16 chips, `95 GiB` HBM2e and `459` peak `bf16` TFLOP/s per chip, `1,520 GiB` total HBM).

---

## 2. Main Results

### Distributed DiT Training on `v5p-16` (`16` TPU v5p Chips)

| Video Workload (`Latent Sequence S`) | Configuration | Global Batch (`Videos/Step`) | Steady Step Time | Cluster Throughput (`Tok/s/chip`) | MFU | Peak HBM / Chip | Comparison |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `81×480×832` (`S = 32,768`) | JAX MaxDiffusion reference (`tpu-recipes`, published) | 8 (`0.5`/chip) | ~20.13 s | 0.397 videos/s | — | — | Published `v5p-16` `bf16` reference (`bsz = 8`) |
| `81×480×832` (`S = 32,768`) | Initial TorchTitan TPU baseline (`FSDP-16`, `FullAC`, `sa=512`, `fp32` reduce) | 16 (`1`/chip) | 33.82 s | 0.473 videos/s (`969.0`) | 35.36% | 25.77 GiB | Initial TPU baseline (`1.00×`); `1.19×` vs. published JAX reference |
| **`81×480×832` (`S = 32,768`)** | **Optimized TorchTitan TPU (`FSDP-16`, `SplashSaveAC`, `sa=1024`, XLA LHS, `bf16` reduce, `2×` grad accum)** | **32 (`2`/chip)** | **53.34 s** | **0.600 videos/s (`1,228.4`)** | **44.83%** | **21.51 GiB** | **1.27×** (`+9.47 pp` MFU) vs. initial TPU baseline; **1.51×** vs. published JAX reference |
| **`41×480×832` (`S = 16,384`)** | **Optimized TorchTitan TPU (`FSDP-16`, `SplashSaveAC`, `sa=1024`, XLA LHS, `bf16` reduce)** | **32 (`2`/chip)** | **19.82 s** | **1.614 videos/s (`1,652.7`)** | **45.81%** | **16.25 GiB** | **2.69×** higher videos/s than `32k`-token workload (`O(S²)` self-attention scales down) |

*Notes:* Steady-state metrics average steps 4 through 8–10 after XLA compilation (step 1 compiles in `69–297 s` depending on flags). Exact bidirectional DiT FLOPs per token (`6·P + 12·L·D·L_txt + 12·L·D·S`) are normalized against `459` `bf16` TFLOP/s/chip. The JAX MaxDiffusion figure (`~20.13 s/step` at global batch `8`) is cited from [`AI-Hypercomputer/tpu-recipes`](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/v5p/wan2.1-14b); our optimized `32k`-token run uses global batch `32` (`1` microbatch × `2` gradient accumulation steps × `16` chips) with pre-encoded in-memory latent tensors and requires zero Ulysses/Ring context parallelism (`21.51 GiB` of `95 GiB` HBM used per chip).

---

## 3. Reproducing the Main Results

Build the training image from [`docker/Dockerfile`](./docker/Dockerfile) and stage [`Wan-AI/Wan2.1-T2V-14B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers) to GCS using [`k8s/storage-gcsfuse.yaml`](./k8s/storage-gcsfuse.yaml) and [`k8s/stage-wan21-14b-job.yaml`](./k8s/stage-wan21-14b-job.yaml):

```bash
kubectl apply -f k8s/storage-gcsfuse.yaml
kubectl create configmap stage-hf-script --from-file=stage_hf_to_gcs.py=scripts/stage_hf_to_gcs.py
kubectl apply -f k8s/stage-wan21-14b-job.yaml
```

### Run the Full `v5p-16` Sweep on GKE
Scripts: [`scripts/wan2_1_torchtitan.py`](./scripts/wan2_1_torchtitan.py) and [`scripts/run_wan21_v5p16_sweep.py`](./scripts/run_wan21_v5p16_sweep.py); manifest: [`k8s/v5p-16-wan21-sweep.yaml`](./k8s/v5p-16-wan21-sweep.yaml):

```bash
kubectl create configmap wan21-v5p16-script \
  --from-file=wan2_1_torchtitan.py=scripts/wan2_1_torchtitan.py \
  --from-file=run_wan21_v5p16_sweep.py=scripts/run_wan21_v5p16_sweep.py
kubectl apply -f k8s/v5p-16-wan21-sweep.yaml
```

### Run the Single Best `81×480×832` (`S = 32,768`) Configuration Directly
Execute on each of the 4 hosts (`NODE_RANK=0..3`) in a `v5p-16` (`2x2x4`) slice:

```bash
export PJRT_DEVICE=TPU
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true --xla_enable_async_all_reduce=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_ici_ag_pipelining=true --xla_tpu_overlap_compute_collective_tc=true --xla_max_concurrent_async_all_gathers=4 --xla_latency_hiding_scheduler_rerun=2 --xla_tpu_scoped_vmem_limit_kib=81920"

torchrun --nnodes=4 --nproc_per_node=4 --rdzv_backend=static \
  --rdzv_endpoint="${MASTER_ADDR}:29610" --node_rank="${NODE_RANK}" \
  -m wan2_1_torchtitan --module=wan2_1_torchtitan --config=wan2_1_14b_tpu \
  --hf_checkpoint_dir=/models/models/Wan-AI/Wan2.1-T2V-14B-Diffusers/transformer \
  --training.max_context_length=32768 \
  --training.num_tokens_per_microbatch_per_dp_rank=32768 \
  --training.num_tokens_per_train_step=1048576 \
  --training.steps=10 \
  --training.mixed_precision_reduce=bfloat16 \
  --parallelism.data_parallel_replicate_degree=1 \
  --parallelism.data_parallel_shard_degree=-1 \
  --sa_block=1024 --save_splash_in_ac --load_pretrained_weights \
  --compile.compile_grad_norm --compile.optimizer_compile=separate_graph
```

---

## 4. Detailed Ablations, Engineering Notes, and Raw Logs

- **[`APPENDIX.md`](./APPENDIX.md):** Complete 6-run `v5p-16` optimization ladder, TorchTitan DiT porting and `SplashSaveAC` design, exact FLOP/MFU derivation, `HSDP 2×8` and `1×` gradient accumulation teardown notes, and comparison caveats against JAX MaxDiffusion.
- **[`results/v5p-16/`](./results/v5p-16/):** Structured metrics ([`summary.json`](./results/v5p-16/summary.json)) and raw per-experiment logs (`*.log`).
