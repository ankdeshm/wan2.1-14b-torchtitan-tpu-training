#!/usr/bin/env python3
"""TPU v5p-16 (2x2x4, 16 chips) TorchTitan + TorchTPU 0.2.0 GA Pretraining Sweep for Wan2.1-T2V-14B (14.28B DiT).

Directly benchmarks PyTorch TorchTitan + TorchTPU on the exact hardware (`16` `v5p` chips, `2x2x4`) and
video resolution (`81x480x832` -> `31,590` patches padded to `S=32,768` tokens/video, plus `S=16,384`)
as Google's JAX MaxDiffusion `AI-Hypercomputer/tpu-recipes/training/v5p/wan2.1-14b/bf16-v5p-16` recipe!
Runs ONLY on idle `tpu-v5p-16-pool` when zero Flint workloads are active on that pool.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time

PEAK_BF16_TFLOPS_PER_CHIP = {"v5p": 459.0, "v6e": 918.0, "v7x": 1153.5}

STEP_RE = re.compile(
    r"step:\s*(?P<step>\d+).*?loss:\s*(?P<loss>[0-9.]+).*?grad_norm:\s*(?P<gn>[0-9.]+)"
    r".*?memory:\s*(?P<mem>[0-9.]+)GiB.*?"
    r"tps:\s*(?P<tps>[0-9,]+).*?tflops:\s*(?P<tflops>[0-9.]+).*?mfu:\s*(?P<mfu>[0-9.]+)%",
    re.IGNORECASE,
)
TIME_RE = re.compile(r"time_end_to_end:\s*(?P<sec>[0-9.]+)s", re.IGNORECASE)


def filter_supported_xla_flags(candidates: list[str]) -> str:
    try:
        import libtpu

        so_dir = Path(libtpu.__file__).parent
        so_files = list(so_dir.glob("*.so"))
        blob = b"".join(p.read_bytes() for p in so_files) if so_files else b""
    except Exception:
        blob = b""
    if not blob:
        return " ".join(candidates)
    kept, dropped = [], []
    for flag in candidates:
        name = flag.lstrip("-").split("=", 1)[0].encode("utf-8")
        if name in blob:
            kept.append(flag)
        else:
            dropped.append(flag)
    print(f"[wan21-sweep] Validated XLA flags: kept={len(kept)}, dropped={dropped}", flush=True)
    return " ".join(kept)


def probe_compile_opt_args(workdir: str) -> list[str]:
    code = (
        "from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig; "
        "c = TPUTrainerConfig(); out = []; "
        "hasattr(c, 'compile') and hasattr(c.compile, 'compile_grad_norm') and out.append('--compile.compile_grad_norm'); "
        "hasattr(c, 'compile') and hasattr(c.compile, 'optimizer_compile') and out.append('--compile.optimizer_compile=separate_graph'); "
        "hasattr(c, 'tpu_config') and hasattr(c.tpu_config, 'compile_grad_norm') and out.append('--tpu_config.compile_grad_norm'); "
        "hasattr(c, 'tpu_config') and hasattr(c.tpu_config, 'compile_optimizer') and out.append('--tpu_config.compile_optimizer'); "
        "print(' '.join(out))"
    )
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = workdir
        res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=25)
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip().split()
    except Exception:
        pass
    return []


def wait_for_master_dns(master_addr: str, timeout_s: int = 300) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            return socket.gethostbyname(master_addr)
        except socket.gaierror:
            time.sleep(2.0)
    return master_addr


def compute_exact_wan21_flops_per_token(seq_len: int, l_txt: int = 512) -> float:
    nparams = 14_284_487_808
    n_layers = 40
    dim = 5120
    return float(6 * nparams + 12 * n_layers * dim * l_txt + 12 * n_layers * dim * seq_len)


def write_summary_files(out_dir: Path, summary: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "# Wan2.1-T2V-14B (`14.28B` 3D Video DiT) — TorchTitan + TorchTPU `v5p-16` Results",
        "",
        f"- **Updated (UTC):** `{summary['updated_utc']}`",
        f"- **Cluster / Nodepool:** `{summary['nodepool']}` (`{summary['topology']}`, `{summary['num_chips']}` TPU `{summary['accel']}` chips)",
        f"- **Model:** `Wan-AI/Wan2.1-T2V-14B-Diffusers` (`14,284,487,808` params, `1,095` keys across 12 safetensors shards)",
        "- **Reference Baseline (`AI-Hypercomputer/tpu-recipes` JAX MaxDiffusion `v5p-16` `81x480x832`):** `~20.13s/step` (`bsz=8` global = `0.5` videos/chip)",
        "",
        "| # | Experiment | Parallelism | AC Policy | `sa_block` | Quantization | Seq `S` | GradAccum | Steady Step (s) | Videos/s (Cluster) | Tok/s/chip | Peak HBM (GiB) | MFU (%) | Loss (`step 1 -> N`) | Status |",
        "|---:|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for idx, exp in enumerate(summary["experiments"], 1):
        if exp["status"] == "ok":
            vids_per_step = (exp["mb_tokens"] * exp["grad_accum"] * summary["num_chips"]) / exp["seq_len"]
            vids_per_s = vids_per_step / max(1e-6, exp["steady_step_s"])
            lines.append(
                f"| {idx} | `{exp['name']}` | `{exp['parallelism']}` | `{exp['ac_mode']}` | `{exp['sa_block']}` | "
                f"`{exp['quant']}` | `{exp['seq_len']:,}` | `{exp['grad_accum']}x` | **{exp['steady_step_s']:.3f}s** | "
                f"`{vids_per_s:.3f}` | **{exp['steady_tps_per_chip']:,.1f}** | `{exp['peak_mem_gib']:.2f}` | "
                f"**{exp['exact_bidir_mfu_pct']:.2f}%** | `{exp['loss_first']:.4f} -> {exp['loss_last']:.4f}` | `OK` |"
            )
        else:
            lines.append(
                f"| {idx} | `{exp['name']}` | `{exp['parallelism']}` | `{exp['ac_mode']}` | `{exp['sa_block']}` | "
                f"`{exp['quant']}` | `{exp['seq_len']:,}` | `{exp['grad_accum']}x` | - | - | - | - | - | - | "
                f"`{exp['status']}` (`{exp.get('error', '')[:60]}`) |"
            )
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accel", default="v5p")
    parser.add_argument("--topology", default="2x2x4")
    parser.add_argument("--nnodes", type=int, default=4)
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--nodepool", default="tpu-v5p-16-pool")
    parser.add_argument("--checkpoint-dir", default="/models/models/Wan-AI/Wan2.1-T2V-14B-Diffusers/transformer")
    parser.add_argument("--results-root", default="/models/wan21-results")
    args = parser.parse_args()

    node_rank = int(os.environ.get("JOB_COMPLETION_INDEX", os.environ.get("NODE_RANK", "0")))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    base_port = int(os.environ.get("MASTER_PORT", "29610"))
    num_chips = args.nnodes * args.nproc_per_node
    peak_tflops = PEAK_BF16_TFLOPS_PER_CHIP[args.accel]

    workdir = Path("/torchtitan-native")
    shutil.copy2(Path("/scripts/wan2_1_torchtitan.py"), workdir / "wan2_1_torchtitan.py")

    chk_env = os.environ.copy()
    chk_env["PYTHONPATH"] = str(workdir)
    chk = subprocess.run(
        [sys.executable, "-c", "import wan2_1_torchtitan; print('IMPORT_OK', wan2_1_torchtitan.__file__)"],
        env=chk_env,
        capture_output=True,
        text=True,
    )
    print(f"[rank={node_rank}] Preflight import: rc={chk.returncode} out={chk.stdout.strip()} err={chk.stderr[-300:]}", flush=True)
    if chk.returncode != 0:
        sys.exit(chk.returncode)

    wait_for_master_dns(master_addr)

    base_xla_flags = filter_supported_xla_flags(
        [
            "--xla_tpu_enable_async_collective_fusion=true",
            "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true",
            "--xla_tpu_enable_async_collective_fusion_multiple_steps=true",
            "--xla_tpu_overlap_compute_collective_tc=true",
            "--xla_enable_async_all_gather=true",
            "--xla_enable_async_collective_permute=true",
        ]
    )
    lhs_xla_flags = filter_supported_xla_flags(
        [
            "--xla_tpu_enable_async_collective_fusion=true",
            "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true",
            "--xla_tpu_enable_async_collective_fusion_multiple_steps=true",
            "--xla_tpu_overlap_compute_collective_tc=true",
            "--xla_enable_async_all_gather=true",
            "--xla_enable_async_collective_permute=true",
            "--xla_tpu_enable_latency_hiding_scheduler=true",
            "--xla_tpu_enable_ag_backward_pipelining=true",
            "--xla_tpu_scoped_vmem_limit_kib=81920",
            "--xla_JF_conv_input_fusion=true",
            "--xla_JF_conv_output_fusion=true",
        ]
    )
    compiled_opt_flags = probe_compile_opt_args(str(workdir))
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.results_root) / "v5p-16-2x2x4_wan21_14b_pass1"

    experiments = [
        {
            "name": "wan21_p1_01_fsdp16_fullac_s32k_81x480x832_pretrained",
            "parallelism": "FSDP-16",
            "dp_rep": 1,
            "dp_shard": -1,
            "ac_mode": "FullAC (recompute attn)",
            "save_splash": False,
            "sa_block": 512,
            "quant": "BF16 (FP32 Reduce)",
            "use_fp8": False,
            "seq_len": 32768,
            "mb_tokens": 32768,
            "grad_accum": 1,
            "reduce_dtype": "float32",
            "steps": 8,
            "load_pretrained": True,
            "xla_flags": base_xla_flags,
            "use_compiled_opt": False,
        },
        {
            "name": "wan21_p1_02_fsdp16_splash_save_ac_s32k_81x480x832_pretrained",
            "parallelism": "FSDP-16",
            "dp_rep": 1,
            "dp_shard": -1,
            "ac_mode": "SplashSaveAC (0% attn recompute)",
            "save_splash": True,
            "sa_block": 512,
            "quant": "BF16 (FP32 Reduce)",
            "use_fp8": False,
            "seq_len": 32768,
            "mb_tokens": 32768,
            "grad_accum": 1,
            "reduce_dtype": "float32",
            "steps": 10,
            "load_pretrained": True,
            "xla_flags": base_xla_flags,
            "use_compiled_opt": False,
        },
        {
            "name": "wan21_p1_03_fsdp16_splash_save_sa1024_xla_lhs_s32k_pretrained",
            "parallelism": "FSDP-16",
            "dp_rep": 1,
            "dp_shard": -1,
            "ac_mode": "SplashSaveAC",
            "save_splash": True,
            "sa_block": 1024,
            "quant": "BF16 (BF16 Reduce)",
            "use_fp8": False,
            "seq_len": 32768,
            "mb_tokens": 32768,
            "grad_accum": 1,
            "reduce_dtype": "bfloat16",
            "steps": 10,
            "load_pretrained": True,
            "xla_flags": lhs_xla_flags,
            "use_compiled_opt": True,
        },
        {
            "name": "wan21_p1_04_fsdp16_splash_save_sa1024_xla_lhs_s32k_ga2",
            "parallelism": "FSDP-16",
            "dp_rep": 1,
            "dp_shard": -1,
            "ac_mode": "SplashSaveAC",
            "save_splash": True,
            "sa_block": 1024,
            "quant": "BF16 (BF16 Reduce)",
            "use_fp8": False,
            "seq_len": 32768,
            "mb_tokens": 32768,
            "grad_accum": 2,
            "reduce_dtype": "bfloat16",
            "steps": 10,
            "load_pretrained": True,
            "xla_flags": lhs_xla_flags,
            "use_compiled_opt": True,
        },
        {
            "name": "wan21_p1_05_hsdp_2x8_splash_save_sa1024_xla_lhs_s32k",
            "parallelism": "HSDP 2x8",
            "dp_rep": 2,
            "dp_shard": 8,
            "ac_mode": "SplashSaveAC",
            "save_splash": True,
            "sa_block": 1024,
            "quant": "BF16 (BF16 Reduce)",
            "use_fp8": False,
            "seq_len": 32768,
            "mb_tokens": 32768,
            "grad_accum": 1,
            "reduce_dtype": "bfloat16",
            "steps": 8,
            "load_pretrained": True,
            "xla_flags": lhs_xla_flags,
            "use_compiled_opt": True,
        },
        {
            "name": "wan21_p1_06_fsdp16_splash_save_sa1024_xla_lhs_s16k",
            "parallelism": "FSDP-16",
            "dp_rep": 1,
            "dp_shard": -1,
            "ac_mode": "SplashSaveAC",
            "save_splash": True,
            "sa_block": 1024,
            "quant": "BF16 (BF16 Reduce)",
            "use_fp8": False,
            "seq_len": 16384,
            "mb_tokens": 32768,
            "grad_accum": 1,
            "reduce_dtype": "bfloat16",
            "steps": 10,
            "load_pretrained": True,
            "xla_flags": lhs_xla_flags,
            "use_compiled_opt": True,
        },
    ]

    summary = {
        "run_timestamp": ts,
        "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "accel": args.accel,
        "topology": args.topology,
        "nodepool": args.nodepool,
        "num_chips": num_chips,
        "peak_bf16_tflops_per_chip": peak_tflops,
        "checkpoint_dir": args.checkpoint_dir,
        "experiments": [],
    }

    for exp_idx, exp in enumerate(experiments):
        port = base_port + exp_idx * 7
        global_batch_tokens = exp["mb_tokens"] * exp["grad_accum"] * num_chips
        cmd = [
            "torchrun",
            f"--nnodes={args.nnodes}",
            f"--nproc_per_node={args.nproc_per_node}",
            f"--node_rank={node_rank}",
            f"--rdzv_backend=c10d",
            f"--rdzv_endpoint={master_addr}:{port}",
            f"--rdzv_id=wan21_{exp_idx}_{port}",
            "wan2_1_torchtitan.py",
            "--module=wan2_1_torchtitan",
            "--config=wan2_1_14b_tpu",
            f"--hf_checkpoint_dir={args.checkpoint_dir}",
            f"--training.max_context_length={exp['seq_len']}",
            f"--training.num_tokens_per_microbatch_per_dp_rank={exp['mb_tokens']}",
            f"--training.num_tokens_per_train_step={global_batch_tokens}",
            f"--training.steps={exp['steps']}",
            f"--training.mixed_precision_reduce={exp['reduce_dtype']}",
            f"--parallelism.data_parallel_replicate_degree={exp['dp_rep']}",
            f"--parallelism.data_parallel_shard_degree={exp['dp_shard']}",
            f"--sa_block={exp['sa_block']}",
        ]
        cmd.append("--save_splash_in_ac" if exp["save_splash"] else "--no-save_splash_in_ac")
        cmd.append("--use_fp8_linear" if exp["use_fp8"] else "--no-use_fp8_linear")
        cmd.append("--load_pretrained_weights" if exp["load_pretrained"] else "--no-load_pretrained_weights")
        if exp["use_compiled_opt"]:
            cmd.extend(compiled_opt_flags)

        env = os.environ.copy()
        env["PYTHONPATH"] = str(workdir)
        env["LIBTPU_INIT_ARGS"] = exp["xla_flags"]
        env["PJRT_DEVICE"] = "TPU"

        print(f"\n{'=' * 88}\n[rank={node_rank}] >>> EXP {exp_idx + 1}/{len(experiments)}: {exp['name']} <<<\n{'=' * 88}", flush=True)
        log_lines: list[str] = []
        step_records: list[dict] = []
        t_start = time.time()
        proc = subprocess.Popen(
            cmd, cwd=str(workdir), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_lines.append(line)
            m = STEP_RE.search(line)
            if m and ("[titan]" in line or "root - INFO" in line):
                step_num = int(m.group("step"))
                if not step_records or step_records[-1]["step"] != step_num:
                    tm = TIME_RE.search(line)
                    step_records.append(
                        {
                            "step": step_num,
                            "loss": float(m.group("loss")),
                            "grad_norm": float(m.group("gn")),
                            "mem_gib": float(m.group("mem")),
                            "tps": float(m.group("tps").replace(",", "")),
                            "tflops": float(m.group("tflops")),
                            "mfu_pct": float(m.group("mfu")),
                            "step_s": float(tm.group("sec")) if tm else 0.0,
                        }
                    )
        rc = proc.wait()
        wall_s = time.time() - t_start

        if node_rank == 0:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{exp['name']}.log").write_text("".join(log_lines))
            steady = step_records[3:] if len(step_records) >= 4 else step_records[-3:]
            if rc == 0 and steady:
                avg_tps = sum(r["tps"] for r in steady) / len(steady)
                avg_step_s = sum(r["step_s"] for r in steady if r["step_s"] > 0) / max(
                    1, len([r for r in steady if r["step_s"] > 0])
                )
                peak_mem = max(r["mem_gib"] for r in step_records)
                exact_flops_tok = compute_exact_wan21_flops_per_token(exp["seq_len"])
                exact_tflops = (avg_tps * exact_flops_tok) / 1e12
                exact_mfu = 100.0 * exact_tflops / peak_tflops
                rec = {
                    **exp,
                    "status": "ok",
                    "wall_s": round(wall_s, 2),
                    "steady_step_s": round(avg_step_s, 4),
                    "steady_tps_per_chip": round(avg_tps, 2),
                    "cluster_tps": round(avg_tps * num_chips, 1),
                    "peak_mem_gib": round(peak_mem, 2),
                    "exact_bidir_tflops_per_chip": round(exact_tflops, 2),
                    "exact_bidir_mfu_pct": round(exact_mfu, 2),
                    "loss_first": step_records[0]["loss"],
                    "loss_last": step_records[-1]["loss"],
                    "steps_recorded": step_records,
                }
            else:
                tail_err = "".join(log_lines[-25:]).strip().replace("\n", " | ")
                rec = {**exp, "status": f"rc={rc}", "wall_s": round(wall_s, 2), "error": tail_err[-400:]}
            summary["experiments"].append(rec)
            summary["updated_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            write_summary_files(out_dir, summary)
        time.sleep(5.0)

    print(f"[rank={node_rank}] ALL {len(experiments)} WAN2.1-14B EXPERIMENTS COMPLETE!", flush=True)


if __name__ == "__main__":
    main()
