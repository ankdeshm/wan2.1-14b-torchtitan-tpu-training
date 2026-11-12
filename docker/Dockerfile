FROM python:3.14

ENV PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates curl libgoogle-perftools-dev && \
    rm -rf /var/lib/apt/lists/*

# 1. Install PyTorch 2.15.0 CPU release wheel first so downstream deps never pull CUDA wheels
RUN pip install "torch==2.15.0" --index-url https://download.pytorch.org/whl/cpu

# 2. Clone google-pytorch/torchtitan branches with TorchTPU support
RUN git clone --depth 1 --branch hf-backend-tpu-20261028-v020-flint-tune \
      https://github.com/google-pytorch/torchtitan.git /torchtitan-hf && \
    git clone --depth 1 --branch alekseyv_flint \
      https://github.com/google-pytorch/torchtitan.git /torchtitan-native

# 3. Install TorchTPU 0.2.0 + JAX/Pallas/Tokamax/libtpu stack + TorchTitan requirements
RUN pip install -r /torchtitan-hf/torchtitan/experiments/tpu/requirements.txt \
      -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && \
    pip install "torch_tpu==0.2.0" --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install -r /torchtitan-hf/requirements.txt && \
    pip install -r /torchtitan-hf/.ci/docker/requirements-flux.txt && \
    pip install "transformers>=5.22.0" safetensors accelerate sentencepiece \
      "huggingface_hub[hf_transfer]" google-cloud-storage && \
    python3 -c "import importlib.metadata as m; print('VERSIONS:', 'torch', m.version('torch'), 'torch_tpu', m.version('torch_tpu'), 'transformers', m.version('transformers'), 'jax', m.version('jax'), 'libtpu', m.version('libtpu'))"

WORKDIR /torchtitan-native

# Enable tcmalloc at runtime (set after all RUN steps)
ENV LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"
