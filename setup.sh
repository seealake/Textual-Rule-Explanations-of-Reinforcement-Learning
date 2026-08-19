#!/usr/bin/env bash
# ============================================================
# Remote-Friendly Setup Script (Linux)
# ============================================================
# Usage:
#   chmod +x setup.sh && ./setup.sh
# Optional env vars:
#   TORCH_CUDA_WHL_INDEX=https://download.pytorch.org/whl/cu130
#   TORCH_NCCL_INDEX=https://download.pytorch.org/whl/cu130
#   HIGHWAY_BLACKWELL_NCCL_VERSION=2.29.7
#   SKIP_CUDA_TORCH=1
# ============================================================

set -euo pipefail

echo "=========================================="
echo "  Project Setup (Linux/Remote)"
echo "=========================================="

if command -v python3 >/dev/null 2>&1; then
	PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
	PYTHON_BIN="python"
else
	echo "Python is not installed. Please install Python 3.10+ first."
	exit 1
fi

GPU_NAME=""
GPU_COMPUTE_CAP=""
KNOWN_PIP_CHECK_MISMATCH=0
BLACKWELL_NCCL_VERSION="${HIGHWAY_BLACKWELL_NCCL_VERSION:-2.29.7}"

if command -v nvidia-smi >/dev/null 2>&1; then
	GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | tr -d '\r')"
	GPU_COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '\r')"
fi

echo "[1/5] Creating virtual environment..."
"${PYTHON_BIN}" -m venv .venv
source .venv/bin/activate
echo "  Virtual environment ready: .venv/"

echo "[2/5] Upgrading pip/setuptools/wheel..."
python -m pip install --upgrade pip setuptools wheel

echo "[3/5] Installing base dependencies..."
python -m pip install -r requirements.txt

echo "[4/5] Configuring PyTorch runtime..."
if command -v nvidia-smi >/dev/null 2>&1 && [ "${SKIP_CUDA_TORCH:-0}" != "1" ]; then
	DEFAULT_TORCH_INDEX="https://download.pytorch.org/whl/cu121"
	if [[ "${GPU_COMPUTE_CAP}" == 12.* ]]; then
		DEFAULT_TORCH_INDEX="https://download.pytorch.org/whl/cu130"
	fi
	TORCH_CUDA_WHL_INDEX="${TORCH_CUDA_WHL_INDEX:-${DEFAULT_TORCH_INDEX}}"
	TORCH_NCCL_INDEX="${TORCH_NCCL_INDEX:-${TORCH_CUDA_WHL_INDEX}}"
	echo "  NVIDIA GPU detected. Trying CUDA torch from ${TORCH_CUDA_WHL_INDEX}"
	if python -m pip install --upgrade torch torchvision torchaudio --index-url "${TORCH_CUDA_WHL_INDEX}"; then
		echo "  CUDA torch installation succeeded."
		if [[ "${GPU_COMPUTE_CAP}" == 12.* ]] && [[ "${TORCH_CUDA_WHL_INDEX}" == *"/cu130" ]]; then
			echo "  Blackwell-class GPU detected (${GPU_NAME}, compute capability ${GPU_COMPUTE_CAP})."
			echo "  Upgrading NCCL runtime to ${BLACKWELL_NCCL_VERSION} for RTX 5090 compatibility."
			python -m pip install --upgrade "nvidia-nccl-cu13==${BLACKWELL_NCCL_VERSION}" --index-url "${TORCH_NCCL_INDEX}"
			KNOWN_PIP_CHECK_MISMATCH=1
		fi
	else
		echo "  CUDA torch installation failed; falling back to default PyPI wheels."
		python -m pip install --upgrade torch torchvision torchaudio
	fi
else
	echo "  GPU not detected or SKIP_CUDA_TORCH=1; using default torch wheels."
	python -m pip install --upgrade torch torchvision torchaudio
fi

echo "[5/5] Verifying installation and key environments..."
python - <<'PY'
import gymnasium as gym
import stable_baselines3
import sklearn, numpy, pandas, matplotlib, seaborn, kneed, yaml, pygame, minigrid, torch
import highway_env  # noqa: F401

print(f"  gymnasium {gym.__version__}")
print(f"  stable-baselines3 {stable_baselines3.__version__}")
print(f"  scikit-learn {sklearn.__version__}")
print(f"  numpy {numpy.__version__}")
print(f"  pandas {pandas.__version__}")
print(f"  matplotlib {matplotlib.__version__}")
print(f"  seaborn {seaborn.__version__}")
print(f"  torch {torch.__version__} (cuda={torch.version.cuda}, available={torch.cuda.is_available()})")

if torch.cuda.is_available():
	print(f"  cuda device {torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")
	x = torch.randn(1024, 1024, device="cuda")
	y = torch.randn(1024, 1024, device="cuda")
	print(f"  CUDA matmul OK ({(x @ y).mean().item():.4f})")

env = gym.make("LunarLander-v3"); env.reset(seed=0); env.close(); print("  LunarLander-v3 OK")
env = gym.make("MiniGrid-Dynamic-Obstacles-8x8-v0"); env.reset(seed=0); env.close(); print("  MiniGrid-Dynamic-Obstacles-8x8-v0 OK")
env = gym.make("merge-v0"); env.reset(seed=0); env.close(); print("  merge-v0 OK")
env = gym.make("intersection-v0"); env.reset(seed=0); env.close(); print("  intersection-v0 OK")
print("  All dependency checks passed.")
PY

PIP_CHECK_OUTPUT="$(python -m pip check || true)"
if [ -z "${PIP_CHECK_OUTPUT}" ]; then
	echo "  pip check OK"
elif [ "${KNOWN_PIP_CHECK_MISMATCH}" = "1" ] && echo "${PIP_CHECK_OUTPUT}" | grep -Fq "torch 2.11.0+cu130 has requirement nvidia-nccl-cu13==2.28.9"; then
	echo "  pip check note: accepting the known torch/NCCL metadata mismatch required for Blackwell GPU support."
else
	printf '%s\n' "${PIP_CHECK_OUTPUT}"
	exit 1
fi

echo
echo "=========================================="
echo "  Setup complete"
echo "  Activate env: source .venv/bin/activate"
echo "=========================================="
