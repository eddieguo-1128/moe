#!/bin/bash

# === Load modules ===
module load cuda/12.4.0
module load gcc

# === Set CUDA environment ===
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# === Verify PyTorch CUDA setup ===
echo "[INFO] Verifying PyTorch + CUDA setup..."
python -c 'import torch; print("Torch version:", torch.__version__); print("NCCL version:", torch.cuda.nccl.version())'

# === Clone and build APEX ===
echo "[INFO] Installing NVIDIA Apex..."
git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
  --config-settings "--build-option=--cpp_ext" \
  --config-settings "--build-option=--cuda_ext" ./
cd ..

# === Clone and patch Megatron-LM ===
echo "[INFO] Cloning Megatron-LM (v2.0)..."
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout v2.0

# Check if patch exists before applying
if [ -f ../fmoefy-v2.2.patch ]; then
  echo "[INFO] Checking patch for Megatron-FMoE compatibility..."
  git apply --check ../fmoefy-v2.2.patch && git apply ../fmoefy-v2.2.patch
else
  echo "[WARNING] Patch file 'fmoefy-v2.2.patch' not found!"
fi
cd ..

# === Clone and install FastMoE ===
echo "[INFO] Cloning and building FastMoE..."
git clone https://github.com/laekov/fastmoe.git
cd fastmoe

export NCCL_HOME=$(python -c "import torch; print(torch.__path__[0])")/lib
export CPATH=$NCCL_HOME/include:$CPATH
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH

# Try standard build first; fallback if NCCL fails
echo "[INFO] Installing FastMoE..."
python setup.py install || USE_NCCL=0 python setup.py install
cd ..

echo "[✅ DONE] Megatron + FastMoE environment setup complete."
