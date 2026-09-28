import torch
from causal_conv1d import causal_conv1d_fn

import mamba_ssm.ops.triton.selective_state_update
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create dummy inputs: batch=2, dim=16, seqlen=32
batch, dim, seqlen = 2, 16, 32
width = 4  # kernel size

x = torch.randn(batch, dim, seqlen, device=device, dtype=torch.float16)
weight = torch.randn(dim, width, device=device, dtype=torch.float16)
bias = torch.randn(dim, device=device, dtype=torch.float16)

try:
    # Test the optimized CUDA/Triton implementation
    out = causal_conv1d_fn(x, weight, bias=bias, activation="silu")
    print("CUDA forward pass successful!")
    print("Output shape:", out.shape)
except Exception as e:
    print("CUDA forward pass failed:", e)


