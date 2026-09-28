#!/bin/bash

export MAMBA_FORCE_BUILD=TRUE
export CAUSAL_CONV1D_FORCE_BUILD=TRUE

CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install --no-build-isolation "causal-conv1d @ git+https://github.com/Dao-AILab/causal-conv1d.git"
MAMBA_FORCE_BUILD=TRUE pip install --no-build-isolation "mamba-ssm[causal-conv1d] @ git+https://github.com/state-spaces/mamba.git"
