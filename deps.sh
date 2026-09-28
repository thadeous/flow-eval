#!/bin/bash

export MAMBA_FORCE_BUILD=TRUE
export CAUSAL_CONV1D_FORCE_BUILD=TRUE

pip install --no-build-isolation "causal-conv1d @ git+https://github.com/Dao-AILab/causal-conv1d.git"
pip install --no-build-isolation "mamba-ssm @ git+https://github.com/state-spaces/mamba.git"
