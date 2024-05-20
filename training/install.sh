#!/usr/bin/env bash

pip install --upgrade pip
pip install "numpy>=1.17.3,<1.25.0"
pip install --upgrade --force-reinstall --no-cache-dir torch==2.1.0 triton --index-url https://download.pytorch.org/whl/cu121
pip install "unsloth[cu121-ampere] @ git+https://github.com/unslothai/unsloth.git"
pip install wandb