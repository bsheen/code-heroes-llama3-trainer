#!/usr/bin/env bash

$HOME/.local/bin/huggingface-cli login --token $HF_TOKEN
$HOME/.local/bin/wandb login $WANDB_API_KEY