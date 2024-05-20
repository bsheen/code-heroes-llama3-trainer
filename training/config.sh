#!/usr/bin/env bash

echo "Configuring ssh daemon..."

echo "AcceptEnv HF_TOKEN" >> /etc/ssh/sshd_config
echo "AcceptEnv HF_ACCOUNT" >> /etc/ssh/sshd_config
echo "AcceptEnv HF_DATASET" >> /etc/ssh/sshd_config
echo "AcceptEnv HF_INPUT_MODEL" >> /etc/ssh/sshd_config
echo "AcceptEnv HF_OUTPUT_MODEL" >> /etc/ssh/sshd_config

echo "AcceptEnv WANDB_API_KEY" >> /etc/ssh/sshd_config
echo "AcceptEnv WANDB_PROJECT" >> /etc/ssh/sshd_config
echo "AcceptEnv WANDB_NAME" >> /etc/ssh/sshd_config

echo "Restarting ssh daemon..."
systemctl restart ssh.service