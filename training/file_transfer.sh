#!/usr/bin/env bash

scp -i $SSH_PUB_KEY -o StrictHostKeyChecking=no ./training/config.sh ubuntu@$HOST:/home/ubuntu/config.sh
ssh -i $SSH_PUB_KEY -o StrictHostKeyChecking=no ubuntu@$HOST chmod +x /home/ubuntu/config.sh
scp -i $SSH_PUB_KEY -o StrictHostKeyChecking=no ./training/install.sh ubuntu@$HOST:/home/ubuntu/install.sh
ssh -i $SSH_PUB_KEY -o StrictHostKeyChecking=no ubuntu@$HOST chmod +x /home/ubuntu/install.sh
scp -i $SSH_PUB_KEY -o StrictHostKeyChecking=no ./training/authenticate.sh ubuntu@$HOST:/home/ubuntu/authenticate.sh
ssh -i $SSH_PUB_KEY -o StrictHostKeyChecking=no ubuntu@$HOST chmod +x /home/ubuntu/authenticate.sh
scp -i $SSH_PUB_KEY -o StrictHostKeyChecking=no ./training/train.py ubuntu@$HOST:/home/ubuntu/train.py
ssh -i $SSH_PUB_KEY -o StrictHostKeyChecking=no ubuntu@$HOST chmod +x /home/ubuntu/train.py