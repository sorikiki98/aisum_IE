#!/bin/bash

cd /home/ec2-user/aisum_IE || exit 1
LOG_DIR="/home/ec2-user/aisum_IE/log"

echo "[[SH]$(date '+%Y-%m-%d %H:%M:%S')] distribute_jobs.py 시작" >> "$LOG_DIR/distribute.log"

python3 distribute_jobs.py >> "$LOG_DIR/distribute.log" 2>&1

echo "[[SH]$(date '+%Y-%m-%d %H:%M:%S')] distribute_jobs.py 종료" >> "$LOG_DIR/distribute.log"
