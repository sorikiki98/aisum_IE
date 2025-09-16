#!/bin/bash

TASK_DIR="/home/ec2-user/aisum_IE/task"
INDEX_SCRIPT="/home/ec2-user/aisum_IE/index_builder.py"
LAST_RUN_FILE="/home/ec2-user/aisum_IE/.last_index_run"
LOG_DIR="/home/ec2-user/aisum_IE/log"
BLOCK_FILE="/home/ec2-user/aisum_IE/.index_block" # 실행 차단 플래그

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_DIR/index.log"
}

# 실행 가능 여부 함수
can_run() {
  if [ -f "$BLOCK_FILE" ]; then
    log "실행 차단 플래그 감지됨 → 실행 중단"
    return 1
  fi

  if pgrep -f "/home/ec2-user/aisum_IE/index_builder.py" > /dev/null; then
    log "index_builder.py 실행 중 → 중복 방지로 실행 중단"
    exit 0
  fi

  return 0
}

main() {
  if ! can_run; then
    exit 0
  fi

  # 마지막 실행 시각 불러오기
  if [ -f "$LAST_RUN_FILE" ]; then
    LAST_RUN_TIME=$(cat "$LAST_RUN_FILE")
  else
    LAST_RUN_TIME=0
  fi

  # 새 파일 있는지 확인
  NEW_FILES=$(find "$TASK_DIR" -type f -newermt "@$LAST_RUN_TIME")

  if [ -n "$NEW_FILES" ]; then
    # 찾은 새로운 파일 목록을 하나씩 반복 처리
    echo "$NEW_FILES" | while read -r TASK_FILE; do
      log "새 파일 감지됨 ($TASK_FILE) → index_builder.py 실행"
      python3 "$INDEX_SCRIPT" "$TASK_FILE" "dreamsim" >> "$LOG_DIR/index.log" 2>&1
    done
    
    # 모든 작업이 끝난 후 마지막 실행 시간을 한 번만 기록
    date +%s > "$LAST_RUN_FILE" 
  else
    log "새 파일 없음 → 실행 안함"
  fi
}

main
