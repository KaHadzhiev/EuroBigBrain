#!/bin/bash
# JOE v2 — claim-queue daemon. Pulls jobs from queue/pending.txt, fires, moves to done.
# Properly atomic: each job line = "<script.py> <priority>"
# Priority is just for sort order; daemon picks highest first.
# When pending empties, daemon idles (still alive, ready for new work).
# Kill: pkill -f joe_mac_v2_queue.sh

set -u
HOME_DIR=~/EuroBigBrain
QDIR="$HOME_DIR/queue"
PENDING="$QDIR/pending.txt"
RUNNING="$QDIR/running.txt"
DONE_DIR="$QDIR/done"
PYDIR="$HOME_DIR/python"
RUNS="$HOME_DIR/runs"
LOG="$RUNS/joe_v2.log"
MAX_CONCURRENT=3
MAX_LOAD=5

mkdir -p "$QDIR" "$DONE_DIR" "$RUNS"
touch "$PENDING" "$RUNNING"

echo "[$(date '+%H:%M:%S')] joe_v2 daemon started PID=$$ MAX_CONCURRENT=$MAX_CONCURRENT MAX_LOAD=$MAX_LOAD" >> "$LOG"

CYCLE=0
while true; do
  CYCLE=$((CYCLE+1))
  TS=$(date '+%H:%M:%S')

  LOAD=$(uptime | awk -F'load averages:' '{print $2}' | awk '{print $1}')
  LOAD_INT=$(echo "$LOAD" | awk '{print int($1)}')
  RUNNING_COUNT=$(wc -l < "$RUNNING" 2>/dev/null | tr -d ' ')
  PENDING_COUNT=$(wc -l < "$PENDING" 2>/dev/null | tr -d ' ')

  # Update RUNNING by checking which PIDs are alive
  if [ "$RUNNING_COUNT" -gt 0 ]; then
    NEW_RUNNING=""
    while IFS=' ' read -r pid script; do
      if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        NEW_RUNNING="$NEW_RUNNING$pid $script"$'\n'
      else
        # job finished — move to done
        echo "[$TS done] $script" >> "$LOG"
        echo "$script" >> "$DONE_DIR/done.txt"
      fi
    done < "$RUNNING"
    printf "%s" "$NEW_RUNNING" > "$RUNNING"
    RUNNING_COUNT=$(wc -l < "$RUNNING" 2>/dev/null | tr -d ' ')
  fi

  # Decide
  if [ "$RUNNING_COUNT" -ge "$MAX_CONCURRENT" ]; then
    ACTION="WAIT (running=$RUNNING_COUNT max=$MAX_CONCURRENT)"
  elif [ "$LOAD_INT" -ge "$MAX_LOAD" ]; then
    ACTION="WAIT (load=$LOAD max=$MAX_LOAD)"
  elif [ "$PENDING_COUNT" -le 0 ]; then
    ACTION="IDLE (pending empty)"
  else
    # Pick highest-priority job from pending (sort by 2nd col descending)
    JOB_LINE=$(sort -k2 -nr "$PENDING" | head -1)
    SCRIPT=$(echo "$JOB_LINE" | awk '{print $1}')
    if [ -f "$PYDIR/$SCRIPT" ]; then
      OUT_LOG="$RUNS/qjob_${SCRIPT%.py}_$(date '+%H%M%S').log"
      cd "$HOME_DIR" && caffeinate -i -s nohup python3 "python/$SCRIPT" > "$OUT_LOG" 2>&1 &
      JOB_PID=$!
      echo "$JOB_PID $SCRIPT" >> "$RUNNING"
      # Remove from pending
      grep -vF "$JOB_LINE" "$PENDING" > "$PENDING.tmp" && mv "$PENDING.tmp" "$PENDING"
      ACTION="LAUNCHED $SCRIPT (pid=$JOB_PID)"
    else
      ACTION="SKIP $SCRIPT (file missing)"
      grep -vF "$JOB_LINE" "$PENDING" > "$PENDING.tmp" && mv "$PENDING.tmp" "$PENDING"
    fi
  fi

  echo "[$TS cycle=$CYCLE] load=$LOAD running=$RUNNING_COUNT pending=$PENDING_COUNT action=$ACTION" >> "$LOG"
  sleep 30
done
