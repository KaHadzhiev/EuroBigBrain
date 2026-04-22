#!/bin/bash
# JOE — Mac-side saturation daemon. Runs forever until killed.
# Keeps Mac saturated by firing jobs from rotation when concurrent < 3 AND load < 4.
# Logs to ~/EuroBigBrain/runs/joe_mac.log
# Kill: pkill -f joe_mac_daemon.sh

set -u
RUNS=~/EuroBigBrain/runs
LOG="$RUNS/joe_mac.log"
PYDIR=~/EuroBigBrain/python
MAX_JOBS=3
MAX_LOAD=4

# Job rotation queue — Joe cycles through these
JOBS=(
  "tb_h10_TRUE_walkforward.py"
  "tb_h10_TRUE_threshold_sweep.py"
  "tb_h10_holdout_bootstrap.py"
  "tb_h10_winner_validate.py"
  "tb_multi_instrument.py"
  "tb_h10_dd_montecarlo.py"
)

mkdir -p "$RUNS"
echo "[$(date '+%H:%M:%S')] joe_mac daemon started, PID=$$, MAX_JOBS=$MAX_JOBS, MAX_LOAD=$MAX_LOAD" >> "$LOG"

CYCLE=0
JOB_IDX=0
while true; do
  CYCLE=$((CYCLE+1))
  TS=$(date '+%H:%M:%S')

  # Mac state
  LOAD=$(uptime | awk -F'load averages:' '{print $2}' | awk '{print $1}')
  RUNNING=$(ps -ef | grep python3 | grep -v grep | grep "EuroBigBrain/python" | wc -l | tr -d ' ')

  # Decide
  LOAD_INT=$(echo "$LOAD" | awk '{print int($1)}')
  if [ "$RUNNING" -ge "$MAX_JOBS" ]; then
    ACTION="WAIT (jobs=$RUNNING >= $MAX_JOBS)"
  elif [ "$LOAD_INT" -ge "$MAX_LOAD" ]; then
    ACTION="WAIT (load=$LOAD >= $MAX_LOAD)"
  else
    JOB="${JOBS[$JOB_IDX]}"
    JOB_IDX=$(( (JOB_IDX + 1) % ${#JOBS[@]} ))
    OUT_LOG="$RUNS/joe_${JOB%.py}_${TS//:/}.log"
    if [ -f "$PYDIR/$JOB" ]; then
      cd ~/EuroBigBrain && caffeinate -i -s nohup python3 "python/$JOB" > "$OUT_LOG" 2>&1 &
      ACTION="LAUNCHED $JOB (pid=$!)"
    else
      ACTION="SKIP $JOB (file not found)"
    fi
  fi

  echo "[$TS cycle=$CYCLE] load=$LOAD jobs=$RUNNING action=$ACTION" >> "$LOG"

  sleep 60
done
