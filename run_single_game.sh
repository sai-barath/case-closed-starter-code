#!/bin/bash

set -e

AGENT1=${1:-./steamroller0/steamroller}
AGENT2=${2:-./prev/v2}
PORT1=10001
PORT2=10002

cleanup() {
    echo "Cleaning up..."
    if [ ! -z "$PID1" ]; then kill -9 $PID1 2>/dev/null || true; fi
    if [ ! -z "$PID2" ]; then kill -9 $PID2 2>/dev/null || true; fi
    exit 0
}

trap cleanup EXIT INT TERM

PORT=$PORT1 $AGENT1 &>/dev/null &
PID1=$!
echo "Started agent1 (PID $PID1) on port $PORT1"

PORT=$PORT2 $AGENT2 &>/dev/null &
PID2=$!
echo "Started agent2 (PID $PID2) on port $PORT2"

sleep 1

echo "Running judge..."
PLAYER1_URL=http://localhost:$PORT1 PLAYER2_URL=http://localhost:$PORT2 uv run judge_engine.py 2>&1 | tail -20

cleanup
