#!/bin/bash
# Benchmark llama-server performance on Strix Halo
# Usage: ./test-llama.sh [port]
# Default port: 8080

set -e

PORT="${1:-8080}"
BASE="http://localhost:${PORT}"

echo "============================================"
echo " llama.cpp Benchmark Suite (Strix Halo)"
echo " Target: ${BASE}"
echo " Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# Wait for server to be ready
echo -n "Waiting for server..."
for i in $(seq 1 30); do
  if curl -s "${BASE}/health" 2>/dev/null | grep -q "ok"; then
    echo " ready."
    break
  fi
  if [ "$i" -eq 30 ]; then
    echo " FAILED (server not responding)"
    exit 1
  fi
  sleep 1
done

# Warmup (2 requests to ensure caches are hot)
echo -n "Warming up..."
curl -s "${BASE}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"coder-qwen","messages":[{"role":"user","content":"Hi"}],"max_tokens":10}' > /dev/null
sleep 1
curl -s "${BASE}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"coder-qwen","messages":[{"role":"user","content":"Hello"}],"max_tokens":10}' > /dev/null
sleep 1
echo " done."

echo ""
echo "Running benchmarks..."
echo ""

# Collect server log lines before tests to find only new timing entries
LOG_MARKER="BENCHMARK_START_$(date +%s%N)"

run_test() {
  local label="$1"
  local prompt="$2"
  local max_tokens="$3"

  curl -s "${BASE}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"coder-qwen\",\"messages\":[{\"role\":\"user\",\"content\":\"${prompt}\"}],\"max_tokens\":${max_tokens}}" > /dev/null

  echo "  ${label}: done"
}

# Count existing log lines so we only parse new ones
if [ -f /tmp/llama-server.log ]; then
  LINES_BEFORE=$(wc -l < /tmp/llama-server.log)
else
  LINES_BEFORE=0
fi

echo "--- Test 1: Short prompt, 100 token generation ---"
run_test "Small" "Write a Python function that checks if a number is prime. Keep it short." 100

echo "--- Test 2: Short prompt, 500 token generation ---"
run_test "Medium" "Write a Python class that implements a binary search tree with insert, search, and delete methods. Include docstrings." 500

echo "--- Test 3: Long prompt (~151 tokens in), 100 token generation ---"
run_test "Long prompt" "I need you to review the following code and explain each function in detail. Here is a comprehensive Python module that handles user authentication, session management, database connections, caching, rate limiting, and API endpoint routing. The module includes decorators for authentication, middleware for request processing, utility functions for password hashing and token generation, database models for users and sessions, configuration management with environment variables, logging setup with rotation, error handling with custom exceptions, input validation with regex patterns, response formatting with JSON serialization, and background task scheduling with async workers. Please analyze the architecture, identify potential security issues, suggest performance improvements, and recommend best practices for each component. Also explain how the components interact with each other and what design patterns are being used." 100

echo ""
echo "============================================"
echo " Results"
echo "============================================"

# Parse timing lines from server log
if [ -f /tmp/llama-server.log ]; then
  TIMING_LINES=$(tail -n +$((LINES_BEFORE + 1)) /tmp/llama-server.log | grep "eval time")

  TEST_NUM=0
  LABELS=("Small (100 tok out)" "Medium (500 tok out)" "Long prompt (100 tok out)")

  while IFS= read -r line; do
    if echo "$line" | grep -q "^prompt"; then
      PROMPT_LINE="$line"
    elif echo "$line" | grep -q "^       eval"; then
      GEN_LINE="$line"

      # Extract tok/s values
      PROMPT_TPS=$(echo "$PROMPT_LINE" | grep -oP '[\d.]+(?= tokens per second)')
      GEN_TPS=$(echo "$GEN_LINE" | grep -oP '[\d.]+(?= tokens per second)')
      GEN_TOKENS=$(echo "$GEN_LINE" | grep -oP '(?<=/ )\s*\d+(?= tokens)')
      PROMPT_TOKENS=$(echo "$PROMPT_LINE" | grep -oP '(?<=/ )\s*\d+(?= tokens)')

      if [ $TEST_NUM -lt ${#LABELS[@]} ]; then
        printf "  %-28s  prompt: %6s tok/s (%s tok)  gen: %6s tok/s (%s tok)\n" \
          "${LABELS[$TEST_NUM]}" "$PROMPT_TPS" "$PROMPT_TOKENS" "$GEN_TPS" "$GEN_TOKENS"
      fi
      TEST_NUM=$((TEST_NUM + 1))
    fi
  done <<< "$TIMING_LINES"
else
  echo "  (no log file at /tmp/llama-server.log — start server with: bash ~/llama/llama-serve.sh &>/tmp/llama-server.log &)"
fi

echo ""
echo "============================================"
