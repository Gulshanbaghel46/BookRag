#!/bin/bash
# Usage: ./ask.sh "Your question here"
set -e
if [ -z "$1" ]; then
  echo "Usage: ./ask.sh \"Your question here\""
  exit 1
fi
QUESTION="$1"
curl -s -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"$QUESTION\", \"top_k\": 3}"
echo
