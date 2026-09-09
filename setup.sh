#!/bin/bash
set -e
if ! command -v docker &> /dev/null; then
  echo "Docker is required"
  exit 1
fi
if [ ! -f .env ]; then
  cp .env.example .env
fi
docker compose up --build -d
echo "Stack is starting. UI: http://localhost:8000  Docs: http://localhost:8000/docs"
echo "First run pulls Ollama models and can take several minutes."
