#!/bin/bash
set -e
echo "Starting Document Q&A Assistant — RAG Pipeline with Open-Source LLM"
if [ ! -f .env ]; then
  cp .env.example .env
fi
docker compose up --build
