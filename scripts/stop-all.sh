#!/usr/bin/env bash
# Stop and remove all containers

echo "Stopping all RAG services..."

docker rm -f rag-app rag-db 2>/dev/null || true

echo "All services stopped and removed."