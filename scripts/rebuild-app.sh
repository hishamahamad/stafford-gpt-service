#!/bin/bash

# Rebuild and restart the unified RAG application
echo "Rebuilding RAG application..."

# Stop and remove existing container
docker stop rag-app 2>/dev/null || true
docker rm rag-app 2>/dev/null || true

# Remove existing image
docker rmi rag-app:latest 2>/dev/null || true

# Build new image
docker build -t rag-app:latest .

# Create network if it doesn't exist
docker network create rag-network 2>/dev/null || true

# Start the application
./scripts/start-app.sh

echo "RAG application rebuilt and restarted!"