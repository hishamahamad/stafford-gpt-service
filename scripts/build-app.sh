#!/bin/bash

# Build and restart Stafford GPT application
echo "Building Stafford GPT application..."

# Stop services if running
docker-compose down

# Remove existing images to force rebuild
DOCKER_BUILDKIT=0 docker-compose build --no-cache

# Start services
DOCKER_BUILDKIT=0 docker-compose up -d
