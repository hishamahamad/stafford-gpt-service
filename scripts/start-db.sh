#!/bin/bash

# Remove any old DB container
docker rm -f rag-db 2>/dev/null || true

# Create network if it doesn't exist
docker network create rag-network 2>/dev/null || true

# Run Postgres+pgvector, mounting only the init SQL
docker run -d \
  --name rag-db \
  --network rag-network \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=example \
  -e POSTGRES_DB=ragdb \
  -v "$(pwd)/db/init":/docker-entrypoint-initdb.d:ro \
  -v pgdata:/var/lib/postgresql/data \
  -p 5432:5432 \
  pgvector/pgvector:0.7.4-pg15

echo "Database started on port 5432"
