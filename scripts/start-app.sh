#!/bin/bash

echo "🚀 Starting RAG Service..."

# Stop any existing containers
docker stop rag-app rag-db 2>/dev/null || true
docker rm rag-app rag-db 2>/dev/null || true

# Create network
docker network create rag-network 2>/dev/null || true

# Start database
echo "📊 Starting database..."
docker run -d \
  --name rag-db \
  --network rag-network \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=example \
  -e POSTGRES_DB=ragdb \
  -v "$(pwd)/db/init":/docker-entrypoint-initdb.d:ro \
  -v ragdb-data:/var/lib/postgresql/data \
  -p 5432:5432 \
  pgvector/pgvector:0.7.4-pg15

# Wait for database
echo "⏳ Waiting for database to be ready..."
for i in {1..30}; do
  if docker exec rag-db pg_isready -U postgres -d ragdb > /dev/null 2>&1; then
    echo "✅ Database is ready!"
    break
  fi
  echo "  Attempt $i/30..."
  sleep 2
done

# Build and start app
echo "🔨 Building application (this may take a few minutes for Playwright)..."
docker build -t rag-app:latest .

echo "🚀 Starting application..."
docker run -d \
  --name rag-app \
  --network rag-network \
  -e DATABASE_URL=postgresql://postgres:example@rag-db:5432/ragdb \
  --env-file .env \
  -p 8000:8000 \
  rag-app:latest

# Wait for app to start
echo "⏳ Waiting for application to start..."
for i in {1..15}; do
  if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Application is ready!"
    break
  fi
  echo "  Attempt $i/15..."
  sleep 2
done

echo ""
echo "🎉 RAG Service is running!"
echo "📍 API: http://localhost:8000"
echo "📖 Docs: http://localhost:8000/docs"
echo "🗄️  Database: localhost:5432"
echo ""
echo "To stop: ./scripts/stop-all.sh"