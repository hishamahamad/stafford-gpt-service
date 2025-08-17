#!/bin/bash

echo "Starting all RAG services..."

# Start database first
./scripts/start-db.sh

# Wait for database to be ready
echo "Waiting for database to be ready..."
sleep 5

# Start the application
./scripts/start-app.sh

echo "All services started!"
echo "Database: localhost:5432"
echo "API: localhost:8000"
echo "API docs: http://localhost:8000/docs"