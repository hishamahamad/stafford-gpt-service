"""
Stafford GPT Service - Main Application
A clean, organized backend for Stafford Global's intelligent student advisor.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.stafford_gpt.config.settings import settings
from src.stafford_gpt.database.connection import get_database_connection
from src.stafford_gpt.services.data_access import DataAccessService
from src.stafford_gpt.services.orchestrator import OrchestratorService
from src.stafford_gpt.api.query import create_query_router
from src.stafford_gpt.api.ingestion import create_ingestion_router

# Create FastAPI app
app = FastAPI(
    title="Stafford GPT Service",
    description="Stafford Global's intelligent student advisor with canonical data architecture",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize database connection and services
try:
    db_connection = get_database_connection()
    
    # Create services (canonical data only)
    data_access = DataAccessService(db_connection)
    orchestrator = OrchestratorService(data_access)

    print("✅ Database and services initialized successfully")
    print("🎯 Running in canonical data mode - fast and reliable!")

except Exception as e:
    print(f"❌ Failed to initialize services: {e}")
    raise

# Include API routers
app.include_router(create_query_router(orchestrator))
app.include_router(create_ingestion_router(orchestrator))

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Stafford GPT Service",
        "mode": "canonical_data_only",
        "message": "Fast, reliable responses from structured program data"
    }
