"""Query API endpoints for chat and search functionality."""

import json
from typing import Optional
from fastapi import APIRouter, Body, HTTPException

from ..models.requests import QueryRequest, QueryResponse
from ..services.orchestrator import OrchestratorService

router = APIRouter(prefix="/api", tags=["query"])


def create_query_router(orchestrator: OrchestratorService) -> APIRouter:
    """Create query router with orchestrator dependency."""

    @router.post("/query", response_model=QueryResponse)
    def query(
        q: str = Body(..., embed=True, description="The user's latest question"),
        session_id: Optional[str] = Body(None, description="UUID of the chat session")
    ):
        """
        Query the knowledge base using canonical program data.
        Uses only structured program information as the source of truth.
        """
        try:
            result = orchestrator.process_query(
                question=q,
                session_id=session_id
            )

            return QueryResponse(
                answer=result["answer"],
                session_id=result["session_id"],
                metadata={
                    "scope": result.get("scope"),
                    "program_variant_ids": result.get("program_variant_ids"),
                    "consistency_issues": result.get("consistency_issues", []),
                    "error": result.get("error"),
                }
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

    return router
