"""Query API endpoints for chat and search functionality."""

from typing import Optional
from fastapi import APIRouter, Body, HTTPException, Path

from ..models.requests import QueryResponse, ProgramDetailsResponse, AllProgramsResponse
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

    @router.get("/programs/{variant_id}", response_model=ProgramDetailsResponse)
    def get_program_details(
        variant_id: str = Path(..., description="The program variant ID to fetch details for")
    ):
        """
        Fetch program details by variant ID.
        Returns comprehensive program information from the canonical database.
        """
        try:
            # Use the data access service to get program data
            program_data = orchestrator.data_access.get_canonical_data([variant_id])

            if not program_data or variant_id not in program_data:
                raise HTTPException(
                    status_code=404,
                    detail=f"Program with variant ID '{variant_id}' not found"
                )

            data = program_data[variant_id]

            return ProgramDetailsResponse(
                program_variant_id=data["program_variant_id"],
                university_id=data["university_id"],
                university_name=data["university_name"],
                program_identifier=data["program_identifier"],
                program_type=data["program_type"],
                program_name=data["program_name"],
                basic_info=data.get("basic_info"),
                duration=data.get("duration"),
                fees=data.get("fees"),
                intake_info=data.get("intake_info"),
                curriculum=data.get("curriculum"),
                accreditation=data.get("accreditation"),
                career_outcomes=data.get("career_outcomes"),
                entry_requirements=data.get("entry_requirements"),
                geographic_focus=data.get("geographic_focus"),
                academic_progression=data.get("academic_progression"),
                completion_rates=data.get("completion_rates"),
                support_services=data.get("support_services"),
                status="success"
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch program details: {str(e)}"
            )

    @router.get("/programs", response_model=AllProgramsResponse)
    def get_all_programs():
        """
        Fetch all available programs.
        Returns a list of all active programs with their basic information.
        """
        try:
            # Use the data access service to get all programs
            all_programs_data = orchestrator.data_access.get_all_programs()

            # Convert to response format
            programs = []
            for data in all_programs_data:
                program = ProgramDetailsResponse(
                    program_variant_id=data["program_variant_id"],
                    university_id=data["university_id"],
                    university_name=data["university_name"],
                    program_identifier=data["program_identifier"],
                    program_type=data["program_type"],
                    program_name=data["program_name"],
                    basic_info=data.get("basic_info"),
                    duration=data.get("duration"),
                    fees=data.get("fees"),
                    status="success"
                )
                programs.append(program)

            return AllProgramsResponse(
                programs=programs,
                total_count=len(programs),
                status="success"
            )

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch programs: {str(e)}"
            )

    return router
