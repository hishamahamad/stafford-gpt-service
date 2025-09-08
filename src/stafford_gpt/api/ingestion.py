from fastapi import APIRouter, HTTPException

from ..models.requests import (
    ProgramDataRequest,
    IngestionResponse,
    ExtractRequest
)
from ..services.orchestrator import OrchestratorService
from ..services.program_extractor import create_program_data_extractor

router = APIRouter(prefix="/api/ingest", tags=["ingestion"])


def create_ingestion_router(orchestrator: OrchestratorService) -> APIRouter:
    extractor = create_program_data_extractor()

    @router.post("/program", response_model=IngestionResponse)
    async def ingest_program_data(request: ProgramDataRequest):
        try:
            success = orchestrator.ingest_enhanced_program_data(request.dict())
            if success:
                return IngestionResponse(
                    status="success",
                    message=f"Program data for {request.program_variant_id} ingested successfully",
                    details={
                        "program_variant_id": request.program_variant_id,
                        "university": request.basic_info.university_name,
                        "program": request.basic_info.program_name
                    }
                )
            else:
                raise HTTPException(status_code=400, detail="Failed to ingest program data")

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Program data ingestion failed: {str(e)}")

    @router.post("/extract", response_model=IngestionResponse)
    async def extract_program_data(request: ExtractRequest):
        try:
            # AI-powered data extraction using URL-based inference
            extracted_data = await extractor.extract_program_data(
                request.url,
                request.university_id,
                request.degree_type
            )

            return IngestionResponse(
                status="success",
                message=f"Successfully extracted program data from {request.url} - ready for review",
                details=extracted_data
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"AI extraction failed: {str(e)}")

    return router
