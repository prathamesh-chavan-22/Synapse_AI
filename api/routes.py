from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
import logging
import os
import tempfile
from typing import Dict, Any

from schemas import QueryRequest, QueryResponse, IngestResponse, ErrorResponse
from services.ingestion_service import ingestion_service_instance
from services.query_service import query_service_instance

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["RAG System"])


@router.post(
    "/ingest",
    response_model=IngestResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Ingest and index a file",
    description="Upload a file (PDF, DOCX, TXT, Image, Audio) to be processed and indexed in the vector database."
)
async def ingest_file(file: UploadFile = File(...)) -> IngestResponse:
    temp_file_path = None

    try:
        logger.info(f"Received file for ingestion: {file.filename}")

        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Filename is required"
            )

        allowed_extensions = ['.pdf', '.docx', '.doc', '.txt', '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.mp3', '.wav', '.m4a', '.flac', '.ogg']
        file_extension = os.path.splitext(file.filename)[1].lower()

        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type: {file_extension}. Allowed types: {', '.join(allowed_extensions)}"
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        result = await ingestion_service_instance.ingest_file(temp_file_path, file.filename)

        return IngestResponse(**result)

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Error during file ingestion: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest file: {str(e)}"
        )

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {str(e)}")


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Query the RAG system",
    description="Submit a natural language query and receive an answer grounded in the indexed documents with source citations."
)
async def query_system(request: QueryRequest) -> QueryResponse:
    try:
        logger.info(f"Received query: {request.query_text[:100]}...")

        if not request.query_text or not request.query_text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query text cannot be empty"
            )

        if request.top_k and (request.top_k < 1 or request.top_k > 20):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="top_k must be between 1 and 20"
            )

        result = await query_service_instance.process_query(
            query_text=request.query_text,
            top_k=request.top_k or 5
        )

        return result

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Error during query processing: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process query: {str(e)}"
        )


@router.get(
    "/health",
    summary="Health check",
    description="Check if the API is running and responsive"
)
async def health_check() -> Dict[str, str]:
    return {"status": "healthy", "message": "Multimodal RAG System is running"}


@router.get(
    "/stats",
    summary="Get database statistics",
    description="Get statistics about the indexed documents in the vector database"
)
async def get_stats() -> Dict[str, Any]:
    try:
        from core.vector_db import vector_db_instance

        stats = vector_db_instance.get_collection_stats()
        return stats

    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve statistics: {str(e)}"
        )
