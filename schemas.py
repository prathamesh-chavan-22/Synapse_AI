from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class QueryRequest(BaseModel):
    query_text: str = Field(..., description="The user's query text")
    top_k: Optional[int] = Field(default=5, description="Number of top results to retrieve", ge=1, le=20)


class Source(BaseModel):
    citation_number: int = Field(..., description="Citation number for referencing")
    filename: str = Field(..., description="Name of the source file")
    content_type: str = Field(..., description="Type of content: pdf_page, image_caption, audio_transcript, etc.")
    details: Dict[str, Any] = Field(..., description="Additional details like page number, timestamp, text snippet")


class QueryResponse(BaseModel):
    query_text: str = Field(..., description="The original query text")
    answer: str = Field(..., description="The generated answer with citations")
    sources: List[Source] = Field(..., description="List of sources used in the answer")


class IngestResponse(BaseModel):
    status: str = Field(..., description="Status of the ingestion: success or error")
    filename: str = Field(..., description="Name of the ingested file")
    message: str = Field(..., description="Detailed message about the ingestion result")
    chunks_created: Optional[int] = Field(None, description="Number of chunks created from the file")


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message")
