from fastapi import FastAPI, HTTPException, File, UploadFile, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import asyncio
import logging
import json
import os
import uuid
from typing import Optional, List
from datetime import datetime, timezone
from pathlib import Path

import httpx

from app.config import get_settings
from app.models import (
    QueryRequest,
    QueryResponse,
    DocumentUploadResponse,
    DocumentStatus,
    DocumentStatusResponse,
    HealthCheckResponse,
)
from app.services.document_service import document_service
from app.services.rag_service import rag_service
from app.services.cache_service import cache_service
from app.services.document_registry import document_registry

settings = get_settings()

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up application...")
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    await cache_service.connect()
    await document_service.initialize()
    await rag_service.initialize(document_service.vector_store)
    logger.info("Application started successfully")
    yield
    logger.info("Shutting down application...")
    await cache_service.disconnect()
    if document_service.rabbitmq_connection:
        await document_service.rabbitmq_connection.close()
    logger.info("Application shutdown complete")


app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan,
    description=(
        "PDF document question answering using LangChain, PostgreSQL/pgvector, "
        "and a local open-source LLM via Ollama."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


async def _ollama_status() -> str:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(settings.OLLAMA_BASE_URL)
            return "healthy" if response.status_code < 500 else "unhealthy"
    except Exception:
        return "unhealthy"


@app.get("/health", response_model=HealthCheckResponse, tags=["system"])
async def health_check():
    services_status = {
        "cache": "healthy" if cache_service.redis_client else "unhealthy",
        "vector_store": "healthy" if document_service.vector_store else "unhealthy",
        "llm": await _ollama_status(),
    }
    overall = "healthy" if all(s == "healthy" for s in services_status.values()) else "degraded"
    return HealthCheckResponse(
        status=overall,
        version=settings.API_VERSION,
        timestamp=datetime.now(timezone.utc),
        services=services_status,
    )


@app.post("/query", response_model=QueryResponse, tags=["query"])
async def query_documents(request: QueryRequest):
    try:
        session_id = request.session_id or str(uuid.uuid4())
        return await rag_service.query(
            query=request.query,
            session_id=session_id,
            filters=request.filters,
            top_k=request.top_k or settings.TOP_K_RETRIEVAL,
            document_id=request.document_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Query processing failed. Check that the document is indexed and dependent services are healthy.",
        )


@app.post("/documents", response_model=DocumentUploadResponse, tags=["ingestion"])
async def upload_document(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="A file name is required")

    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in settings.ALLOWED_FILE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type {file_ext} not allowed. Use: {', '.join(settings.ALLOWED_FILE_TYPES)}",
        )

    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    if file_size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds {settings.MAX_FILE_SIZE_MB}MB limit",
        )
    if file_size == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        metadata_dict = json.loads(metadata) if metadata else {}
        if not isinstance(metadata_dict, dict):
            raise ValueError("metadata must be a JSON object")
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid metadata: {e}") from e

    try:
        document_id = await document_service.upload_document(
            file=file.file,
            file_name=file.filename,
            metadata=metadata_dict,
        )
        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            status=DocumentStatus.PENDING,
            message="Document queued for processing",
            estimated_processing_time_seconds=60,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Document upload failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document upload failed. Check that dependent services are healthy and try again.",
        )


@app.get("/documents", response_model=List[DocumentStatusResponse], tags=["ingestion"])
async def list_documents():
    try:
        return await asyncio.to_thread(document_registry.list_all)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Listing documents failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve documents. Check that the database is reachable.",
        )


@app.get("/documents/{document_id}", response_model=DocumentStatusResponse, tags=["ingestion"])
async def get_document(document_id: str):
    try:
        record = await asyncio.to_thread(document_registry.get, document_id)
    except Exception:
        logger.exception("Fetching document %s failed", document_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve document. Check that the database is reachable.",
        )
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")
    return record


@app.delete("/documents/{document_id}", tags=["ingestion"])
async def delete_document(document_id: str):
    success = await document_service.delete_document(document_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    await cache_service.clear_pattern("query:*")
    return {"message": "Document deleted successfully", "document_id": document_id}


@app.delete("/sessions/{session_id}", tags=["query"])
async def clear_session(session_id: str):
    await rag_service.clear_session(session_id)
    return {"message": "Session cleared successfully", "session_id": session_id}


@app.get("/", include_in_schema=False)
async def serve_ui():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(index)


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        workers=1,
    )
