from datetime import datetime, timezone
from typing import List, Optional
import logging

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json

from app.config import get_settings
from app.models import DocumentStatus, DocumentStatusResponse

logger = logging.getLogger(__name__)
settings = get_settings()


def _connect():
    return psycopg.connect(settings.postgres_dsn, row_factory=dict_row)


class DocumentRegistry:
    def initialize(self) -> None:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        document_id TEXT PRIMARY KEY,
                        file_name TEXT NOT NULL,
                        status TEXT NOT NULL,
                        uploaded_at TIMESTAMPTZ NOT NULL,
                        processed_at TIMESTAMPTZ,
                        error_message TEXT,
                        chunk_count INTEGER,
                        metadata JSONB DEFAULT '{}'::jsonb
                    )
                    """
                )
            conn.commit()
        logger.info("Document registry ready")

    def create(
        self,
        document_id: str,
        file_name: str,
        metadata: Optional[dict] = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents
                        (document_id, file_name, status, uploaded_at, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        document_id,
                        file_name,
                        DocumentStatus.PENDING.value,
                        now,
                        Json(metadata or {}),
                    ),
                )
            conn.commit()

    def update_status(
        self,
        document_id: str,
        status: DocumentStatus,
        error_message: Optional[str] = None,
        chunk_count: Optional[int] = None,
    ) -> None:
        processed_at = (
            datetime.now(timezone.utc)
            if status in (DocumentStatus.COMPLETED, DocumentStatus.FAILED)
            else None
        )
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE documents
                    SET status = %s,
                        error_message = %s,
                        chunk_count = COALESCE(%s, chunk_count),
                        processed_at = COALESCE(%s, processed_at)
                    WHERE document_id = %s
                    """,
                    (
                        status.value,
                        error_message,
                        chunk_count,
                        processed_at,
                        document_id,
                    ),
                )
            conn.commit()

    def get(self, document_id: str) -> Optional[DocumentStatusResponse]:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM documents WHERE document_id = %s",
                    (document_id,),
                )
                row = cur.fetchone()
        return self._to_model(row) if row else None

    def list_all(self) -> List[DocumentStatusResponse]:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM documents ORDER BY uploaded_at DESC")
                rows = cur.fetchall()
        return [self._to_model(row) for row in rows]

    def delete(self, document_id: str) -> bool:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM documents WHERE document_id = %s",
                    (document_id,),
                )
                deleted = cur.rowcount > 0
            conn.commit()
        return deleted

    @staticmethod
    def _to_model(row: dict) -> DocumentStatusResponse:
        return DocumentStatusResponse(
            document_id=row["document_id"],
            status=DocumentStatus(row["status"]),
            file_name=row["file_name"],
            uploaded_at=row["uploaded_at"],
            processed_at=row["processed_at"],
            error_message=row["error_message"],
            chunk_count=row["chunk_count"],
        )


document_registry = DocumentRegistry()
