import asyncio
from typing import Optional, BinaryIO
from datetime import datetime, timezone
from pathlib import Path
import uuid
import json
import logging
import aio_pika
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector

from app.config import get_settings
from app.models import DocumentStatus
from app.services.document_registry import document_registry, _connect

logger = logging.getLogger(__name__)
settings = get_settings()


def _delete_vectors_sql(document_id: str) -> None:
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM langchain_pg_embedding
                WHERE cmetadata->>'document_id' = %s
                """,
                (document_id,),
            )
        conn.commit()


class DocumentService:
    def __init__(self):
        if settings.LLM_PROVIDER == "ollama":
            self.embeddings = OllamaEmbeddings(
                model=settings.EMBEDDING_MODEL,
            )
            logger.info("Using Ollama embeddings: %s", settings.EMBEDDING_MODEL)
        else:
            self.embeddings = OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                openai_api_key=settings.OPENAI_API_KEY,
            )
            logger.info("Using OpenAI embeddings: %s", settings.EMBEDDING_MODEL)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.vector_store: Optional[PGVector] = None
        self.rabbitmq_connection: Optional[aio_pika.RobustConnection] = None
        self.rabbitmq_channel: Optional[aio_pika.Channel] = None
        Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

    async def initialize(self):
        try:
            await asyncio.to_thread(document_registry.initialize)

            self.vector_store = PGVector(
                embeddings=self.embeddings,
                connection=settings.database_url,
                collection_name="documents",
                use_jsonb=True,
            )

            for attempt in range(1, 7):
                try:
                    self.rabbitmq_connection = await aio_pika.connect_robust(
                        settings.RABBITMQ_URL
                    )
                    self.rabbitmq_channel = await self.rabbitmq_connection.channel()
                    await self.rabbitmq_channel.declare_queue(
                        "document_processing", durable=True
                    )
                    break
                except Exception:
                    if attempt == 6:
                        raise
                    logger.warning(
                        "RabbitMQ connection attempt %s/6 failed; retrying", attempt
                    )
                    await asyncio.sleep(attempt)

            logger.info("Document service initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize document service: %s", e)
            raise

    async def upload_document(
        self,
        file: BinaryIO,
        file_name: str,
        metadata: dict,
    ) -> str:
        document_id = str(uuid.uuid4())
        safe_name = Path(file_name).name
        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / f"{document_id}_{safe_name}"

        file.seek(0)
        file_path.write_bytes(file.read())

        await asyncio.to_thread(
            document_registry.create,
            document_id,
            safe_name,
            metadata,
        )

        message = {
            "document_id": document_id,
            "file_path": str(file_path),
            "file_name": safe_name,
            "metadata": metadata,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }

        await self.rabbitmq_channel.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps(message).encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            ),
            routing_key="document_processing",
        )

        logger.info("Document %s queued for processing", document_id)
        return document_id

    def _load_documents(self, file_path: str):
        suffix = Path(file_path).suffix.lower()
        if suffix == ".pdf":
            return PyPDFLoader(file_path).load()
        if suffix in {".txt", ".md"}:
            return TextLoader(file_path, encoding="utf-8").load()
        if suffix == ".docx":
            return Docx2txtLoader(file_path).load()
        raise ValueError(f"Unsupported file type: {suffix}")

    async def process_document(
        self,
        document_id: str,
        file_path: str,
        file_name: str,
        metadata: dict,
    ):
        try:
            logger.info("Processing document %s", document_id)
            await asyncio.to_thread(
                document_registry.update_status,
                document_id,
                DocumentStatus.PROCESSING,
            )

            documents = await asyncio.to_thread(self._load_documents, file_path)
            if not documents:
                raise ValueError("No text could be extracted from the document")

            for doc in documents:
                doc.metadata.update(
                    {
                        "document_id": document_id,
                        "file_name": file_name,
                        **(metadata or {}),
                    }
                )

            chunks = self.text_splitter.split_documents(documents)
            total_chunks = len(chunks)
            if total_chunks == 0:
                raise ValueError("Document produced no text chunks")

            logger.info("Document %s split into %s chunks", document_id, total_chunks)

            batch_size = 10
            for i in range(0, total_chunks, batch_size):
                batch = chunks[i : i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (total_chunks + batch_size - 1) // batch_size
                logger.info(
                    "Processing batch %s/%s (%s chunks)",
                    batch_num,
                    total_batches,
                    len(batch),
                )
                await asyncio.to_thread(self.vector_store.add_documents, batch)

            await asyncio.to_thread(
                document_registry.update_status,
                document_id,
                DocumentStatus.COMPLETED,
                None,
                total_chunks,
            )
            logger.info("Document %s processed successfully (%s chunks)", document_id, total_chunks)
        except Exception as e:
            logger.error("Document processing failed for %s: %s", document_id, e)
            await asyncio.to_thread(
                document_registry.update_status,
                document_id,
                DocumentStatus.FAILED,
                str(e),
            )
            raise

    async def delete_document(self, document_id: str) -> bool:
        try:
            record = await asyncio.to_thread(document_registry.get, document_id)
            if not record:
                return False

            if self.vector_store:
                try:
                    await self.vector_store.adelete(filter={"document_id": document_id})
                except Exception as exc:
                    logger.warning("Vector delete via LangChain failed for %s: %s", document_id, exc)
                    await asyncio.to_thread(_delete_vectors_sql, document_id)

            await asyncio.to_thread(document_registry.delete, document_id)

            upload_dir = Path(settings.UPLOAD_DIR)
            for leftover in upload_dir.glob(f"{document_id}_*"):
                leftover.unlink(missing_ok=True)

            logger.info("Document %s deleted successfully", document_id)
            return True
        except Exception as e:
            logger.error("Document deletion failed: %s", e)
            return False


document_service = DocumentService()
